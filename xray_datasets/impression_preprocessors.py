"""
Module for the impression preprocessors for medklip and medkebert.
"""

import copy
import json
import os
from os import path
from pathlib import Path
import re

import pandas as pd

ENTITY_LABEL_DICT = {
    'ANAT-DP': 'Anatomy Definitely Present',
    'OBS-DP': 'Observation Definitely Present',
    'OBS-DA': 'Observation Definitely Absent',
    'OBS-U': 'Observation Uncertain',
}


class MedKLIPPreprocessor:
    """
    Processor for the impressions of medklip.
    """

    def __init__(self, data_dir,
                 explanation_file= "$WORK/cxr_phrase_grounding/xray_datasets/observation_explanation.json"):
        """
        :param data_dir: Directory containing the preprocessed radgraph files.
        :param explanation_file: Path to a json file explaining medical keywords.
        """
        self.radgraph_files = {}
        for i in range(10, 20):
            radgraph_file = pd.read_csv(path.join(data_dir, f"p{i}_parsed_modified.csv"))
            self.radgraph_files[f"p{i}"] = radgraph_file
        with open(os.path.expandvars(explanation_file), 'r', encoding="utf-8") as loaded_file:
            self.obs_expl = json.load(loaded_file)

    def preprocess_impressions(self, samples):
        """
        Replaces the impressions of the passed samples with a triplet of disease explanation, location and presence.
        :param samples: Dictionary of data samples to evaluate.
        :return: The samplres with replaced impressions.
        """
        for sample in samples:
            path_parts = sample["rel_path"].split('/')
            p = path_parts[1]
            subject_id = int(path_parts[2][1:])
            study_id = int(path_parts[3][1:])
            meta_data = self.radgraph_files[p]
            relevant_data = meta_data.loc[
                (meta_data["study_id"] == int(study_id)) & (meta_data["subject_id"] == int(subject_id))]
            relevant_data = relevant_data.dropna(subset=['observation'])
            relevant_data.reset_index()
            impression = ""
            explanation_set = set()
            for _, row in relevant_data.iterrows():
                explanation = self.obs_expl.get(str(row['observation']).lower().replace(" ", "_"), None)
                if explanation is None or explanation in explanation_set:
                    continue
                explanation_set.add(explanation)
                impression += f"{row['observation']}: {explanation} "
                observations = ' and '.join(row['obs_lemma'].split('|'))
                if observations != "unspecified":
                    impression += f"It is located at {observations}. "
                impression += f"{ENTITY_LABEL_DICT[row['label']]}. "
            if impression:
                sample["impression"] = impression
            else:
                if "label_text" in sample:
                    sample["impression"] = sample["label_text"]
        return samples


class MedKeBERTPreprocessor:
    """
    Loads radbert knowledge graphs and impressions preprocessed by scispacy from json files, in order to preprocess
    the impressions for med-kebert.
    """

    def __init__(self, data_path, csv_path):
        """
        :param data_path: Path to the directory containing the files preprocessed by scispacy and the files
               preprocessed by radgraph. Names should be p10.json - p19.json for radgraph and
               p10_structured_entities,json - p19_structured_entities.json for radgraph.
        :param csv_path: Path to the metadata csv file of the main MIMIC dataset.
        """
        self.data_path = data_path
        self.csv_path = csv_path

    def _create_lookup_table(self, dicts, key):
        """
        Creates dictionary that maps dictionary values of a key to their corresponding dictionaries.
        :param dicts: List of dictionaries that should be looked up.
        :param key: Key for the dictionary values that should be used to look up the dictionary.
        :return: Created lookup dictionary.
        """
        lookup_table = {}
        for d in dicts:
            lookup_table[Path(d[key]).stem] = d
        return lookup_table

    def _extract_path_segment(self, path_w_segments, pattern):
        """
        Helper method used to extract a pattern from a path.
        :param path_w_segments: Path from which a segment should be extracted.
        :param pattern: Pattern to extract.
        :return: None if no match could be found, otherwise the extracted segment.
        """
        # Search for the pattern in the path
        match = re.search(pattern, path_w_segments)
        if match:
            # Return the path up to the end of the matched pattern
            return path_w_segments[:match.end()]
        return None

    def preprocess_impressions(self, samples):
        """
        Based on https://arxiv.org/abs/2302.14042. Overwrites the impression of the samples with radgraph entities that
        have been among the most common found spacy entities. Results in such a pattern:
        ENTITY PRESENCE ... ENTITY PRESENCE [SEP] ... ENTITY PRESENCE [SEP]. Might not affect an impression if scispacy
        did not find a single relevant entity, which happens for one impression entry in the MS-CXR dataset.
        :param samples: Dictionary of data samples.
        :return: Samples with overwritten impressions.
        """
        umls_info, radgraph_json_info = self._load_data_info()
        lookup_table = self._create_lookup_table(umls_info, 'file_path')
        radgraph_json_info = {Path(k).stem: v for k, v in radgraph_json_info.items()}

        samples = self._process_samples(samples, lookup_table, radgraph_json_info)
        return samples

    def _load_data_info(self):
        """
        Loads data from JSON files for UMLS and RadGraph based on the patient list.
        :return: Tuple containing UMLS info list and RadGraph JSON info dictionary.
        """
        umls_info = []
        radgraph_json_info = {}
        p_list = self._determine_patient_list()
        for p in p_list:
            radgraph_json_info.update(self._load_json(p + ".json"))
            umls_info.extend(self._load_json(p + "_structured_entities.json"))
        return umls_info, radgraph_json_info

    def _determine_patient_list(self):
        """
        Determines the list of patients based on the data file.
        :return: List of patient identifiers.
        """
        data_info = pd.read_csv(os.path.expandvars(self.csv_path))
        return data_info['p'].unique() if 'p' in data_info else ["p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17",
                                                                 "p18", "p19"]

    def _load_json(self, filename):
        """
        Loads JSON data from a specified file.
        :param filename: The filename of the JSON file.
        :return: JSON object loaded from the file.
        """
        with open(os.path.expandvars(os.path.join(self.data_path, filename)), 'r', encoding="utf-8") as file:
            return json.load(file)

    def _process_samples(self, samples, lookup_table, radgraph_json_info):
        """
        Processes each sample, updating the impression field based on entity information.
        :param samples: Dictionary or list of data samples.
        :param lookup_table: Lookup table containing UMLS information.
        :param radgraph_json_info: Dictionary containing RadGraph entities.
        :return: Dictionary of samples with updated impressions.
        """
        original_samples = copy.deepcopy(samples) if isinstance(samples, dict) else None
        samples = self._initialize_samples(samples)
        for sample in samples:
            sample["impression"] = self._construct_impression(sample, lookup_table, radgraph_json_info)

        if original_samples:
            samples = self._reintegrate_samples(samples, original_samples)
        return samples

    def _initialize_samples(self, samples):
        """
        Initializes the sample structure for processing.
        :param samples: Original dictionary of data samples.
        :return: List of dictionaries with relative path and impression.
        """
        if isinstance(samples, dict):
            return [{'rel_path': rel_path, 'impression': impression} for rel_path, impression in
                    zip(samples['rel_path'], samples["impression"])]
        return samples

    def _construct_impression(self, sample, lookup_table, radgraph_json_info):
        """
        Constructs the new impression for a sample based on entity information.
        :param sample: Single sample dictionary.
        :param lookup_table: Lookup table for UMLS entities.
        :param radgraph_json_info: Dictionary of RadGraph entities.
        :return: Updated impression string or a warning message.
        """
        lookup_path = Path(self._extract_path_segment(sample["rel_path"], 's[0-9]+')).stem
        umls = lookup_table.get(lookup_path)
        sentences = umls["sentences"] if umls else []

        try:
            radgraph_entities = radgraph_json_info[lookup_path]['entities']
            radgraph_list = [(entity["tokens"], ENTITY_LABEL_DICT[entity["label"]]) for entity in
                             radgraph_entities.values()]
            entity_info = self._integrate_entities(sentences, radgraph_list)
        except KeyError:
            entity_info = self._integrate_entities(sentences, [])

        return entity_info if entity_info else f"WARNING: no entities could be extracted from {sample['rel_path']}"

    def _integrate_entities(self, sentences, radgraph_list):
        """
        Integrates RadGraph entities into the existing text of sentences.
        :param sentences: List of sentences from UMLS data.
        :param radgraph_list: List of tuples containing RadGraph entity tokens and labels.
        :return: String with integrated entity information.
        """
        entity_info = ""
        for sentence in sentences:
            if sentence["entities"]:
                for entity in sentence['entities']:
                    match = next((rg for rg in radgraph_list if rg[0] in entity["text"]), None)
                    entity["presence"] = match[1] if match else entity["presence"]
                    entity_info += entity["text"] + " " + entity["presence"] + " "
                entity_info += " [SEP] "
        return entity_info

    def _reintegrate_samples(self, modified_samples, original_samples):
        """
        Reintegrates the modified samples back into their original order based on relative paths.
        :param modified_samples: List of modified samples.
        :param original_samples: Original dictionary of samples before modifications.
        :return: Dictionary of samples with updated impressions in original order.
        """
        rank = {value: idx for idx, value in enumerate(original_samples['rel_path'])}
        modified_samples = sorted(modified_samples, key=lambda sample: rank[sample['rel_path']])
        original_samples["impression"] = [sample['impression'] for sample in modified_samples]
        return original_samples
