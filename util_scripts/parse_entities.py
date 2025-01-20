"""
Preprocessing of impressions for a knowledge enhanced text encoder as described in https://arxiv.org/abs/2302.14042,
using scispacy and radgraph.
Tries to be mostly faithful to the description in the paper, as far as it made sense in the context of text
conditioning.
"""

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

# Load the spaCy model and add the EntityLinker pipe
spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_scibert")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

entity_counter = Counter()

# Define the base path and target directories
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=False,
                    help='Base path of report files')
parser.add_argument('--output_path', type=str, required=True,
                    help='Path of directory where to store the entity files')
parser.add_argument('--csv_file', type=str, required=False,
                    help='Path to CSV file containing list of file paths')
args = parser.parse_args()

# Get file paths from CSV or directories
file_list = []
if args.csv_file:
    df = pd.read_csv(args.csv_file)
    paths = df['paths'].tolist()
    for path in paths:
        base_path = os.path.join(Path(args.input_path).parent, os.path.dirname(path))
        new_path = base_path + '.txt'
        file_list.append(new_path)
else:
    target_dirs = [f"p{num}" for num in range(10, 20)]  # Creates a list of directories from p10 to p19
    for directory in target_dirs:
        path = Path(args.input_path) / directory
        file_list += [str(item) for item in glob.glob(f"{path}/**/*.txt", recursive=True)]

all_documents = []
text_data = []
file_paths = []
for i, file_path in enumerate(file_list):
    if i % 100 == 0:
        print(f"Loaded {i} files")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data.append(file.read())
            file_paths.append(file_path)
    except UnicodeDecodeError as e:
        print(e.reason)
        print(f"Encoding error at {file_path}")

for doc, file_path in tqdm(zip(nlp.pipe(text_data, batch_size=500), file_paths), total=len(text_data),
                           desc="Processing Documents"):
    filtered_entities = []
    current_presence = {}

    for entity in doc.ents:
        t_list = []
        if entity._.kb_ents:
            t_list = linker.kb.cui_to_entity[entity._.kb_ents[0][0]].types
        if "T033" in t_list or "T047" in t_list:
            filtered_entities.append(entity)
            entity_counter[entity.text] += 1

    # Update the most common entities set
    most_common_entity_names = {entity for entity, count in entity_counter.most_common(80)}
    entity_presence = {name: "absent" for name in most_common_entity_names}

    document_entities = [
        {"text": ent.text, "start_char": ent.start_char, "end_char": ent.end_char} for ent in filtered_entities if
        ent.text in most_common_entity_names
    ]

    sentences_data = []
    for sent in doc.sents:
        contains_key_words = any(word in sent.text.lower().split() for word in ["no", "normal", "none"])

        if contains_key_words:
            sentence_entities = [
                {"text": ent.text, "start_char": ent.start_char, "end_char": ent.end_char, "presence": "absent"} for ent
                in sent.ents if ent in filtered_entities and ent.text in most_common_entity_names
            ]
        else:
            sentence_entities = [
                {"text": ent.text, "start_char": ent.start_char, "end_char": ent.end_char, "presence": "present"} for
                ent in sent.ents if ent in filtered_entities and ent.text in most_common_entity_names
            ]
            for ent in sentence_entities:
                current_presence[ent['text']] = "present"

        sentences_data.append({"sentence_text": sent.text, "entities": sentence_entities})

    if "present" not in entity_presence.values():
        entity_presence["normal"] = "present"

    document_info = {
        "file_path": file_path,
        "document_entities": document_entities,
        "sentences": sentences_data,
        "entity_presence": current_presence
    }
    all_documents.append(document_info)

# Save to JSON file
output_path = os.path.join(args.output_path, "structured_entities.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_documents, f, ensure_ascii=False, indent=4)

print("Finished processing files")
