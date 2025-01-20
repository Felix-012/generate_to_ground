# Reproduce Paper

By following these steps, the results in the paper can be reproduced.

## Setup Conda Environment

To reproduce our conda environment, run`conda env create --name=cxr_phrase_grounding --file=environment.yml`
in the main directory.

## Setup Data

You first need to gain access to MIMIC-CXR (https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
and MS-CXR (https://physionet.org/content/ms-cxr/0.1/).

Then under ./configs/config_mscxr.yml you have to specify the relevant paths as described in the comments.
Note that for fine-tuning, you first need to generate .csv files containing the relevant paths and impressions based on the 
csv files provided in the MIMIC-CXR-JPG.
An example structure for the training csv can be found [here](https://github.com/MischaD/chest-distillation/blob/master/experiments/mimic_metadata_preprocessed.csv).
The evaluation csv can be the same as in the MS-CXR repository.

## Setup Checkpoints

Our CXR-BERT checkpoint is already located at ./finetune/normal/biovil/checkpoint-30000.
For other LLMs, the checkpoints first need to be added.
## Reproduce Main Results

Before running the evaluation scripts, first set the CXR_PR environment variable to the path to your repository and
activate the correct conda environment.

### Reproduce Evaluation of Main Results

To reproduce the main results without Bimodal Bias Merging (BBM), run `./evaluation/evaluation_launch_scripts/normal_biovil` or
`./evaluation/evaluation_launch_scripts/normal_biovil_bbm` to use BBM.

You can also use disease filtering approach instead of our lexcical filtering approach by running
`./evaluation/evaluation_launch_scripts/normal_biovil_no_lexical`.

### Reproduce Training of Main Results

To reproduce training, simply run `./training/launch_scripts/launch_training_biovil`.



# Additional Overview

For more specialized needs, you need to write your own scripts.
The following sections should help by giving an overview over the structure of the project.

### SLURM

Under cxr_phrase_grounding/slurm/... you can find slurm scripts to reproduce the used resource allocation in a HPC environment when invoking the fine-tuning scripts.
Use slurm_train_biovil to reproduce the fine-tuning used for the main results.

### Checkpoints

The checkpoints should be saved in the subdirectories cxr_phrase_grounding/finetune/...:
- ./control/clip/checkpoint-30000 contains the ControlNet checkpoint
- ./lora/clip/<rank>/checkpoint-30000 contains the LoRA results for ranks 4,8,16,...,256
- ./normal/<text_encoder>/checkpoint-30000 contains the default fine-tuning results for the specified text encoder (NOTE: The main results are located in *biovil*.)

## Evaluation

The evaluation script is cxr_phrase_grounding/evaluation/compute_bbox.py.
An exhaustive list of arguments for compute_bbox.py and their description can be found in cxr_phrase_grounding/evaluation/utils_evaluation.py.

## Datasets

All relevant dataset code can be found in cxr_phrase_grounding/xray_datasets/...
Specifically, mimic.py contains code for handling MIMIC-CXR and MS-CXR, while xray14.py handles ChestXRay14 data.
The corresponding data is expected to be in data/mimic and data/xray14 respectively.
However, this can be configured via the config files.

## Config

The config files are located in cxr_phrase_grounding/configs...
For some paths, you should set the corresponding environment variables or change them to your needs.




