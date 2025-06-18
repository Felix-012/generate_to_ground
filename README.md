# Reproducing Paper Results

Follow the steps below to reproduce the results presented in our paper.
Weights can be downloaded from our [Hugging Face Repository](https://huggingface.co/FelixNuetzel/cxr_bert_ldm/).

---

## 1. Setup Conda Environment

To create the required Conda environment, run the following command in the main directory:

```bash
conda env create --name cxr_phrase_grounding --file environment.yml
```

---

## 2. Setup Data

### Required Datasets

You need access to the following datasets:

- **MIMIC-CXR:** [PhysioNet MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- **MS-CXR:** [PhysioNet MS-CXR v0.1](https://physionet.org/content/ms-cxr/0.1/)

### Configuration

- Update the paths in `./configs/config_mscxr.yml` according to the comments provided.
- For fine-tuning, generate `.csv` files containing the relevant paths and impressions using the MIMIC-CXR-JPG data.
- Example training CSV structure: [mimic_metadata_preprocessed.csv](https://github.com/MischaD/chest-distillation/blob/master/experiments/mimic_metadata_preprocessed.csv)
- For evaluation, the CSV file provided in the MS-CXR repository can be used.

---

## 3. Setup Checkpoints

The fine-tuned LDM checkpoint with CXR-BERT conditioning is available at:

[Hugging Face Repository](https://huggingface.co/FelixNuetzel/cxr_bert_ldm/)

The checkpoint will be automatically loaded when running the standard evaluation script. Alternatively, download it manually and specify the checkpoint path in the script arguments.

---

## 4. Reproducing Main Results

### Environment Setup

Before running evaluation scripts:

1. Set the `CXR_PG` environment variable to the path of your repository.
2. Set the `WORK` environment variable to the path of your parent repository or change the path in config.
3. Activate the Conda environment:

```bash
conda activate cxr_phrase_grounding
```

### Evaluation

To reproduce evaluation results:

- Without Bimodal Bias Merging (BBM):
  ```bash
  ./evaluation/evaluation_launch_scripts/normal_biovil
  ```
- With BBM:
  ```bash
  ./evaluation/evaluation_launch_scripts/normal_biovil_bbm
  ```
- Using disease filtering instead of lexical filtering:
  ```bash
  ./evaluation/evaluation_launch_scripts/normal_biovil_no_lexical
  ```

### Training

---

You will probably first need to review the accelerate configuration in 
`./configs/accelerate_config.yaml`.

Then, to reproduce the training process, run:

```bash
./training/launch_scripts/launch_training_biovil
```

The checkpoints for training are saved and loaded in `./finetune/normal/biovil` by default.


## 5. Additional Overview

### SLURM

SLURM scripts for resource allocation in an HPC environment are located in:

```
cxr_phrase_grounding/slurm/
```

Use `slurm_train_biovil` to reproduce the fine-tuning used for the main results.

### Checkpoints Directory Structure

Checkpoints are expected to be stored in this way:

```
cxr_phrase_grounding/finetune/
```

- **ControlNet Checkpoint:** `./control/clip/checkpoint-30000`
- **LoRA Results (ranks 4, 8, 16, ..., 256):** `./lora/clip/<rank>/checkpoint-30000`
- **Default Fine-tuning Results:** `./normal/<text_encoder>/checkpoint-30000`
  - *Main results are located in the `biovil` directory.*

---

### Evaluation

The evaluation script is located at:

```
cxr_phrase_grounding/evaluation/compute_bbox.py
```

A comprehensive list of arguments and descriptions can be found in:

```
cxr_phrase_grounding/evaluation/utils_evaluation.py
```

---

### Datasets

All dataset-related code is available in:

```
cxr_phrase_grounding/xray_datasets/
```

- `mimic.py`: Handles MIMIC-CXR and MS-CXR datasets.
- `xray14.py`: Handles ChestXRay14 dataset.

Default dataset locations:

```
data/mimic
```
```
data/xray14
```

Paths can be adjusted in the config files.

---

### Configuration

Configuration files are located in:

```
cxr_phrase_grounding/configs/
```

Ensure environment variables are set or adjust file paths as needed.

---

By following the steps above, you can successfully reproduce the results presented in the paper.


# Acknowledgement

(Some) HPC resources were provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) under the NHR projects b143dc and b180dc. NHR funding is provided by federal and Bavarian state authorities. NHR@FAU hardware is partially funded by the German Research Foundation (DFG) – 440719683.

