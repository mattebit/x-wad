# X-WAD (eXplainable Web Anomaly Detection)

X-WAD is a tool for explainable web anomaly detection using Transformer-based Language Models (TLMs) to identify and explain malicious network traffic patterns.

This work was initially developed during my master thesis "Explainable Anomaly Detection in Web Data using Transformer-based Language Models".

<video src="https://github.com/user-attachments/assets/02ddc8e2-091a-4ea2-a8a5-f97b39cf8fbb"></video>

## Features

- Supports both CLM and MLM paradigms for anomaly detection and explainability
- Natively supported models: ModernBERT (MLM) and SmolLM2-360M (CLM)
- Evaluation on SRBH2020 (CAPEC) dataset

## Installation

Install the required dependencies:

```bash
# Optionally create venv
python3 -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Note on optimization dependencies
This tool supports specific optimizations for NVIDIA GPUs based on Ampere archietecture or newer. The package `flash-attn` is required for such optimization to be used, however it is not directly included in the requirements file for compatibility purposes. If you want to use the available optimization install the package manually.

### Model download
- [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-large)

To automatically download the models execute the `setup.sh` script.

### Evaluation dataset download

- [SR-BH2020 Dataset](https://doi.org/10.7910/DVN/OGOIXX)

To download the SRBH2020 dataset needed for evaluation, download it from the above link and
extract the file `data_capec_multilabel.csv` into this directory [datasets/srbh/original/](datasets/srbh/original/). It is possible to change the location of the dataset directory through the `.env` file.

## Usage

The tool is based on three different phases:

1) The **model** need to be **finetuned** or **trained** from scratch on the specific dataset through the scripts in the experiment folder
2) (optional) the **model** can be **evaluated** on the **dataset**, through the same scripts in the experiment folder
3) (optional) The **samples** can be **explained** by the trained model with the `explain.py` script

### Running Individual Experiments

You can run any experiment script directly from the [experiments/srbh2020/](experiments/srbh2020/) directory, for example:

```bash
cd experiments/srbh2020/
python srbh_modernbert.py --do-preprocess --do-split --do-train --do-test --do-val --do-plot
```

Each experiment supports the following optional flags:

- `--do-preprocess`: Run data preprocessing
- `--do-split`: Split dataset into train/val/test sets
- `--do-train`: Train the model
- `--do-val`: Run evaluation on test set
- `--do-test`: Generate predictions on test set
- `--do-plot`: Generate visualization plots
- `--no-preprocess`: Skip preprocessing (default)
- `--no-split`: Skip dataset splitting (default)
- `--no-train`: Skip training (default)
- `--no-test`: Skip testing (default)
- `--no-val`: Skip validation (default)
- `--no-plot`: Skip plotting (default)

**NOTE**: After the dataset is preprocessed and splitted the first time, it is not necessary to do it again. Just remove `--do-preprocess` `--do-split`, otherwise the dataset splits will change.

### Running All Experiments

To run all experiments with a single command, use the [run_all.py](run_all.py) script:

```bash
python run_all.py
```

**Note:** By default the script assumes that the dataset is in the [datasets/](datasets/) folder.

### Explain the samples

This project is still a prototype and still misses a dedicated UI. In order to visualize the explainability it is possible to use the [explain.py](explain.py) editing it to specify the model to use (that should have previously been trained) and the sample to be explained. The file is configured by default to explain a sample from the srbh fix dataset using SmolLM2-360M.

```bash
python explain.py
```

## Project Structure

```
x-wad/
├── experiments/
│   └── srbh2020/
│       ├── srbh_modernbert.py      # ModernBERT original
│       ├── srbh_smol.py            # SmolLM2-360M original
│       ├── srbh_fix_modernbert_s.py # Fixed ModernBERT variant
│       ├── srbh_fix_smol_s.py      # Fixed SmolLM2-360M
│       └── tools.py                # Common utilities for srbh
├── run_all.py                      # Script to run all experiments
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## License

```
   Copyright 2026 Matteo Bitussi

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
