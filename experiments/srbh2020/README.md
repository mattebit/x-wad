# SRBH2020 experiments

This folder contains the experiments done on the SRBH2020 dataset. Each experiment is composed by training and evaluation.

Each model has been evaluated on the original dataset in the experiment [srbh_smol.py](srbh_smol.py) with SmolLM2-360M and [srbh_modernbert.py](srbh_modernbert.py) with ModernBERT.

Other experiments are available for the fixed version of SRBH2020, which first identifies the mislabeled samples and then exports a fixed version of the dataset training the models on it. This is done with SmolLM2-360M in [srbh_fix_smol_s.py](srbh_fix_smol_s.py) and with ModernBERT in [srbh_fix_modernbert_s.py](srbh_fix_modernbert_s.py).

The indexes of the fixed samples (changed from normal to anomalous) can be found in the csv [mislabeled.csv](mislabeled.csv)