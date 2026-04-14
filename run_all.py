import subprocess
import os

from common.utils import DATASETS_FOLDER_PATH

srbh_experiments_path = "experiments/srbh2020/"
SRBH_DATASET_PATH = os.path.join(DATASETS_FOLDER_PATH, "srbh")

PREPROCESS = False
SPLIT = False
TRAIN = False
TEST = False
VAL = True
PLOT = False


def build_params():
    builded = []
    builded.append("--do-preprocess" if PREPROCESS else "--no-preprocess")
    builded.append("--do-split" if SPLIT else "--no-split")
    builded.append("--do-train" if TRAIN else "--no-train")
    builded.append("--do-test" if TEST else "--no-test")
    builded.append("--do-val" if VAL else "--no-val")
    builded.append("--do-plot" if PLOT else "--no-plot")

    return builded


# SRBH
subprocess.run(["python", "srbh_modern_bert.py"] + build_params() + [SRBH_DATASET_PATH], cwd=srbh_experiments_path)
#subprocess.run(["python", "srbh_smol.py"] + build_params() + [SRBH_DATASET_PATH], cwd=srbh_experiments_path)

# SRBH-fix
#subprocess.run(["python", "srbh_fix_modernbert_s.py"] + build_params() + [SRBH_DATASET_PATH], cwd=srbh_experiments_path)
#subprocess.run(["python", "srbh_fix_smol_s.py"] + build_params() + [SRBH_DATASET_PATH], cwd=srbh_experiments_path)
