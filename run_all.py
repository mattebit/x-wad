import subprocess
import os

from common.utils import DATASETS_FOLDER_PATH

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
subprocess.run(
    ["python", "-m", "experiments.srbh2020.srbh_modernbert"]
    + build_params()
    + [SRBH_DATASET_PATH]
)
subprocess.run(
    ["python", "-m", "experiments.srbh2020.srbh_smol"]
    + build_params()
    + [SRBH_DATASET_PATH]
)

# SRBH-fix
subprocess.run(
    ["python", "-m", "experiments.srbh2020.srbh_fix_modernbert_s"]
    + build_params()
    + [SRBH_DATASET_PATH]
)
subprocess.run(
    ["python", "-m", "experiments.srbh2020.srbh_fix_smol_s"]
    + build_params()
    + [SRBH_DATASET_PATH]
)
