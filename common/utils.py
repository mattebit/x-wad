import argparse
import os
from pathlib import Path

import pandas
import torch
from pandas import DataFrame

from dotenv import load_dotenv

load_dotenv()

# Load config arguments
DATASETS_FOLDER_PATH = os.environ.get("DATASETS_FOLDER_PATH", "./datasets")
MODELS_FOLDER_PATH = os.environ.get("MODELS_FOLDER_PATH", "./models")
ALLOW_OPTIMIZATIONS = os.environ.get("ALLOW_OPTIMIZATIONS", "False") == "true"
IMAGES_FOLDER_PATH = os.environ.get("IMAGES_FOLDER_PATH", "./images")

# Automatically create all folder paths if they don't exist
for folder_path in [DATASETS_FOLDER_PATH, MODELS_FOLDER_PATH, IMAGES_FOLDER_PATH]:
    os.makedirs(folder_path, exist_ok=True)

try:
    MULTIPROCESSING_OVERRIDE_NUM_CORES = int(
        os.environ.get("MULTIPROCESSING_OVERRIDE_NUM_CORES", "0")
    )
except Exception:
    MULTIPROCESSING_OVERRIDE_NUM_CORES = 0

trainer_base_args = {
    "save_strategy": "steps",  # Save checkpoint every save_steps
    "save_steps": 200,
    "save_total_limit": 2,
    "bf16": True,  # Improves performance, better than fp16, works with RTX3090
    "logging_steps": 1,
    "dataloader_num_workers": os.cpu_count(),  # Use all available CPU cores to
    "gradient_checkpointing": True,  # to save VRAM
}

tokenizer_base_args = {
    "return_tensors": "pt",
    "max_length": 512,  # If left unset or set to None, this will use the predefined model maximum length
    "truncation": True,  # Must be True to avoid overflow
    "padding": "max_length",  # Add padding if sample is smaller than available context length
    "return_overflowing_tokens": False,  # Output format may differ in standard tokenization. Truncated tokens are lost
}

tokenizer_base_args_predict = tokenizer_base_args
tokenizer_base_args_predict.pop("return_overflowing_tokens")

mlm_default_collator_config_train = {
    "mlm_probability": 0.30,
    "mask_replace_prob": 1.0,
    "random_replace_prob": 0.0,
}

mlm_default_collator_config_predict = {
    "mlm_probability": 0.40,
    "mask_replace_prob": 1.0,
    "random_replace_prob": 0.0,
}

default_split_perc = {
    "train_frac_normals": 0.7,
    "train_frac_anomalous": 0,
    "eval_frac_normals": 0.1,
    "eval_frac_anomalous": 0,
    "test_frac_normals": 0.2,
    "test_frac_anomalous": 1,
}

model_load_settings_normal = {
    # "local_files_only": True,
    "dtype": torch.float16,
    "device_map": "cuda",
    # "use_cache":False
}

model_load_settings_optimized = {
    # "local_files_only": True,
    "attn_implementation": "flash_attention_2",  # Enable flash attention
    "dtype": torch.bfloat16,
    "device_map": "cuda",
    # "use_cache":False
}


def get_cpu_count():
    if MULTIPROCESSING_OVERRIDE_NUM_CORES != 0:
        return MULTIPROCESSING_OVERRIDE_NUM_CORES
    else:
        return os.cpu_count() - 1 if os.cpu_count() > 1 else os.cpu_count()


def get_dataset_split_paths_names(TRUTH_DATSET_PATH: str, dataset_prefix: str):
    THIS_DATASET_PATH, TRUTH_DATASET_NAME = os.path.split(TRUTH_DATSET_PATH)
    TRAIN_DATASET_PATH = os.path.join(
        THIS_DATASET_PATH, f"{dataset_prefix}-train-{TRUTH_DATASET_NAME}"
    )
    TRAIN_DATASET_NAME = os.path.splitext(os.path.basename(TRAIN_DATASET_PATH))[0]
    VALIDATION_DATASET_PATH = os.path.join(
        THIS_DATASET_PATH, f"{dataset_prefix}-eval-{TRUTH_DATASET_NAME}"
    )
    VALIDATION_DATASET_NAME = os.path.splitext(
        os.path.basename(VALIDATION_DATASET_PATH)
    )[0]
    TEST_DATASET_PATH = os.path.join(
        THIS_DATASET_PATH, f"{dataset_prefix}-test-{TRUTH_DATASET_NAME}"
    )
    TEST_DATASET_NAME = os.path.splitext(os.path.basename(TEST_DATASET_PATH))[0]
    return (
        TRAIN_DATASET_PATH,
        TRAIN_DATASET_NAME,
        VALIDATION_DATASET_PATH,
        VALIDATION_DATASET_NAME,
        TEST_DATASET_PATH,
        TEST_DATASET_NAME,
    )


def get_model_related_paths(model_folder_path, model_name, test_dataset_name):
    OUTPUT_MODEL_PATH = os.path.join(model_folder_path, model_name)
    EVALUATIONS_FOLDER_PATH = os.path.join(OUTPUT_MODEL_PATH, "evaluations/")
    PREDICTIONS_FOLDER_PATH = os.path.join(OUTPUT_MODEL_PATH, "predictions/")
    PREDICT_FILE_PATH = os.path.join(
        PREDICTIONS_FOLDER_PATH, f"pred_{test_dataset_name}.csv"
    )
    EVAL_FILE_PATH = os.path.join(
        EVALUATIONS_FOLDER_PATH, f"eval_{test_dataset_name}.csv"
    )

    return (
        EVALUATIONS_FOLDER_PATH,
        PREDICTIONS_FOLDER_PATH,
        PREDICT_FILE_PATH,
        EVAL_FILE_PATH,
    )


def get_argument_parser_experiments():
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.RawTextHelpFormatter
    )

    # required Argument
    parser.add_argument(
        "THIS_DATASET_PATH",
        type=str,
        help="The file path to the initial raw dataset (e.g., 'data/raw/dataset.csv').",
    )

    parser.add_argument(
        "--models-folder-path",
        type=str,
        default="models",
        help="Directory where trained models will be saved.\n(Default: models)",
    )
    # store_true means if the flag is present, the value is True.

    parser.add_argument(
        "--do-preprocess",
        action="store_true",
        default=False,
        help="Set this flag to enable the data preprocessing stage.\n(Default: False)",
    )
    parser.add_argument(
        "--no-preprocess",
        dest="do_preprocess",
        action="store_false",
        help="Explicitly disable the data preprocessing stage.",
    )

    parser.add_argument(
        "--do-split",
        action="store_true",
        default=False,
        help="Set this flag to enable the data splitting stage.\n(Default: False)",
    )
    parser.add_argument(
        "--no-split",
        dest="do_split",
        action="store_false",
        help="Explicitly disable the data splitting stage.",
    )

    parser.add_argument(
        "--do-train",
        action="store_true",
        default=True,
        help="Set this flag to enable the model training stage (Default behavior).\n(Default: True)",
    )
    # Using --no-train to explicitly override the default=True behavior
    parser.add_argument(
        "--no-train",
        dest="do_train",
        action="store_false",
        help="Explicitly disable the model training stage.",
    )

    parser.add_argument(
        "--do-test",
        action="store_true",
        default=True,
        help="Set this flag to enable the model test stage (Default behavior).\n(Default: True)",
    )
    parser.add_argument(
        "--no-test",
        dest="do_test",
        action="store_false",
        help="Explicitly disable the model test stage.",
    )

    parser.add_argument(
        "--do-val",
        action="store_true",
        default=True,
        help="Set this flag to enable the model validation stage (Default behavior).\n(Default: True)",
    )
    parser.add_argument(
        "--no-val",
        dest="do_val",
        action="store_false",
        help="Explicitly disable the model validation stage.",
    )

    parser.add_argument(
        "--do-plot",
        action="store_true",
        default=False,
        help="Set this flag to enable the performance plotting stage.\n(Default: False)",
    )
    parser.add_argument(
        "--no-plot",
        dest="do_plot",
        action="store_false",
        help="Explicitly disable the performance plotting stage.",
    )
    return parser


def check_and_create_file(file_path: str, create_if_missing: bool = False) -> bool:
    """
    Checks if a file exists at the given path. If it doesn't exist, can be created, including any necessary parent directories.

    :param file_path:  The full path to the file to be checked.
    :param create_if_missing: If set to True, the function will create the
                              file and its directories if they do not already
                              exist. Defaults to False.

    :return: True if the file exists at the end of the operation (either
          because it was already there or because it was created).
          False if the file does not exist and was not created, or if an
          error occurred.
    """
    if os.path.exists(file_path):
        return True

    # If the file does not exist, check the 'create_if_missing' flag
    if create_if_missing:
        try:
            # Get the directory part of the file path
            directory = os.path.dirname(file_path)

            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(file_path, "w") as f:
                pass  # The file is created but remains empty.

            return True
        except OSError as e:
            return False
    return False


def split_dataset(
    dataset_path: str,
    train_frac_normals: float,
    train_frac_anomalous: float,
    eval_frac_normals: float,
    eval_frac_anomalous: float,
    test_frac_normals: float,
    test_frac_anomalous: float,
    dataset_folder_path=None,
    prefix_str="",
    dry_run=False,
):
    """
    Specify how to split datasets
    :param dataset_path: relative to dataset_folder_path if dataset_Folder_path is not NOne
    :param train_frac_normals:
    :param train_frac_anomalous:
    :param eval_frac_normals:
    :param eval_frac_anomalous:
    :param test_frac_normals:
    :param test_frac_anomalous:
    :param dataset_folder_path:
    :return:
    """

    if dataset_folder_path is not None:
        dataset_file_path = os.path.join(dataset_folder_path, dataset_path)
    else:
        dataset_file_path = dataset_path

    df = pandas.read_csv(dataset_file_path)

    full_path = Path(dataset_file_path)
    directory_path = full_path.parent
    file_name = full_path.name

    def get_norm_anom(df_in: DataFrame):
        normal: DataFrame = df_in.loc[df_in["anomalous"] == 0]
        anomalous: DataFrame = df_in.loc[df_in["anomalous"] == 1]
        return normal, anomalous

    def print_stats(name, normal, anomalous, total):
        len_normal = len(normal)
        len_anomalous = len(anomalous)
        len_total = len(total)

        print(name)
        try:
            perc_over_total = len_total / total_count
            print(f"\tsplit perc: {perc_over_total:.2%}")
            print(f"\tnormal perc: {len_normal / len_total:.2%}")
            print(f"\tanomalous perc: {len_anomalous / len_total:.2%}")
        except:
            pass

    total_count = len(df)

    normal, anomalous = get_norm_anom(df)
    normal_count = len(normal)
    anomalous_count = len(anomalous)

    train_count_normals = int(
        train_frac_normals * normal_count
    )  # int() truncates, so should be fine
    train_count_anomalous = int(train_frac_anomalous * anomalous_count)
    eval_count_normals = int(eval_frac_normals * normal_count)
    eval_count_anomalous = int(eval_frac_anomalous * anomalous_count)
    test_count_normals = int(test_frac_normals * normal_count)
    test_count_anomalous = int(test_frac_anomalous * anomalous_count)

    if train_frac_normals + eval_frac_normals + test_frac_normals > 1:
        print("Error, normals fractions sum > 1")

    if train_frac_anomalous + eval_frac_anomalous + test_frac_anomalous > 1:
        print("Error, anomalous fractions sum > 1")

    def split_set(df_in: DataFrame, count_normals, count_anomalous, name=""):
        df_normal, df_anomalous = get_norm_anom(df_in)

        df_out_normals = df_normal.sample(n=count_normals)
        df_out_anomalous = df_anomalous.sample(n=count_anomalous)

        df_out = pandas.concat([df_out_normals, df_out_anomalous], axis=0)
        df_out = df_out.sample(frac=1).reset_index(drop=True)

        print_stats(name, df_out_normals, df_out_anomalous, df_out)

        return df_out

    # Train set
    df_train = split_set(df, train_count_normals, train_count_anomalous, name="train")
    df = df[
        ~df["original_index"].isin(df_train["original_index"])
    ]  # remove used samples

    # Eval set
    df_eval = split_set(df, eval_count_normals, eval_count_anomalous, name="eval")
    df = df[
        ~df["original_index"].isin(df_train["original_index"])
    ]  # remove used samples

    # Test set
    df_test = split_set(df, test_count_normals, test_count_anomalous, name="test")
    df = df[
        ~df["original_index"].isin(df_train["original_index"])
    ]  # remove used samples

    # Export
    train_path_name = os.path.join(directory_path, f"{prefix_str}-train-{file_name}")
    eval_path_name = os.path.join(directory_path, f"{prefix_str}-eval-{file_name}")
    test_path_name = os.path.join(directory_path, f"{prefix_str}-test-{file_name}")

    if not dry_run:
        df_train.to_csv(train_path_name)
        df_eval.to_csv(eval_path_name)
        df_test.to_csv(test_path_name)
    return 0
