import argparse
import os
from pathlib import Path

import pandas
import torch
from pandas import DataFrame

from dotenv import load_dotenv
from transformers.trainer_utils import get_last_checkpoint

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
    "save_steps": 50,
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
    "dtype": torch.bfloat16,
    "attn_implementation": "eager",  # Setting it to "eager" forces the model to use standard PyTorch operations. Crucially, PyTorch's eager/math attention backend automatically detects BF16 inputs and upcasts the intermediate attention and softmax matrix to FP32 for stability.
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
        except OSError:
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
        df_train.to_csv(train_path_name, index=False)
        df_eval.to_csv(eval_path_name, index=False)
        df_test.to_csv(test_path_name, index=False)
    return 0


def get_max_context_size(model) -> int:
    """Get a model maximum context size

    Args:
        model (_type_): The model object

    Returns:
        int: The maximum context size
    """
    return (
        getattr(
            model.config, "max_position_embeddings", None
        )  # LLaMA, Mistral, Opt, etc.
        or getattr(model.config, "n_positions", None)  # GPT-2
        or getattr(model.config, "n_ctx", None)  # Older GPT variants
    )


def prepare_dataset(
    dataset,
    tokenizer,
    max_context_size,
    divide_samples=True,
    tokenizer_args={},
    column_name="text",
):
    """Function used to prepare a dataset for training or inference. Two methods are available:
    divide_samples = False -> all samples in dataset are concatenated without any special token in between
    divide_samples = True -> samples in dataset are concatenated with a special token in between to separate their context
    This function guarantees that no samples are splitted along blocks (apart from samples longer than block size)
    # TODO: add position IDS and attention mask fix when using divide_samples=True, otherwise this will not work correctly

    Args:
        dataset: The dataset object to be processed
        tokenizer: The tokenizer to be used for tokenization
        max_context_size: The model's maximum context size
        divide_samples (bool, optional): Which method to be used. Defaults to True.
        tokenizer_args (optional): Tokenizer arguments to pass to the tokenizer
        column_name: The column name of the content in the dataset
    """

    # Ensure the tokenizer has a pad token assigned
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Clean dataset by removeing non essential columns

    # Define the essential columns you want to keep
    essential_columns = {"input_ids", "attention_mask", "labels", column_name}

    # Identify the columns to remove by finding the difference
    columns_to_remove = [
        col for col in dataset.column_names if col not in essential_columns
    ]

    # Drop the non-essential columns
    cleaned_dataset = dataset.remove_columns(columns_to_remove)

    def pack_logs_to_max_context(examples, block_size=4096):
        result = {k: [] for k in examples.keys()}
        result["labels"] = []

        current_chunks = {k: [] for k in examples.keys()}
        current_length = 0

        num_samples = len(examples["input_ids"])

        if tokenizer.eos_token_id is None:
            # If tokenizer doesn't have EOS token, it is probably a MLM based model
            # It should use SEP as a separator betwen sentences
            separator_token = tokenizer.sep_token_id
        else:
            # If EOS is present use EOS, which means CLM-based model
            separator_token = tokenizer.eos_token_id

        for i in range(num_samples):
            # Prepare the current sample
            current_sample = {}
            for k in examples.keys():
                seq = examples[k][i]

                # If True, inject the EOS token to separate contexts
                if divide_samples:
                    if k == "input_ids":
                        seq = seq + [separator_token]
                    elif k == "attention_mask":
                        seq = seq + [1]  # Attention should be active on the EOS token
                    else:
                        continue  # Ignore other columns

                current_sample[k] = seq

            sample_len = len(current_sample["input_ids"])

            # Edge Case: A single sample is larger than the block_size
            if sample_len > block_size:
                if current_length > 0:
                    pad_len = block_size - current_length
                    for k in examples.keys():
                        pad_val = tokenizer.pad_token_id if k == "input_ids" else 0
                        result[k].append(current_chunks[k] + [pad_val] * pad_len)
                    result["labels"].append(
                        current_chunks["input_ids"] + [-100] * pad_len
                    )

                    current_chunks = {k: [] for k in examples.keys()}
                    current_length = 0

                # Truncate and flush oversized sample
                for k in examples.keys():
                    result[k].append(current_sample[k][:block_size])
                result["labels"].append(current_sample["input_ids"][:block_size])
                continue

            # If adding this sample exceeds block_size, pad and save the current chunk
            if current_length + sample_len > block_size:
                pad_len = block_size - current_length
                for k in examples.keys():
                    pad_val = tokenizer.pad_token_id if k == "input_ids" else 0
                    result[k].append(current_chunks[k] + [pad_val] * pad_len)

                result["labels"].append(current_chunks["input_ids"] + [-100] * pad_len)

                current_chunks = {k: [] for k in examples.keys()}
                current_length = 0

            # Accumulate the current sample into the current chunk
            for k in examples.keys():
                current_chunks[k].extend(current_sample[k])
            current_length += sample_len

        # Flush any remaining tokens in the final chunk
        if current_length > 0:
            pad_len = block_size - current_length
            for k in examples.keys():
                pad_val = tokenizer.pad_token_id if k == "input_ids" else 0
                result[k].append(current_chunks[k] + [pad_val] * pad_len)

            result["labels"].append(current_chunks["input_ids"] + [-100] * pad_len)

        return result

    def tokenize_function(examples):
        return tokenizer(examples[column_name], **tokenizer_args)

    tokenized_dataset = cleaned_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=get_cpu_count(),
        remove_columns=[column_name],
    )

    packed_dataset = tokenized_dataset.map(
        pack_logs_to_max_context,
        batched=True,
        fn_kwargs={"block_size": max_context_size},
        num_proc=get_cpu_count(),
    )

    return packed_dataset


def get_checkpoint(output_model_path):
    checkpoint = None
    if os.path.isdir(output_model_path):
        last_checkpoint = get_last_checkpoint(output_model_path)
        if last_checkpoint is not None:
            print(f"Checkpoint found. Resuming training from: {last_checkpoint}")
            print("Warning, provided model config will be ignored")
            checkpoint = last_checkpoint

    return checkpoint
