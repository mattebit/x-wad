import os

import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments, AutoTokenizer, EarlyStoppingCallback, AutoConfig, AutoModelForMaskedLM,
    AutoModelForCausalLM
)
from transformers.trainer_utils import get_last_checkpoint

from common.utils import trainer_base_args, tokenizer_base_args, get_cpu_count, model_load_settings_normal


def train_model(
        train_dataset_path: str,
        model_output_dir: str,
        training_args: dict,
        model_config_override_args: dict = {},
        tokenizer_override_config_args: dict = {},
        tokenizer_args: dict = tokenizer_base_args,
        auto_resume_from_checkpoint=True,
        val_dataset_path=None,
        custom_fixed_tokenizer_tokens=[],
        model_name="ModernBERT-large",
        model_is_mlm=True,
        model_load_settings=model_load_settings_normal
):
    """
    Trains a model from scratch starting from its initial configuration


    :param train_dataset_path: The path of the dataset to use in the custom format
    :param model_output_dir: the path of a directory where to save the model
    :param training_args: the arguments passed to the trainer of the model
    :param model_config_override_args: If needed, it is possible to override the model config
    :param tokenizer_override_config_args: The configuration to be used when loading the tokenizer
    :param tokenizer_args: The arguments used by the tokenizer to tokenize the dataset
    :param auto_resume_from_checkpoint: Whether to search and load a previous checkpoint automatically before start
    training
    :param val_dataset_path: Optional validation dataset to be used during training
    :param custom_fixed_tokenizer_tokens: Optional custom tokens to be used in the tokenizer
    :param model_name: The name of the model to be used
    :param model_is_mlm: True if the model uses MLM, false for CLM
    :param model_load_settings: The settings used to load the model
    """

    if not os.path.exists(model_output_dir):
        # Load model config from pretrained model. There is the possibility of overriding some args
        config = AutoConfig.from_pretrained(model_name, **model_config_override_args)
        print("Created model config")

        # Load original tokenizer, but retrain it
        original_tokenizer = AutoTokenizer.from_pretrained(model_name)

        dataset = load_dataset("csv", data_files=train_dataset_path, split="train")

        def get_training_corpus(dataset, batch_size=1000):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i: i + batch_size]["request"]

        if "vocab_size" not in tokenizer_override_config_args:
            tokenizer_override_config_args["vocab_size"] = config.vocab_size

        tokenizer = original_tokenizer.train_new_from_iterator(
            get_training_corpus(dataset),
            **tokenizer_override_config_args
        )

        tokenizer.add_tokens(custom_fixed_tokenizer_tokens)

        if not model_is_mlm:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.save_pretrained(model_output_dir)

        # Update model vocab_size (if needed)
        config.vocab_size = len(tokenizer)

        config.bos_token_id = tokenizer.bos_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.eos_token_id

        # Save new model
        if model_is_mlm:
            model = AutoModelForMaskedLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_config(config)
        model.save_pretrained(model_output_dir)
    else:
        print("Tokenizer already exists. Skipping training.")
        tokenizer = AutoTokenizer.from_pretrained(model_output_dir)

    if model_is_mlm:
        model = AutoModelForMaskedLM.from_pretrained(model_output_dir, **model_load_settings)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_output_dir, **model_load_settings)

    print(f"Tokenizer size {len(tokenizer)}")
    print(f"Model tokenizer size {model.config.vocab_size}")

    if "early_stop" in training_args.keys():
        early_stop = training_args.pop("early_stop")
    else:
        early_stop = None

    if model_is_mlm:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=training_args["mlm_probability"]
        )
        training_args.pop("mlm_probability")

    else:
        data_collator = DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer)

    data_files = {"train": train_dataset_path}

    if val_dataset_path is not None:
        data_files["evaluation"] = val_dataset_path

    full_dataset = load_dataset(
        'csv',
        data_files=data_files
    )
    dataset = full_dataset.select_columns("request")

    def tokenize_function(examples):
        return tokenizer(
            examples["request"],
            **tokenizer_args
        )

    # Check that the dataset contains all valid strings (str type and not empty)
    def is_valid_string(example):
        return isinstance(example['request'], str) and example['request'].strip() != ""

    # Apply the filter to remove invalid data
    dataset = dataset.filter(is_valid_string)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=get_cpu_count(),
        remove_columns=["request"],
    )

    # Check if there are any existing checkpoints in the output directory
    checkpoint = None
    if os.path.isdir(model_output_dir) and auto_resume_from_checkpoint:
        last_checkpoint = get_last_checkpoint(model_output_dir)
        if last_checkpoint is not None:
            print(f"Checkpoint found. Resuming training from: {last_checkpoint}")
            print(f"Warning, provided model config will be ignored")
            checkpoint = last_checkpoint

    if "torch_compile" in trainer_base_args.keys():
        torch._dynamo.config.recompile_limit = 1e4  # Remove recompilation limits for multiple consecutive runs

    args = trainer_base_args | training_args

    training_args_ = TrainingArguments(
        output_dir=model_output_dir,
        **args,
    )

    callbacks = []

    if early_stop is not None and early_stop == True:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,
            # early_stopping_threshold=0.025,
        )
        callbacks.append(early_stopping_callback)

    trainer = Trainer(
        model=model,
        args=training_args_,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["evaluation"],
        callbacks=callbacks,
    )

    print("Starting training...")

    # If `checkpoint` is None, it will start training from scratch.
    trainer.train(resume_from_checkpoint=checkpoint)

    print("Training complete. Saving the final model.")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    return 0
