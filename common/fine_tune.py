import os

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling, AutoModelForMaskedLM, )
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

from common.TrainerCausal import TrainerCausal
from common.TrainerMasked import TrainerMasked
from common.utils import trainer_base_args, tokenizer_base_args, get_cpu_count, \
    mlm_default_collator_config_train


def get_checkpoint(output_model_path):
    checkpoint = None
    if os.path.isdir(output_model_path):
        last_checkpoint = get_last_checkpoint(output_model_path)
        if last_checkpoint is not None:
            print(f"Checkpoint found. Resuming training from: {last_checkpoint}")
            print(f"Warning, provided model config will be ignored")
            checkpoint = last_checkpoint

    return checkpoint

# TODO: unify the finetune functions in a single function parametrizing the type of task (MLM or CLM)

def finetune_masked(
        train_dataset_path: str,
        source_model_path: str,
        output_model_path: str,
        finetune_training_args: dict,
        eval_dataset_path: str = None,
        collator_settings=mlm_default_collator_config_train,
        auto_resume_from_checkpoint=True,
        model_load_settings=None
):
    """
    Function used to finetune an already existing model, starting from its pretrained weights

    :param train_dataset_path: The training set to use in training
    :param source_model_path: The original model to finetune
    :param output_model_path: The new generated model path
    :param finetune_training_args: The arguments used for training
    :param eval_dataset_path: The evaluation dataset path
    :param collator_settings: the settings for MLM data collator, default .9 mask prob, 0% random substitution
    :param auto_resume_from_checkpoint: Set to true to automatically resume the training from a checkpoint
    :param model_load_settings: The settings to be used when loading model
    """

    if "early_stop" in finetune_training_args.keys():
        early_stop = finetune_training_args.pop("early_stop")
    else:
        early_stop = None

    train_dataset = load_dataset('csv', data_files={"train": train_dataset_path})
    train_dataset.select_columns(["request"])

    if eval_dataset_path is not None:
        eval_dataset = load_dataset('csv', data_files={"evaluation": eval_dataset_path})
        # Remove useless columns
        columns_to_remove = []
        for c in eval_dataset["evaluation"].column_names:
            if c not in ["request", "anomalous", "evaluation"]:
                columns_to_remove.append(c)

        eval_dataset["evaluation"] = eval_dataset["evaluation"].remove_columns(columns_to_remove)
        eval_dataset.select_columns(["request", "anomalous"])

    model = AutoModelForMaskedLM.from_pretrained(
        source_model_path,
        **model_load_settings
    )
    tokenizer = AutoTokenizer.from_pretrained(source_model_path)

    def tokenize_function(examples):
        return tokenizer(
            examples["request"],
            **tokenizer_base_args
        )

    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=get_cpu_count(),
        remove_columns=["request"],
        batch_size=100  # to limit ram usage
    )

    if eval_dataset_path is not None:
        tokenized_eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=get_cpu_count(),
            remove_columns=["request"],
            batch_size=100  # to limit ram usage
        )

    training_args = TrainingArguments(
        **trainer_base_args,
        output_dir=output_model_path,
        **finetune_training_args
    )

    callbacks = []
    if early_stop is not None and early_stop == True:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,
            #early_stopping_threshold=0.005,
        )
        callbacks.append(early_stopping_callback)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        **collator_settings
    )

    trainer = TrainerMasked(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset["train"],
        eval_dataset=tokenized_eval_dataset["evaluation"] if eval_dataset_path is not None else None,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # Check if there are any existing checkpoints in the output directory
    if auto_resume_from_checkpoint:
        checkpoint = get_checkpoint(output_model_path)
    else:
        checkpoint = None

    trainer.train(resume_from_checkpoint=checkpoint)
    print("Training complete. Saving the final model.")
    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)


def finetune_causal(
        train_dataset_path: str,
        source_model_path: str,
        output_model_path: str,
        finetune_training_args: dict,
        eval_dataset_path: str = None,
        auto_resume_from_checkpoint=True,
        model_load_settings=None
):
    # Check if early stop is asked in the training arguments
    if "early_stop" in finetune_training_args.keys():
        early_stop = finetune_training_args.pop("early_stop")
    else:
        early_stop = None

    tokenizer = AutoTokenizer.from_pretrained(source_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(source_model_path, **model_load_settings)
    model.resize_token_embeddings(len(tokenizer))  # shouldn't be necessary if no tokens are added

    def tokenize_function(examples):
        return tokenizer(examples["request"])

    train_dataset = load_dataset("csv", data_files={"train": train_dataset_path})
    train_dataset.select_columns(["request"])
    eval_dataset = load_dataset("csv", data_files={"evaluation": eval_dataset_path})

    # Remove useless columns
    columns_to_remove = []
    for c in eval_dataset["evaluation"].column_names:
        if c not in ["request", "anomalous", "evaluation"]:
            columns_to_remove.append(c)

    eval_dataset["evaluation"] = eval_dataset["evaluation"].remove_columns(columns_to_remove)
    eval_dataset.select_columns(["request", "anomalous"])

    tokenized_dataset_train = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=get_cpu_count(),
        remove_columns=["request"],
    )

    tokenized_dataset_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=get_cpu_count(),
        remove_columns=["request"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    callbacks = []
    if early_stop is not None and early_stop == True:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=.005,
        )
        callbacks.append(early_stopping_callback)

    training_arguments = TrainingArguments(
        **trainer_base_args,
        output_dir=output_model_path,
        **finetune_training_args
    )

    trainer = TrainerCausal(
        model=model,
        args=training_arguments,
        data_collator=data_collator,
        train_dataset=tokenized_dataset_train["train"],
        eval_dataset=tokenized_dataset_eval["evaluation"] if eval_dataset_path is not None else None,
        callbacks=callbacks
    )

    # Check if there are any existing checkpoints in the output directory
    if auto_resume_from_checkpoint:
        checkpoint = get_checkpoint(output_model_path)
    else:
        checkpoint = None

    trainer.train(resume_from_checkpoint=checkpoint)
    print("Training complete. Saving the final model.")
    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)
