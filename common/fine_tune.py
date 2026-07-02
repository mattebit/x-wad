from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers import EarlyStoppingCallback

from common.utils import (
    get_checkpoint,
    get_cpu_count,
    get_max_context_size,
    prepare_dataset,
    trainer_base_args,
    mlm_default_collator_config_train,
)


def finetune(
    train_dataset_path: str,
    source_model_path: str,
    output_model_path: str,
    finetune_training_args: dict,
    model_is_mlm: bool,
    eval_dataset_path: str = None,
    collator_settings_mlm=mlm_default_collator_config_train,
    auto_resume_from_checkpoint=True,
    model_load_settings=None,
    override_trainer=None,
):
    # Check if early stop is asked in the training arguments
    if "early_stop" in finetune_training_args.keys():
        early_stop = finetune_training_args.pop("early_stop")
    else:
        early_stop = None

    # Load original tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(source_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        source_model_path, **model_load_settings
    )
    model.resize_token_embeddings(
        len(tokenizer)
    )  # shouldn't be necessary if no tokens are added

    # Load train and validation datasets
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

    eval_dataset["evaluation"] = eval_dataset["evaluation"].remove_columns(
        columns_to_remove
    )
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

    # Differentiate between CLM and MLM
    if model_is_mlm:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            **collator_settings_mlm,
        )
        finetune_training_args.pop("mlm_probability")

    else:
        data_collator = DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer)

    callbacks = []
    if early_stop is not None and early_stop == True:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.005,
        )
        callbacks.append(early_stopping_callback)

    training_arguments = TrainingArguments(
        **trainer_base_args, output_dir=output_model_path, **finetune_training_args
    )

    trainer_args = {
        "model": model,
        "args": training_arguments,
        "data_collator": data_collator,
        "train_dataset": tokenized_dataset_train["train"],
        "eval_dataset": tokenized_dataset_eval["evaluation"]
        if eval_dataset_path is not None
        else None,
        "callbacks": callbacks,
    }

    if override_trainer is not None:
        trainer = override_trainer(**trainer_args)
    else:
        trainer = trainer(**trainer_args)

    # Check if there are any existing checkpoints in the output directory
    if auto_resume_from_checkpoint:
        checkpoint = get_checkpoint(output_model_path)
    else:
        checkpoint = None

    trainer.train(resume_from_checkpoint=checkpoint)
    print("Training complete. Saving the final model.")
    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)
