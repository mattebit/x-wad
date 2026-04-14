import os
from pathlib import Path

from common import utils
from common.eval import evaluate, get_AUC_ROC_from_predictions
from common.fine_tune import finetune_masked, finetune_causal
from common.predict import get_batch_predictions_masked_strided, get_batch_predictions_causal, \
    estimate_threshold_from_predictions
from common.train_from_scratch import train_model
from common.utils import check_and_create_file, MODELS_FOLDER_PATH, mlm_default_collator_config_train

# Force HF offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def execute(
        train_dataset_path: str,
        output_model_name: str,
        models_folder_path=MODELS_FOLDER_PATH,
        validate_dataset_path=None,
        do_train_from_scratch=False,
        train_from_scratch_args: dict = None,
        model_config_args: dict = None,
        do_train_finetuning=False,
        base_model_name: str = None,
        train_finetuning_args: dict = None,
        task="mlm",
        do_predict=False,
        predict_batch_size=50,
        predict_dataset_path=None,  # TODO: internally manage paths of such datasets?
        predict_name_suffix: str = "",
        do_evaluate=False,
        evaluate_params: list = None,
        eval_auto_threshold_est=True,
        auto_threshold_est_method="percentile",
        truth_dataset_path: str = None,
        eval_name_suffix: str = "",
        tokenizer_tokens=[],
        train_resume_from_checkpoint=True,
        model_load_settings=utils.model_load_settings_normal
):
    """
    This function executes training, validation, test and plot depending on how it is configured
    :param train_dataset_path: The train dataset path to be used for training
    :param output_model_name: The name of the model that is going to be created
    :param models_folder_path: The folder where the models are going to be saved and loaded from (default
    MODELS_FOLDER_PATH env variable)
    :param validate_dataset_path: If left to None train dataset is used for validation
    :param do_train_from_scratch: Enables the training from scratch. Uses the model "base_model_name" as architecture
    :param train_from_scratch_args: Training arguments used to train from scratch
    :param model_config_args: (optional) set or overwrite model configuration parameters
    :param do_train_finetuning: Enables training via finetuning starting from the model pretrained weights. Uses model
    "base_model_name".
    :param base_model_name: The name of the model to be used for training (should be present in models folder or path
    to huggingface repo)
    :param train_finetuning_args: The arguments used during finetuning training
    :param task: specify if the model uses mlm or causal
    :param do_predict: Enable the prediction step, where both the validation and test set get predicted and saved to a
    file
    :param predict_batch_size: The batch size to be used during prediction
    :param predict_dataset_path: The path of the dataset to be predicted
    :param predict_name_suffix: (optional) add a suffix to the output prediction file
    :param do_evaluate: Enables the evaluation step on the prediction file generated in the prediction step
    :param evaluate_params: (optional) Specify parameters that define how the evaluation is done
    :param eval_auto_threshold_est: Enable the automatic estimation of the threshold during evaluation
    :param auto_threshold_est_method: Specify the method for the auto threhsold estimation (gaussian or percentile)
    :param truth_dataset_path: The truth dataset path is the original dataset processed csv file containing the labels
    needed for prediction
    :param eval_name_suffix: (optional) add a suffix to the evaluation output file
    :param tokenizer_tokens: A list of fixed tokens to be used by the tokenizer during the training of the model
    :param train_resume_from_checkpoint: Enable the automatic resuming of training from a found checkpoint
    :param model_load_settings: Specifies the settings used to load the model
    :return:
    """
    # TODO: validate_dataset_path change to enable to skip validation from here?

    # Check the correctness of the parameters
    check_and_create_file(train_dataset_path)
    check_and_create_file(models_folder_path)

    if validate_dataset_path is None:
        print("Warning! evaluation dataset automatically set to train dataset")
        validate_dataset_path = train_dataset_path

    OUTPUT_MODEL_PATH = os.path.join(models_folder_path, output_model_name)

    # Logging settings
    mandatory_training_args = {
        "logging_dir": os.path.join(OUTPUT_MODEL_PATH, "logs"),
        "report_to": "tensorboard"
    }

    if do_train_from_scratch:
        if train_from_scratch_args is None:
            training_args_default = {
                "learning_rate": 0.1,
                "num_train_epochs": 2.5,
                "per_device_train_batch_size": 32,
                "mlm_probability": 0.15,
                "tokenizer_type": "wordpiece"
            }

            print("Warning! using default training args")
            train_from_scratch_args = training_args_default

        train_from_scratch_args_ok = train_from_scratch_args | mandatory_training_args

        if model_config_args is None:
            model_config_args = {}

        res = train_model(
            train_dataset_path,
            OUTPUT_MODEL_PATH,
            train_from_scratch_args_ok,
            model_config_args,
            val_dataset_path=validate_dataset_path,
            custom_fixed_tokenizer_tokens=tokenizer_tokens,
            auto_resume_from_checkpoint=train_resume_from_checkpoint,
            model_name=base_model_name,
            model_is_mlm=task == "mlm",
            model_load_settings=model_load_settings
        )
        if res == 1:
            # TODO manage errors
            return

    if do_train_finetuning:
        if base_model_name is None:
            # TODO manage errors
            print("finetune model name cannot be empty")
            return

        train_finetuning_args_ok = train_finetuning_args | mandatory_training_args

        FINETUNE_MODEL_PATH = os.path.join(models_folder_path, base_model_name)
        if task == "mlm":
            finetune_masked(
                train_dataset_path,
                FINETUNE_MODEL_PATH,
                OUTPUT_MODEL_PATH,
                train_finetuning_args_ok,
                eval_dataset_path=validate_dataset_path,
                collator_settings=mlm_default_collator_config_train,
                auto_resume_from_checkpoint=train_resume_from_checkpoint,
                model_load_settings=model_load_settings
            )
        else:
            finetune_causal(
                train_dataset_path,
                FINETUNE_MODEL_PATH,
                OUTPUT_MODEL_PATH,
                train_finetuning_args_ok,
                eval_dataset_path=validate_dataset_path,
                auto_resume_from_checkpoint=train_resume_from_checkpoint,
                model_load_settings=model_load_settings
            )

    prediction_dataset_path = train_dataset_path if predict_dataset_path is None else predict_dataset_path
    prediction_dataset_name = os.path.splitext(os.path.basename(prediction_dataset_path))[0]
    prediction_dir_path = os.path.join(OUTPUT_MODEL_PATH, "predictions")
    Path(prediction_dir_path).mkdir(parents=True, exist_ok=True)
    prediction_output_path = os.path.join(prediction_dir_path,
                                          f"pred_{prediction_dataset_name}{predict_name_suffix}.csv")

    # Need to get predictions to estimate threshold
    validate_dataset_name = os.path.splitext(os.path.basename(validate_dataset_path))[0]
    validate_prediction_output_path = os.path.join(prediction_dir_path,
                                                   f"pred_{validate_dataset_name}{predict_name_suffix}.csv")

    if do_predict:
        if task == "mlm":
            # Get predictions of test set
            get_batch_predictions_masked_strided(
                prediction_dataset_path,
                OUTPUT_MODEL_PATH,
                prediction_output_path,
                batch_size=predict_batch_size,
                model_load_settings=model_load_settings
            )
            # Get predictions of validation set
            get_batch_predictions_masked_strided(
                validate_dataset_path,
                OUTPUT_MODEL_PATH,
                validate_prediction_output_path,
                batch_size=predict_batch_size,
                model_load_settings=model_load_settings
            )
        else:
            # Get predictions of test set
            get_batch_predictions_causal(
                prediction_dataset_path,
                OUTPUT_MODEL_PATH,
                prediction_output_path,
                batch_size=predict_batch_size,
                model_load_settings=model_load_settings
            )
            # Get predictions of validation set
            get_batch_predictions_causal(
                validate_dataset_path,
                OUTPUT_MODEL_PATH,
                validate_prediction_output_path,
                batch_size=predict_batch_size,
                model_load_settings=model_load_settings
            )

    if do_evaluate:
        params = []
        if evaluate_params is None:
            # Use default evaluation parameters
            for i in range(1, 100, 1):
                params.append((3000, 0.01 * i, "loss", True, False, False, 0.0, False, True, True))
        else:
            params = evaluate_params

        if eval_auto_threshold_est:
            estimated_threshold = estimate_threshold_from_predictions(
                validate_prediction_output_path, method=auto_threshold_est_method, k=.5, plot=True)

            params.append((3000, estimated_threshold, "loss", True, False, False, 0.0, False, True, True, True))

        evaluation_dir_path = os.path.join(OUTPUT_MODEL_PATH, "evaluations")
        Path(evaluation_dir_path).mkdir(parents=True, exist_ok=True)
        eval_output_path = os.path.join(evaluation_dir_path, f"eval_{prediction_dataset_name}{eval_name_suffix}.csv")

        evaluate(prediction_output_path, truth_dataset_path, params, eval_output_path)

    try:
        auc_roc, auc_pr, auc_roc_ppl, auc_pr_ppl = get_AUC_ROC_from_predictions(prediction_output_path,
                                                                                truth_dataset_path)
        print(f"Calculated AUC score: loss:{auc_roc}\tPPL:{auc_roc_ppl}")
        print(f"Calculated AUC-PR score: loss:{auc_pr}\tPPL:{auc_pr_ppl}")

    except Exception:
        pass


if __name__ == "__main__":
    pass
