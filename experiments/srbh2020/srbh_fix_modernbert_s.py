import os

from common.eval import print_from_file
from common.execute import execute
from common.plot import (
    plot_prediction_samples,
    plot_eval,
    plot_roc_curve,
    plot_pr_curve,
    plot_train_stats,
)
from common.utils import (
    MODELS_FOLDER_PATH,
    split_dataset,
    default_split_perc,
    get_dataset_split_paths_names,
    get_model_related_paths,
    get_argument_parser_experiments,
    model_load_settings_optimized,
    ALLOW_OPTIMIZATIONS,
    model_load_settings_normal,
)
from experiments.srbh2020.tools import preprocess, fix_srbh, results_based_on_class_plot


def run_experiment(
    THIS_DATASET_PATH: str,
    MODELS_FOLDER_PATH=MODELS_FOLDER_PATH,
    dataset_prefix="fix-modernbert",
    do_preprocess=False,
    do_split=False,
    do_train=True,
    do_test=True,
    do_val=False,
    do_plot=False,
    allow_optimizations=ALLOW_OPTIMIZATIONS,
):
    TRUTH_DATASET_NAME = "srbhfix.requests"
    TRUTH_DATASET_PATH = os.path.join(THIS_DATASET_PATH, TRUTH_DATASET_NAME)

    if do_preprocess:
        # preprocess dataset
        preprocess(
            os.path.join(THIS_DATASET_PATH, "original", "data_capec_multilabel.csv"),
            os.path.join(THIS_DATASET_PATH, "tmp.requests"),
        )

        # generate fixed dataset
        fix_srbh(os.path.join(THIS_DATASET_PATH, "tmp.requests"), TRUTH_DATASET_PATH)

    if do_split:
        split_dataset(
            TRUTH_DATASET_NAME,
            **default_split_perc,
            dataset_folder_path=THIS_DATASET_PATH,
            prefix_str=dataset_prefix,
        )

    (
        TRAIN_DATASET_PATH,
        TRAIN_DATASET_NAME,
        VALIDATION_DATASET_PATH,
        VALIDATION_DATASET_NAME,
        TEST_DATASET_PATH,
        TEST_DATASET_NAME,
    ) = get_dataset_split_paths_names(TRUTH_DATASET_PATH, dataset_prefix)

    base_model = "ModernBERT-large"
    OUTPUT_MODEL_NAME = f"{base_model}-s-srbh-fix"
    (
        EVALUATIONS_FOLDER_PATH,
        PREDICTIONS_FOLDER_PATH,
        PREDICT_FILE_PATH,
        EVAL_FILE_PATH,
    ) = get_model_related_paths(
        MODELS_FOLDER_PATH, OUTPUT_MODEL_NAME, TEST_DATASET_NAME
    )

    execute(
        TRAIN_DATASET_PATH,
        OUTPUT_MODEL_NAME,
        models_folder_path=MODELS_FOLDER_PATH,
        base_model_name=os.path.join(MODELS_FOLDER_PATH, base_model),
        do_train_from_scratch=do_train,
        do_predict=do_test,
        do_evaluate=do_val,
        train_from_scratch_args={
            "learning_rate": 8e-4,
            "lr_scheduler_type": "linear",
            "warmup_steps": 500,
            "num_train_epochs": 100,
            "per_device_train_batch_size": 50,
            "gradient_accumulation_steps": 8,  # virtually increase batch size total should be 512 or more
            "per_device_eval_batch_size": 50,
            "mlm_probability": 0.3,
            "eval_strategy": "steps",
            "eval_steps": 200,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "early_stop": True,
            "weight_decay": 0.01,
        },
        predict_dataset_path=TEST_DATASET_PATH,
        validate_dataset_path=VALIDATION_DATASET_PATH,
        predict_batch_size=50,
        truth_dataset_path=TRUTH_DATASET_PATH,
        auto_threshold_est_method="gaussian",
        task="mlm",
        model_load_settings=model_load_settings_optimized
        if allow_optimizations
        else model_load_settings_normal,
    )

    generic_title = f"f {base_model} SRBH fix mlm"

    if do_plot:
        try:
            plot_prediction_samples(
                PREDICT_FILE_PATH,
                TRUTH_DATASET_PATH,
                window=3000,
                sort=True,
                title=generic_title,
                hide_anomalous=False,
                hide_normal=False,
                metric="loss",
            )
        except Exception as e:
            print(e)

        print_from_file(EVAL_FILE_PATH)

        plot_roc_curve(EVAL_FILE_PATH, generic_title)
        plot_pr_curve(EVAL_FILE_PATH, generic_title)
        plot_train_stats(
            os.path.join(MODELS_FOLDER_PATH, OUTPUT_MODEL_NAME), generic_title
        )
        plot_eval(EVAL_FILE_PATH, title=f"{generic_title} F1-FN", x_metric="fn")
        plot_eval(EVAL_FILE_PATH, title=f"{generic_title} F1-threshold")

        results_based_on_class_plot(
            EVAL_FILE_PATH,
            TRUTH_DATASET_PATH,
            plot=True,
            title="SRBH fix mlm evaluation: False Negatives by anomaly class",
        )


if __name__ == "__main__":
    print(os.path.basename(__file__))
    parser = get_argument_parser_experiments()
    args = parser.parse_args()
    run_experiment(
        args.THIS_DATASET_PATH,
        args.models_folder_path,
        do_preprocess=args.do_preprocess,
        do_split=args.do_split,
        do_test=args.do_test,
        do_train=args.do_train,
        do_val=args.do_val,
        do_plot=args.do_plot,
    )
