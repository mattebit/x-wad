import ast
import multiprocessing

import numpy as np
import pandas
from pandas import Series
from sklearn import metrics
from sklearn.metrics import average_precision_score

from common.utils import get_cpu_count


class Eval_entry(Series):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    is_estimated = False

    def get_fp(self):
        return self.fp

    def get_fn(self):
        return self.fn

    def get_tp(self):
        return self.tp

    def get_tn(self):
        return self.tn

    def __init__(self, entry):
        super().__init__(entry)
        self.fp = self["false_positives"]
        self.fn = self["false_negatives"]
        self.tp = self["true_positives"]
        self.tn = self["true_negatives"]
        self.is_estimated = self["estimated"]

    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def get_TPR(self):
        if (self.tp + self.fn) == 0:
            return 0

        return self.tp / (self.tp + self.fn)

    def get_FPR(self):
        if (self.fp + self.tn) == 0:
            return 0

        return self.fp / (self.fp + self.tn)

    def get_precision(self):
        if (self.fp + self.tn) == 0:
            return 0

        return self.tp / (self.tp + self.fp)

    def get_recall(self):
        if (self.tp + self.fn) == 0:
            return 0

        return self.tp / (self.tp + self.fn)

    def get_f1(self):
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)

    def get_FNR(self):
        if (self.fn + self.tp) == 0:
            return 0

        return self.fn / (self.fn + self.tp)

    def get_TNR(self):
        if (self.tn + self.fp) == 0:
            return 0

        return self.tn / (self.tn + self.fp)

    def print(self):
        print(
            f"F1:{self.get_f1():.2%} "
            f"ACC:{self.get_accuracy():.2%} "
            f"FPR:{self.get_FPR():.2%} "
            f"TPR:{self.get_TPR():.2%} "
            f"FNR:{self.get_FNR():.2%} "
            f"TNR:{self.get_TNR():.2%} "
            f"Dev_mult:{self['std_multiplier']} "
            f"Window:{self['window_size']}"
        )


def upside_deviation(data: list[int | float], target=None):
    """
    Calculates the upside deviation of a list of values
    Upside deviation measures volatility of returns above a target return.

    :param data: the list from which calculate the deviation
    :param target: wheter to use a custom target or defaults to average

    :return The upside deviation of the data.
    """

    data = np.asarray(data)

    if target is None:
        target = np.mean(data)

    # There must be at least 2 data points to calculate sample deviation
    if len(data) < 2:
        return 0.0

    # 1. Calculate deviations from the target
    deviations = data - target

    # 2. Keep only positive deviations (upside), setting negative ones to 0
    upside_deviations = np.maximum(0, deviations)

    # 3. Square the upside deviations, sum them, and divide by n-1
    upside_variance = np.sum(upside_deviations**2) / (len(data) - 1)

    # 4. The upside deviation is the square root of the upside variance
    return np.sqrt(upside_variance)


def evaluate_anomalies_parallel(
    df_prediction: pandas.DataFrame,
    window_size: int,
    threshold_offset: float,
    metric_str: str = "loss",
    average_anomalies: bool = False,
    use_std_as_offset: bool = False,
    std_multiplier: float = 1,
    log_fp_indexes: bool = False,
    log_fn_indexes: bool = False,
    estimated: bool = False,
):
    """
    Evaluate the given prediction file against the truth dataset.

    :param df_prediction: A dataframe containing the predictions of a model merged with the truth value
    :param window_size: The window size to use
    :param threshold_offset: The threshold above the average to consider a sample anomalous
    :param metric_str: The metric to use "std_dev", "perplexity" or "loss"
    :param average_anomalies: If set to true, anomalies are used to compute average
    :param use_std_as_offset: If set to true, calculate std deviation of samples and use it as sum of threshold
    :param std_multiplier: Multiply the STD or the UPD with this multiplier
    :param use_average: if to use average of samples during calculation
    :param log_fp_indexes: Set to true to log the false positive indexes
    :param log_fn_indexes: Set to true to log the false negatives indexes
    :param estimated: Set to true to set that the current setting has been estimated to be later plotted differently

    :return: a tuple with different metrics for evaluation
    """

    # TODO: remove all unused methods to adjust threshold

    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    correct = 0
    fp_indexes = []
    fn_indexes = []
    calculated_thresholds = []

    if use_std_as_offset:
        # Calculate beforehand the rolling STD
        average_std = df_prediction[metric_str].std()

        rolling_std = df_prediction[metric_str].rolling(window=window_size).std()
        rolling_std = rolling_std.fillna(
            average_std
        )  # Replace NaNs with average STD. In real world this will be done with prior data
        df_prediction["rolling_std"] = rolling_std

    for index, prediction in df_prediction.iterrows():
        metric_value = prediction[metric_str]
        is_normal = prediction["anomalous"] == 0

        calculated_threshold = threshold_offset + (
            (prediction["rolling_std"] if use_std_as_offset else 0) * std_multiplier
        )
        calculated_thresholds.append(calculated_threshold)

        classified_anomaly = metric_value > calculated_threshold

        if classified_anomaly and is_normal:
            if log_fp_indexes:
                fp_indexes.append(prediction["original_index"])
            false_positives += 1

        if not classified_anomaly and is_normal:
            correct += 1
            true_negatives += 1

        if not classified_anomaly and not is_normal:
            if log_fn_indexes:
                fn_indexes.append(prediction["original_index"])
            false_negatives += 1

        if classified_anomaly and not is_normal:
            correct += 1
            true_positives += 1

    return (
        window_size,
        threshold_offset,
        metric_str,
        average_anomalies,
        use_std_as_offset,
        std_multiplier,
        np.array(calculated_thresholds).mean().item(),
        false_positives,
        false_negatives,
        true_negatives,
        true_positives,
        correct,
        fp_indexes,
        fn_indexes,
        estimated,
    )


def evaluate(
    prediction_csv_path: str,
    csv_truth: str,
    params: list,
    results_file_path: str,
    log_all=True,
    print_to_console=True,
):
    """
    Evaluate the predictions from the given prediction file and save the results to a file
    :param prediction_csv_path: the path of the prediction files
    :param csv_truth: the truth dataset
    :param params: The parameters to evaluate the predictions, se inputs of evaluate_anomalies_parallel
    :param results_file_path: the path of the file where to save results
    :param log_all: set to true to log also FP and FN indexes
    :return: nothing
    """
    df_pred = pandas.read_csv(prediction_csv_path, index_col=False)
    df_truth = pandas.read_csv(csv_truth, index_col=False)

    df = pandas.merge(df_pred, df_truth, on="original_index", how="inner")

    inputs = []

    if len(params) > 1:
        # If there are more than one evaluation parallelize, otherwise avoid multiprocessing
        for i in params:
            inputs.append((df,) + i)

        with multiprocessing.Pool(get_cpu_count()) as pool:
            results = pool.starmap(evaluate_anomalies_parallel, inputs)

    else:
        results = [evaluate_anomalies_parallel(*((df,) + params[0]))]

    df_out = pandas.DataFrame(results)
    df_out.columns = [
        "window_size",
        "threshold_offset",
        "metric",
        "average_anomalies",
        "use_std_as_offset",
        "std_multiplier",
        "calculated_thresholds_avg",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "true_positives",
        "correct",
        "fp_indexes",
        "fn_indexes",
        "estimated",
    ]

    if not log_all:
        df_out = df_out.drop(
            columns=[
                "fp_indexes",
                "fn_indexes",
                "average_anomalies",
                "use_std_as_offset",
            ]
        )

    df_out.to_csv(results_file_path)

    if print_to_console:
        print_from_file(results_file_path)

    return df_out


def get_best_result(eval_file_path: str, by: str = "f1") -> Eval_entry:
    """
    Return the best result by the selected metric
    :param eval_file_path: the path of the evaluation file to get the best result from
    :param by: f1, or acc, or <fn
    :return:
    """

    df = pandas.read_csv(eval_file_path)

    df = df.sort_values("correct")

    best_f1 = 0
    best_acc = 0
    best_fn = 1.0
    best_f1_fn = 0.0

    best_entry = None

    for index, content in enumerate(df.iterrows()):
        _, row = content
        e = Eval_entry(row)

        f1 = e.get_f1()
        acc = e.get_accuracy()
        fn = e.get_FNR()
        f1fn = f1 - fn

        if by == "estimated" and e.is_estimated:
            return e

        if by == "f1" and f1 > best_f1:
            best_f1 = f1
            best_entry = e

        if by == "acc" and acc > best_acc:
            best_acc = acc
            best_entry = e

        if by == "fn" and fn <= best_fn:
            best_fn = fn
            best_entry = e

        if by == "f1fn" and f1fn > best_f1_fn:
            best_f1_fn = f1fn
            best_entry = e

    return best_entry


def print_from_file(prediction_csv_path: str, all_=False):
    """
    Print an evaluation csv file to visualize the bes results
    :param prediction_csv_path: The path of the file
    :param all_: print all the evaluations
    """
    df = pandas.read_csv(prediction_csv_path)
    df = df.sort_values(["correct", "false_negatives", "false_positives"])

    best_f1_indx = 0
    best_f1_val = 0
    best_acc_indx = 0
    best_acc_val = 0

    estimated = None

    for index, i in enumerate(df.iterrows()):
        _, entry = i
        e = Eval_entry(entry)

        if entry["estimated"] == True:
            estimated = e

        f1 = e.get_f1()
        if f1 > best_f1_val:
            best_f1_indx = index
            best_f1_val = f1

        acc = e.get_accuracy()
        if acc > best_acc_val:
            best_acc_val = acc
            best_acc_indx = index

        if all_:
            e.print()

    print("Best F1:")
    Eval_entry(df.iloc[best_f1_indx]).print()

    if estimated is not None:
        print("Estimated:")
        Eval_entry(estimated).print()

    e = get_best_result(prediction_csv_path, "f1fn")
    if e is not None:
        print("Best F1FN:")
        e.print()

    e = get_best_result(prediction_csv_path, "fn")
    if e is not None:
        print("Lowest FN:")
        e.print()


def get_fn_indexes(entry: Eval_entry):
    try:
        fn_indexes = ast.literal_eval(entry["fn_indexes"])
    except Exception:
        return None

    return fn_indexes


def get_fp_indexes(entry: Eval_entry):
    try:
        fp_indexes = ast.literal_eval(entry["fp_indexes"])
    except Exception:
        return None

    return fp_indexes


def get_AUC_ROC_from_predictions(prediction_file_path: str, truth_file_path: str):
    """
    Get the AUC_ROC and AUC_PR scores from the predictions and the dataset truth file
    :param prediction_file_path:
    :param truth_file_path:
    :return:
    """
    df_pred = pandas.read_csv(prediction_file_path)
    df_truth = pandas.read_csv(truth_file_path)

    df = pandas.merge(df_pred, df_truth, on="original_index")

    all_labels = df["anomalous"]
    all_loss = df["loss"]

    auc_roc = metrics.roc_auc_score(all_labels, all_loss)
    auc_pr = average_precision_score(all_labels, all_loss)

    auc_roc_ppl = metrics.roc_auc_score(all_labels, df["perplexity"])
    auc_pr_ppl = average_precision_score(all_labels, df["perplexity"])

    return auc_roc, auc_pr, auc_roc_ppl, auc_pr_ppl
