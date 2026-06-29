import glob
import itertools
import math
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from common.eval import Eval_entry, evaluate
from common.predict import estimate_threshold_from_predictions
from common.utils import IMAGES_FOLDER_PATH

plt.style.use("bmh")


def plot_roc_curve(eval_file_path: str, title=""):
    df = pandas.read_csv(eval_file_path)

    x = []
    y = []

    b_actual = 0
    b_actual_indx = 0
    best = 1
    colors = []

    for index, content in enumerate(df.iterrows()):
        _, row = content
        e = Eval_entry(row)

        try:
            FPR = e.get_FPR()
            TPR = e.get_TPR()

            # distanze euclidiana da 0,1
            dist = math.sqrt(FPR**2 + (TPR - 1) ** 2)

            if best > dist:
                best = dist
                b_actual = e
                b_actual_indx = index
        except ZeroDivisionError:
            pass

        colors.append("blue")
        x.append(FPR)
        y.append(TPR)

    print("ROC Nearest:", end="")
    b_actual.print()
    colors[b_actual_indx] = "red"

    fig, ax = plt.subplots()

    ax.scatter(x, y, color=colors)
    plt.title(f"ROC: {title}")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.plot([0, 1], [0, 1], "r--", alpha=0.75, zorder=0)
    plt.xlim(0, 1)  # Set x-axis
    plt.ylim(0, 1)
    ax.grid(True)
    plt.savefig(
        os.path.join(IMAGES_FOLDER_PATH, f"roc_{''.join(title.split(' '))}.pdf")
    )
    plt.show()


def plot_pr_curve(eval_file_path: str, title=""):
    df = pandas.read_csv(eval_file_path)

    x = []
    y = []

    b_actual = 0
    b_actual_indx = 0
    best = 1
    colors = []
    for index, content in enumerate(df.iterrows()):
        _, row = content
        e = Eval_entry(row)

        try:
            precision = e.get_precision()
            recall = e.get_recall()

            # distanze euclidiana da 1,1
            dist = math.sqrt((precision - 1) ** 2 + (recall - 1) ** 2)

            if best > dist:
                best = dist
                b_actual = e
                b_actual_indx = index
        except ZeroDivisionError:
            continue

        colors.append("blue")
        x.append(precision)
        y.append(recall)

    print("PR Nearest:", end="")
    b_actual.print()
    colors[b_actual_indx] = "red"

    fig, ax = plt.subplots()

    ax.scatter(x, y, color=colors)
    plt.title(f"PR: {title}")
    ax.set_xlabel("precision")
    ax.set_ylabel("recall")
    ax.plot([0, 1], [1, 0], "r--", alpha=0.75, zorder=0)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    ax.grid(True)
    plt.savefig(os.path.join(IMAGES_FOLDER_PATH, f"pr_{''.join(title.split(' '))}.pdf"))
    plt.show()


def plot_eval(
    eval_file_path: str | list[str],
    title="",
    x_metric="threshold_offset",
    metric_label="threshold",
    y_metric="f1",
    eval_files_labels: list[str] = None,
    colors_gradient=True,
    xlim=None,
    ylim=None,
):
    """
    Plot an evaluation file(s) with a graph
    :param eval_file_path: The evaluation file path or a list of evaluation files
    :param title: The title to add to the graph
    :param x_metric: The metric to be used as x axis
    :param metric_label: A label to be assigned to the x metric. By default x_metric itself
    :param y_metric: The metric to be used as y axis
    :param eval_files_labels: The labels to be used according to each evaluation file
    :param colors_gradient: Set to true to use a color gradiend, false is default colors.
    :param xlim: Used to limit the x axis interval
    :param ylim: Used to limit the y axis interval
    :return:
    """
    to_plot = eval_file_path if type(eval_file_path) is list else [eval_file_path]

    if colors_gradient:
        colormap = plt.get_cmap("magma")
        colors = [colormap(i) for i in np.linspace(0, 1, len(to_plot))]
    else:
        colors = None

    for indx, path in enumerate(to_plot):
        try:
            df = pandas.read_csv(path)
        except FileNotFoundError:
            print(f"Warning, file {path} not found, skipping..")
            continue

        x = []
        y = []

        x_estimated = None

        for index, content in enumerate(df.iterrows()):
            _, row = content
            e = Eval_entry(row)

            if e["estimated"] == True:
                if x_metric == "fn":
                    x_estimated = e.get_FNR()
                else:
                    x_estimated = row["threshold_offset"]
                continue

            if x_metric == "fn":
                x.append(e.get_FNR())
            else:
                x.append(row[x_metric])

            if y_metric == "f1":
                y.append(e.get_f1())
            elif y_metric == "accuracy":
                y.append(e.get_accuracy())
            elif y_metric == "fn":
                y.append(e.get_FNR())

        if eval_files_labels is None:
            plt.plot(x, y, color=colors[indx] if colors is not None else None)
        else:
            plt.plot(
                x,
                y,
                label=eval_files_labels[indx],
                color=colors[indx] if colors is not None else None,
            )

    if x_estimated is not None and len(to_plot) == 1:
        plt.axvline(x=x_estimated, color="green", linestyle=":")
        plt.text(
            x_estimated + 0.01,
            0.5,
            "Estimated",
            rotation=90,
            verticalalignment="center",
        )

    plt.title(title)
    ax = plt.gca()
    # ax.set_ylim([0, 1])
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.grid(True)
    plt.xlabel(x_metric if metric_label is None else metric_label)
    plt.ylabel(y_metric)
    if eval_files_labels is not None:
        plt.legend()

    plt.savefig(os.path.join(IMAGES_FOLDER_PATH, f"{''.join(title.split(' '))}.pdf"))
    plt.show()


def plot_confusion_matrix(
    e: Eval_entry,
    classes=["Benign", "Anomalous"],
    normalize=False,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    :param e: the evaluation entry to plot the confusion matrix from
    :param classes: classes names
    :param normalize: Set to true to apply normalization
    :param title: the title to assign to the plot
    :param cmap: Colors to use
    """
    # The structure is [[TN, FP], [FN, TP]]
    cm = np.array([[e.tn, e.fp], [e.fn, e.tp]])

    # Create the plot figure
    plt.figure(figsize=(8, 6.5))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # Set the tick marks and labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Normalize the matrix if specified
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Add text annotations to the cells
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], ".2f" if normalize else "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_FOLDER_PATH, f"confusion_{title.strip()}.pdf"))
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def plot_train_stats(model_folder: str, title: str):
    """
    Used to plot graphs of training and evaluation loss from model logs folder
    :param model_folder: The complete path of the model folder
    :param title: the title to assign to graphs
    :return:
    """
    log_path = os.path.join(model_folder, "logs")

    if not os.path.exists(log_path):
        print(f"Error, logs folder not found in {model_folder}")
        return

    # Find all event files in the log directory
    event_files = glob.glob(os.path.join(log_path, "events.out.tfevents.*"))

    if not event_files:
        print(f"Error, no event files found in {log_path}")
        return

    # Find the most recent event file based on modification time
    latest_event_file = max(event_files, key=os.path.getmtime)

    ea = event_accumulator.EventAccumulator(
        latest_event_file,
        size_guidance={event_accumulator.SCALARS: 0},  # 0 means load all scalars
    )

    # Load the events from disk
    ea.Reload()

    loss_events = ea.Scalars("train/loss")
    grad_norm_events = ea.Scalars("train/grad_norm")
    try:
        eval_loss_events = ea.Scalars("eval/loss")
        eval_roc_events = ea.Scalars("eval/auc_roc")
    except KeyError:
        eval_loss_events = None
        print("No eval data found, skipping..")

    # Convert to a more usable format like a DataFrame
    loss_df = pandas.DataFrame(
        [(e.step, e.value) for e in loss_events], columns=["step", "loss"]
    )
    grad_norm_df = pandas.DataFrame(
        [(e.step, e.value) for e in grad_norm_events], columns=["step", "grad_norm"]
    )

    if eval_loss_events is not None:
        eval_loss_df = pandas.DataFrame(
            [(e.step, e.value) for e in eval_loss_events], columns=["step", "eval_loss"]
        )
        # eval_roc_df = pandas.DataFrame([(e.step, e.value) for e in eval_roc_events], columns=["step", "eval_roc"])

    eval_roc_df = None

    plt.figure(figsize=(10, 4))

    # Plotting the Training Loss
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(loss_df["step"], loss_df["loss"])
    plt.title(f"Training Loss {title}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)

    # Plotting the eval loss
    if eval_loss_events is not None:
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        plt.plot(eval_loss_df["step"], eval_loss_df["eval_loss"], color="orange")
        plt.title("Eval loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)

    if eval_roc_df is not None:
        plt.subplot(2, 2, 3)
        plt.plot(eval_roc_df["step"], eval_roc_df["eval_roc"], color="orange")
        plt.title("Eval ROC AUC")
        plt.xlabel("Step")
        plt.ylabel("AUC")
        plt.grid(True)

    plt.tight_layout()  # Adjusts subplot params for a tight layout.
    plt.savefig(os.path.join(IMAGES_FOLDER_PATH, f"{title}.pdf"))
    plt.show()


def plot_prediction_samples(
    prediction_file_path: str,
    truth_file_path: str = None,
    window: int = 1000,
    w_offset: int = 1000,
    metric: str = "std_dev",
    title: str = "prediction",
    sort: bool = False,
    plot_type: str = "scatter",
    hide_anomalous: bool = False,
    hide_normal=False,
):
    """
    Function used to plot a scatterplot of the prediction samples to visualize the distribution
    :param prediction_file_path:
    :param truth_file_path:
    :param window:
    :param w_offset:
    :param metric:
    :param title:
    :param sort:
    :param plot_type:
    :param hide_anomalous:
    :param hide_normal:
    :return:
    """
    df_pred = pandas.read_csv(prediction_file_path)

    if truth_file_path is not None:
        df_truth = pandas.read_csv(truth_file_path)
        df_pred = pandas.merge(df_pred, df_truth, on="original_index", how="inner")

    if hide_anomalous:
        df_pred = df_pred[df_pred["anomalous"] != 1]
    if hide_normal:
        df_pred = df_pred[df_pred["anomalous"] != 0]

    if window != 0:
        df_split = df_pred[w_offset : window + w_offset]
    else:
        df_split = df_pred

    if sort:
        df_split = df_split.sort_values([metric], ignore_index=True)

    # Assign colors to types of samples
    colors = []
    for i, row in df_split.iterrows():
        if truth_file_path is not None:
            if row["anomalous"] == 1:
                colors.append("red")
                continue

        colors.append("gray")

    plt.figure(figsize=(18, 14), dpi=80)
    if plot_type == "scatter":
        x = np.arange(len(df_split))
        plt.scatter(x, df_split[metric], c=colors)
    elif plot_type == "hist":
        plt.hist(df_split[metric], bins=140, edgecolor="black", alpha=0.7)
    else:
        print("Invalid plot type selected")
        return

    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("sample_index")
    plt.savefig(os.path.join(IMAGES_FOLDER_PATH, f"{title.strip()}.pdf"))
    plt.show()


def estimate_and_evaluate(
    predictions_file_eval: str,
    k: float,
    truth_dataset_path: str,
    predictions_file_test_set: str = None,
):
    """
    Estimatesthe threshold with the given predictions and k, and then does evaluation and get the FPR on the eval set. Optionally, if predictions_file_test_set is given, it computes also f1 score from the test set
    """
    estimated_threshold = estimate_threshold_from_predictions(
        predictions_file_eval, k=k
    )

    fpr = -1
    f1 = -1

    params = [(3000, estimated_threshold, "loss", True, False, 0.0, True, True, True)]
    res = evaluate(
        predictions_file_eval,
        truth_dataset_path,
        params,
        "tmp",
        log_all=False,
        print_to_console=False,
    )
    e = Eval_entry(res.iloc[0])
    fpr = e.get_FPR().item()

    if predictions_file_test_set is not None:
        # test set
        res = evaluate(
            predictions_file_test_set,
            truth_dataset_path,
            params,
            "tmp",
            log_all=False,
            print_to_console=False,
        )
        e = Eval_entry(res.iloc[0])
        f1 = e.get_f1()

    return (fpr, f1)


def plot_theshold_k_estimation(
    predictions_file_eval, truth_dataset_path: str, predictions_file_test=None
):
    """
    Plots the FPR (and optionally F1 score) with respect the chosen k for threshold computation
    """
    ks = []
    fprs = []
    f1s = []

    if True:
        inputs = []

        for i in range(1, 200, 2):
            k = i / 100
            ks.append(k)
            inputs.append(
                (predictions_file_eval, k, truth_dataset_path, predictions_file_test)
            )

        with multiprocessing.Pool(12) as pool:
            results = pool.starmap(estimate_and_evaluate, inputs)

        for i, j in results:
            fprs.append(i)
            if j != -1:
                f1s.append(j)

    print(fprs)
    print(ks)

    df = pd.DataFrame({"k": ks, "FPR": fprs})

    df.to_csv("out.csv", index=False)

    fig, ax1 = plt.subplots()

    ax1.plot(ks, fprs)
    ax1.set_xlabel("k")
    ax1.set_ylabel("FPR")

    ax1.grid(True, linestyle="--", alpha=0.6, color="gray")

    if len(f1s) > 0:
        ax2 = ax1.twinx()

        ax2.set_ylabel("Test set F1 score", color="tab:orange")
        ax2.plot(ks, f1s, color="tab:orange", linestyle="--")

    plt.savefig("test.pdf")
    """
    first_derivative_fpr = np.gradient(np.array(fprs), np.array(ks))
    #first_derivative_fpr = savgol_filter(np.array(fprs), window_length=10, polyorder=5, deriv=1, delta = ks[1] - ks[0])

    axs[1].plot(ks, first_derivative_fpr)
    axs[1].set_ylabel("first derivative FPR")
    axs[1].set_xlabel("k")

    second_derivative_fpr = np.gradient(first_derivative_fpr, np.array(ks))
    #second_derivative_fpr = savgol_filter(np.array(fprs), window_length=10, polyorder=5, deriv=2, delta = ks[1] - ks[0])

    ax_bottom= axs[2]
    ax_bottom.plot(ks, second_derivative_fpr)
    ax_bottom.set_ylabel("second derivative FPR")
    ax_bottom.set_xlabel("k")

    plt.savefig("derivative.png", dpi=300)
    """
