import os
import re
import sys

import pandas
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from common.eval import get_best_result, get_fn_indexes, get_fp_indexes
from common.utils import IMAGES_FOLDER_PATH


def reconstruct_http_request(row) -> str | None:
    """
    Reconstructs a single HTTP request string from a row of a Pandas DataFrame with columns of a SRBH dataset.

    :param row: A row from the DataFrame containing the log data.
    :return: str: A string representing the HTTP request. None if errors occurred
    """
    # build first request row
    try:
        method = row.get('request_http_method')
        uri = row.get('request_http_request')
        protocol = row.get('request_http_protocol')
    except KeyError:
        return None

    if pd.isna(method) or pd.isna(uri) or pd.isna(protocol):
        return None  # Skip rows with incomplete essential data

    request_line = f"{method} {uri} {protocol}"

    # build headers
    headers = []
    header_mapping = {
        'request_host': 'Host',
        'request_user_agent': 'User-Agent',
        'request_accept': 'Accept',
        'request_accept_language': 'Accept-Language',
        'request_accept_encoding': 'Accept-Encoding',
        'request_referer': 'Referer',
        'request_origin': 'Origin',
        'request_cookie': 'Cookie',
        'request_content_type': 'Content-Type',
        'request_do_not_track': 'DNT',
        'request_connection': 'Connection',
    }

    for col_name, header_name in header_mapping.items():
        # Check if the column exists and its value is not null/NaN
        if col_name in row and pd.notna(row[col_name]):
            headers.append(f"{header_name}: {row[col_name]}")

    # Append request body
    try:
        body = row.get('request_body')
    except KeyError:
        return None

    # if body is NaN set empty string
    if pd.isna(body):
        body = ''

    # Assemble request
    full_request_parts = [request_line] + headers

    # Add a blank line between headers and body, as per HTTP specification
    full_request_parts.append('')

    # Add the body if it exists
    if body:
        full_request_parts.append(str(body))

    return "[r][n]".join(full_request_parts)  # rebuild the request inline


ALL_ANO_CLASS = [
    "272 - Protocol Manipulation",
    "242 - Code Injection",
    "88 - OS Command Injection",
    "126 - Path Traversal",
    "66 - SQL Injection",
    "16 - Dictionary-based Password Attack",
    "310 - Scanning for Vulnerable Software",
    "153 - Input Data Manipulation",
    "248 - Command Injection",
    "274 - HTTP Verb Tampering",
    "194 - Fake the Source of Data",
    "34 - HTTP Response Splitting",
    "33 - HTTP Request Smuggling"
]


def preprocess(input_file, output_file) -> None:
    """
    Takes the SRBH original csv file and converts it to a standard csv format, it also reconstructs HTTP requests from
    the original tabular format.
    The assigned index in the resulting output file will be referred to the original dataset index
    :param input_file: the original srbh csv file
    :param output_file: the output csv file
    """
    print(f"[*] Reading data from '{input_file}'...")

    try:
        df = pd.read_csv(input_file, low_memory=False)
    except FileNotFoundError:
        print(f"[ERROR] The file '{input_file}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An error occurred while reading the CSV: {e}")
        sys.exit(1)

    reconstructed_requests = []
    for index, (_, row) in tqdm(enumerate(df.iterrows()), total=(len(df)), desc="processing.."):
        request_str = reconstruct_http_request(row)

        if request_str is None:
            continue

        ano_class = "normal"

        if row["000 - Normal"] == 0:
            # if anomalous, save anomaly class to dedicated column
            for c_name in ALL_ANO_CLASS:
                if row[c_name] == 1:
                    ano_class = c_name
                    break

        if request_str:
            reconstructed_requests.append({
                "original_index": index + 1,  # start counting by 1
                "anomalous": 1 if row["000 - Normal"] == 0 else 0,  # invert anomalous flag logic
                "ano_class": ano_class,
                "request": request_str
            })

    if not reconstructed_requests:
        print("[WARNING] No valid HTTP requests could be reconstructed.")
        return

    print(f"[*] Saving {len(reconstructed_requests)} reconstructed requests to '{output_file}'...")

    try:
        df_out = pandas.DataFrame(reconstructed_requests)
        df_out.to_csv(output_file, index=False)
    except Exception as e:
        print(f"[ERROR] Could not write to output file '{output_file}': {e}")
        sys.exit(1)

    print(f"[SUCCESS] Processing complete. Output saved to '{output_file}'.")


def results_based_on_class_plot(
        eval_files,
        truth_file,
        plot=False,
        title="",
        labels=["false negatives"]
):
    """
    Plots the False Negatives results based on anomaly class of a specific evaluation file or files. THe truth file
    is the reference file for which the percentage are calculated. It can be the full dataset or the evaluation dataset.
    """

    if isinstance(eval_files, str):
        eval_files = [eval_files]

    df_truth = pd.read_csv(truth_file)

    # Filter out 'normal' and NaN classes
    df_truth_filtered = df_truth[
        (df_truth["ano_class"] != "normal") &
        (df_truth["ano_class"].notna())
        ].copy()

    all_stats_list = []

    # Loop through each evaluation file
    for i, f_path in enumerate(eval_files):
        label = labels[i]

        e = get_best_result(f_path, "estimated")

        fn_indexes = set(get_fn_indexes(e))
        fp_indexes = set(get_fp_indexes(e))

        current_df = df_truth_filtered.copy()

        current_df["is_fn"] = current_df["original_index"].isin(fn_indexes)
        current_df["is_fp"] = current_df["original_index"].isin(fp_indexes)

        # Group by class and calculate counts
        stats = current_df.groupby("ano_class").agg(
            total=("original_index", "count"),
            fn_count=("is_fn", "sum"),
            fp_count=("is_fp", "sum")
        ).reset_index()

        # Calculate percentages
        stats["fn_rate"] = (stats["fn_count"] / stats["total"])
        stats["Model"] = label

        stats = stats[stats["fn_rate"] > 0.001]

        all_stats_list.append(stats)

    # Combine all results
    final_df = pd.concat(all_stats_list, ignore_index=True)

    # Sort
    final_df = final_df.sort_values(["fn_rate"], ascending=False)  # ["ano_class", "Model"])

    # Print console
    print(f"\n--- Analysis for {len(eval_files)} files ---")
    print(final_df[["Model", "ano_class", "total", "fn_count", "fn_rate"]].to_string(index=False))
    # final_df.to_csv("original_fix_fnr_comparison.csv")

    if plot:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(14, 7))

        ax = sns.barplot(
            data=final_df,
            x="ano_class",
            y="fn_rate",
            hue="Model",  # it groups bars by the 'Model' column
            palette="viridis",
            edgecolor="black"
        )

        plt.title(title if title else "False Negative Rate per Class Comparison", fontsize=15)
        plt.xlabel("Anomaly Class", fontsize=12)
        plt.ylabel("Per-class FN (%)", fontsize=12)
        plt.xticks(rotation=30, ha="right")

        for container in ax.containers:
            # map back to the dataframe to get the absolute counts
            # assumes the container order matches the df iteration order for that hue.

            for bar in container:
                height = bar.get_height()
                if height > 0:
                    label = f"{height:.2%}"

                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + (height * 0.02),  # slight offset
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="black",
                        rotation=0
                    )

        plt.legend()
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(os.path.join(IMAGES_FOLDER_PATH, f"anomaly_results_comparison.pdf"))
        plt.show()


def anomaly_per_class_plot(
        truth_file,
        plot=False,
        title=""
):
    """
    Plot the number of anomalies for each class in a bar plot.
    """
    df_truth = pd.read_csv(truth_file)

    # Filter out 'normal' and NaN classes
    df_filtered = df_truth[
        (df_truth["ano_class"] != "normal") &
        (df_truth["ano_class"].notna())
        ].copy()

    # Group by class and calculate counts
    stats = df_filtered.groupby("ano_class").agg(
        total=("original_index", "count")
    ).reset_index()

    stats = stats[stats["fn_rate"] > 0.001]

    # Sort by Total count descending
    stats = stats.sort_values("total", ascending=False)

    if plot:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 6))

        sns.barplot(
            data=stats,
            x="ano_class",
            y="total",
            color="lightgray",
            label="Anomalies",
            edgecolor="black"
        )

        plt.title(title, fontsize=15)
        plt.xlabel("Anomaly Class", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=30, ha="right")  # Rotate labels

        height_col = "total"
        # Add percentage text labels on top of the bars
        for i, (ind, row) in enumerate(stats.iterrows()):
            if row[height_col] > 0:
                label = f"{int(row['total'])}"
                bar_height = row[height_col]
                plt.text(
                    x=i,
                    y=bar_height + (bar_height * 0.02),
                    s=label,
                    color='black',
                    ha="center",
                    fontsize=9
                )

        plt.legend()
        plt.subplots_adjust(bottom=0.35)

        plt.savefig(os.path.join(IMAGES_FOLDER_PATH, f"anomaly_results.pdf"))
        plt.show()


def fix_srbh(original_path, output_fix, output_mislabeled="mislabeled_indexes.csv") -> None:
    """
    Takes the processed original SRBH dataset, and relabels samples that contains unusual patterns.
    :param original_path: The source original SRBH dataset (already processed)
    :param output_mislabeled: a csv file containing the indexes of only the samples containing unusual patterns
    :param output_fix: the file path where to save the fixed dataset
    """
    df = pandas.read_csv(original_path)
    df_fix = df

    patterns = [
        "sleep(15)",
        "User-Agent: ( ",
        "User-Agent: ) ",
        "User-Agent: + ",
        "User-Agent: '\"\" ",
        "User-Agent: ' ",
        "User-Agent: \"\" ",
        "User-Agent: any ",
        "User-Agent: any? ",
        "Referer: \"\"\'",
        "Referer: <!-- ",
        "Referer: ]]> ",
        "Firefox/70.0\"\" ",
        "Firefox/70.0\' ",
        "Firefox/70.0\'\"\" ",
        "Firefox/70.0( ",
        "Firefox/70.0) ",
        "Firefox/70.0; ",
        "Firefox/70.0NULL ",
        ".chr(",
        "--",
        "<!--#EXEC"
        "c%3A%5C",
        "c%3A%2F",
        "..\\",
        "../",
        "/.svn",
        "/.git",
        "Set-cookie%3A+Tamper",
        "%3C%21",
        "boot.ini,"
        "XYZABCDEFGHIJ",
        "%5D%5D%3E",
        "${",
        "//",
        "owasp.org",
        "passwd",
        "/blog/Set-cookie",
        "/blog/c%3A%2F",
        "/blog/c%3A%5C",
        "%20-%20Copy",
        "SELECT+SLEEP",
        "ORDER+BY",
        "0W45pz4p",
        "ORDER+BY",
        # fuzzing stuff
        "zApPX",
        "zApPX107sS",
        "030W45pz4p",
        "/%40",
        "/%22",
        "/%27",
        "/%28",
        "/%29",
        "/%2B",
        "/%2F",
        "/%3B",
        "/%40",
        "/%7C",
        "/%5C",
        "/blog%25",
        "/blog%26",
        "/blog%27",
        "/blog%3B",
        "/blog%60",
        "/blog%7C",
        "/blog+%2F",
        "/blog/%00",
        "/blog/%22",
        "/blog%22"
        "/blog/%27",
        "/blog/%28",
        "/blog/%2B",
        "/blog/%2F",
        "/blog/%3C",
        "/blog/%40",
        "/blog/%5C",
        "/blog/%7C",
        "/blog/index.php%22",
        "/blog/index.php/%24"
        "/blog/index.php%25",
        "/blog/index.php%26",
        "/blog/index.php%27",
        "/blog/index.php%3B",
        "/blog/index.php%60",
        "/blog/index.php%7C",
        "/blog/index.php%29",
        "/blog/index.php+%2F",
        "/blog/index.php/%00",
        "/blog/index.php/%22",
        "/blog/index.php/%27",
        "/blog/index.php/%28",
        "/blog/index.php/%2F",
        "/blog/index.php/%3C",
        "/blog/index.php/%5C",
        "/blog/index.php/2020%22",
        "/blog/index.php/2020%25",
        "/blog/index.php/2020%25",
        "/blog/index.php/2020%26",
        "/blog/index.php/2020%27",
        "/blog/index.php/2020%3B",
        "/blog/index.php/2020%60",
        "/blog/index.php/2020%7C",
        "/blog/index.php/2020+%2F",
        "/blog/index.php/2020/%00",
        "/blog/index.php/2020/%22",
        "/blog/index.php/2020/%27",
        "/blog/index.php/2020/%28",
        "/blog/index.php/2020/%2F",
        "/blog/index.php/2020/%3C",
        "/blog/index.php/2020/%5C",
        "/blog/index.php/2020/03%",
        "/blog/index.php/2020/03/23%",
        "/blog/index.php/2020/03/27/%",
        "/blog/index.php/2020/03/22+%2F",
        "/blog/index.php/2020/03/22%",
        "/blog/index.php/2020/03/27/qui-ratione-maxime-dolores-consequatur/%,"
        "%00",
        "/blog/index.php/2020/04/%",
        "/blog/index.php/2020/04/%",
        "/blog/index.php/2020/04/04%"
    ]

    re_patterns = [
        # "[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}" # fuzzer's UUID (causes more FP)
        r"(GET|POST|PUT)\\s(\\/[^\\n\\?]*?)([\\+/])?%(20|22|24|25|26|27|28|3B|60|7C|2F|00|2B|3C|40|5C|28|29).*HTTP\\/"
        # all the fuzzing
        # r"(GET|POST|PUT)\s(\/[^\n\?]*?)([\+/])?%(22|24|25|26|27|28|3B|60|7C|2F|00|2B|3C|40|5C|28|29).*HTTP\/"
        r"(\/.+)+(\.bak|\.backup|\.log|\.swp|~|\.old)\s"  # protocol manipulation
    ]

    compiled_patterns = [re.compile(p) for p in re_patterns]

    wrong_fp_indexes = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Fixing labels.."):
        if row["anomalous"] == 1:
            # if already flagged as anomalous skip
            continue

        req_str = row["request"]
        indx = row["original_index"]

        # Check for pattern in request string
        if any(pattern in req_str for pattern in patterns):
            wrong_fp_indexes.append(indx)
            df_fix.loc[df_fix["original_index"] == indx, "anomalous"] = 1
        else:
            for compiled_pattern in compiled_patterns:
                if compiled_pattern.search(req_str):
                    wrong_fp_indexes.append(indx)
                    df_fix.loc[df_fix["original_index"] == indx, "anomalous"] = 1
                    break

    df_misslabeled = df[df["original_index"].isin(wrong_fp_indexes)]
    df_misslabeled = df_misslabeled["original_index"]
    df_misslabeled.to_csv(output_mislabeled)

    print(f"Tot wrong: {len(wrong_fp_indexes)} over {len(df)} FP: {len(wrong_fp_indexes) / len(df)}")
    df_fix.to_csv(output_fix)
