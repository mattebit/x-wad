import os

import pandas
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModel,
)

from common.eval import get_best_result, get_fn_indexes, get_fp_indexes
from common.utils import (
    model_load_settings_normal,
    model_load_settings_optimized,
    MODELS_FOLDER_PATH,
    DATASETS_FOLDER_PATH,
)
from common.predict import explain_causal, explain_masked

is_mlm = False

SRBH_DATASET_PATH = os.path.join(DATASETS_FOLDER_PATH, "srbh", "srbh.requests")
SRBH_FIX_DATASET_PATH = os.path.join(DATASETS_FOLDER_PATH, "srbh", "srbhfix.requests")

explain_function = {True: explain_masked, False: explain_causal}

model_modernbert_large_srbh = {
    "model": os.path.join(MODELS_FOLDER_PATH, "ModernBERT-large-f-srbh"),
    "is_mlm": True,
    "allows_optimizations": False,
    "dataset_path": SRBH_DATASET_PATH,
    "eval_file": os.path.join(
        MODELS_FOLDER_PATH,
        "ModernBERT-large-f-srbh",
        "evaluations/eval_standard-mlm-modernebert-test-capec.csv",
    ),
}

model_modernbert_large_srbh_fix = {
    "model": os.path.join(MODELS_FOLDER_PATH, "ModernBERT-large-s-srbh-fix"),
    "is_mlm": True,
    "allows_optimizations": True,
    "dataset_path": SRBH_FIX_DATASET_PATH,
    "eval_file": os.path.join(
        MODELS_FOLDER_PATH,
        "ModernBERT-large-s-srbh-fix/evaluations/eval_fix-modernbert-test-srbhfix.csv",
    ),
}

model_smol_srbh = {
    "model": os.path.join(MODELS_FOLDER_PATH, "SmolLM2-360M-f-srbh"),
    "is_mlm": False,
    "allows_optimizations": False,
    "dataset_path": SRBH_DATASET_PATH,
    "eval_file": os.path.join(
        MODELS_FOLDER_PATH,
        "SmolLM2-360M-f-srbh/evaluations/eval_standard-test-capec.csv",
    ),
}

model_smol_srbh_fix = {
    "model": os.path.join(MODELS_FOLDER_PATH, "SmolLM2-360M-s-srbh-fix"),
    "is_mlm": False,
    "allows_optimizations": False,
    "dataset_path": SRBH_FIX_DATASET_PATH,
    "eval_file": os.path.join(
        MODELS_FOLDER_PATH,
        "SmolLM2-360M-s-srbh-fix/evaluations/eval_fix-smol-test-srbhfix.csv",
    ),
}

sample_srbh_8590 = "GET /%2Fetc%2Fpasswd/index.php HTTP/1.1[r][n]Host: test-site.com[r][n]User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:73.0) Gecko/20100101 Firefox/73.0[r][n]Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8[r][n]Accept-Language: en-US,en;q=0.5[r][n]Connection: keep-alive[r][n]"

sample_srbh_27884 = "GET /blog/index.php/wp-json/oembed/1.0/embed?format=xml&url=http%3A%2F%2Ftest-site.com%2Fblog%2Findex.php%2Fvitae-dolores-quidem-possimus-aut-voluptatibus%2F HTTP/1.1[r][n]Host: test-site.com[r][n]User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0[r][n]Referer: ../../../../../../../../../../../../../../../../etc/passwd[r][n]Cookie: wordpress_test_cookie=WP+Cookie+check; wordpress_logged_in_1aefbe2f76edd740f8e362f39da3353b=rafael%7C1595188528%7ClHi7YVFypQtuJVAqG6B4FOpzZ24r5fr11PP0w4JBwKX%7C05eff642e7638456de705bca89de40119bc343cb09e3b0c57372fe30717543b0[r][n]"


def load_model_and_tokenizer(model_to_use: dict):
    """
    Load model and tokenizer given the model config in the form:
    model_to_use = {
        "model": path, the path of the model
        "is_mlm": True, # If it is masked
        "allows_optimizations": True, # true enable optimizations such as flash attention
        "dataset_path": "", # the path of the dataset (optional)
        "eval_file": "" the evaluation file (optional)
    }
    :param model_to_use: a dict in the format above
    :return: model, tokenizer
    """

    args = {"pretrained_model_name_or_path": model_to_use["model"]}
    if model_to_use["allows_optimizations"] == True:
        args |= model_load_settings_optimized
    else:
        args |= model_load_settings_normal

    model = (
        AutoModelForMaskedLM.from_pretrained(**args).to("cuda")
        if model_to_use["is_mlm"]
        else AutoModelForCausalLM.from_pretrained(**args).to("cuda")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_to_use["model"])

    return model, tokenizer


if __name__ == "__main__":
    model_to_use = model_modernbert_large_srbh_fix # Select which model to use
    model, tokenizer = load_model_and_tokenizer(model_to_use)

    if True:
        # Explain single sample
        log_string, avg_prob = explain_function[model_to_use["is_mlm"]](
            sample_srbh_8590,
            model,
            tokenizer,
            suppress_print=True,
            print_output="terminal",
            gradient=True,
        )
        print(f"With gradient {avg_prob}\n{log_string}")

    if False:
        # Explain entire dataset
        dataset_path = model_to_use["dataset_path"]
        df = pandas.read_csv(dataset_path)
        df = df.sample(frac=1).reset_index(drop=True)

        eval_file = model_to_use["eval_file"]
        fp_indexes = get_fp_indexes(get_best_result(eval_file, by="f1fn"))
        fn_indexes = get_fn_indexes(get_best_result(eval_file, by="f1fn"))

        for i, row in df.iterrows():
            original_index = row["original_index"]
            if row["anomalous"] == 1:
                # if original_index in fn_indexes:
                if model_to_use["is_mlm"]:
                    explain_masked(
                        row["request"],
                        model,
                        tokenizer,
                        batch_size=60,
                        print_output="terminal",
                        # print_more_than=1
                    )
                else:
                    l, p = explain_causal(
                        row["request"],
                        model,
                        tokenizer,
                        print_output="terminal",
                    )
