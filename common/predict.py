import os
import shutil
from time import perf_counter

import numpy as np
import pandas
import seaborn as sns
import torch
from datasets import Dataset, load_from_disk
from matplotlib import pyplot as plt
from scipy.stats import norm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM

from common.utils import (
    tokenizer_base_args_predict,
    tokenizer_base_args,
    get_cpu_count,
    model_load_settings_normal,
    IMAGES_FOLDER_PATH,
)


def get_loss_masked_batch(inputs, logits, return_not_sum=False):
    """
    From batched inputs and resulting logits, get per-sample loss and perplexities
    :param inputs: the input inputs
    :param logits: the model output logits from the given inputs
    :param return_not_sum: Return the per-token loss of each sample if set to true
    :return: per-sample loss and perplexity (and per-token loss and perplexity depending on return_not_sum)
    """
    # Labels should have shape (batch_size, seq_len)
    # Non-masked tokens are -100, masked tokens have the true token ID
    labels = inputs.get("labels")
    if labels is None:
        raise ValueError("Inputs must contain 'labels' to calculate MLM loss.")

    batch_size, seq_len, vocab_size = logits.shape

    # Reduction none keeps all values for each sample
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    # Flatten logits and labels to (batch_size * seq_len, ...)
    # Loss function expects (N, C) logits and (N) labels
    flat_logits = logits.view(-1, vocab_size)
    flat_labels = labels.view(-1)

    # Shape: (batch_size * seq_len)
    # Tokens with label -100 will have a loss of 0.0
    per_token_loss = loss_fct(flat_logits, flat_labels)

    # Reshape loss back to (batch_size, seq_len)
    per_token_loss = per_token_loss.view(batch_size, seq_len)

    # Sum the loss for each sample (across the seq_len dimension)
    # This gives the total loss for all masked tokens in each sample
    total_loss_per_sample = per_token_loss.sum(dim=1)  # Shape: (batch_size)

    # Count the number of masked tokens in each sample
    # We count where labels are NOT -100
    num_masked_tokens_per_sample = (labels != -100).sum(dim=1)  # Shape: (batch_size)

    # Calculate the per-sample loss
    # Handle division by zero if a sample has no masked tokens
    avg_loss_per_sample = torch.where(
        num_masked_tokens_per_sample > 0,
        total_loss_per_sample / num_masked_tokens_per_sample,
        torch.tensor(0.0, device=logits.device),  # Use 0.0 if no masked tokens
    )

    # Calculate per-sample perplexity
    perplexity_per_sample = torch.exp(avg_loss_per_sample)

    if return_not_sum:
        return (
            avg_loss_per_sample.to("cpu"),
            perplexity_per_sample.to("cpu"),
            per_token_loss.to("cpu"),
            torch.exp(per_token_loss).to("cpu"),
        )

    return avg_loss_per_sample.to("cpu"), perplexity_per_sample.to("cpu")


def prepare_explained_string(
    token_probabilities,
    tokens,
    print_output="terminal",
    gradient=True,
    contrast_factor=1,
) -> str:
    """Function used to prepare the explained strings in the format specified with the "print_output" parameter.


    Args:
        token_probabilities (_type_):  The list of token probabilities
        tokens (_type_): The tokens list in string format
        print_output (str, optional): The output format of the explained string (terminal, html, latex, latex_text). Defaults to "terminal".
        gradient (bool, optional): If set to true uses a gradient coloring, otherwise it uses the percentile-based coloring. Defaults to True.
        contrast_factor (int, optional): A factor used to isolate rare events during coloring, set it <0. Defaults to 1 means no contrast applied.

    Returns:
        str: the explained string
    """
    log_string = ""

    if print_output == "html":
        log_string = '<div style="background-color: #cccccc; color: white; padding: 20px; font-family: monospace; line-height: 1.5;">'

    if not gradient:
        thresholds = np.percentile(token_probabilities, [2, 6, 14])

        color_mapped_thresholds = [
            (thresholds[0], "FF0000"),  # Red
            (thresholds[1], "FFA500"),  # Orange
            (thresholds[2], "FFFF00"),  # Yellow
        ]

    def get_terminal_color(hex_str, style="foreground"):
        """
        Converts a HEX string (e.g., 'FF0000' or '#FF0000') into a 24-bit ANSI escape code.
        style: "background" to color the box behind text, "foreground" to color the text itself.
        """
        hex_str = hex_str.lstrip("#")
        # Parse the HEX string into RGB integer values
        r, g, b = tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))

        if style == "background":
            return f"\033[30m\033[48;2;{r};{g};{b}m"
        else:
            return f"\033[38;2;{r};{g};{b}m"

    for indx, token_id in enumerate(token_probabilities):
        if tokens[indx] in [
            "<|endoftext|>",
            "[EOS]",
            "</s>",
            "<EOS>",
            "[PAD]",
            "<PAD>",
            "[CLS]",
        ]:
            # Skip if special token
            # TODO: can be made more efficient
            continue

        actual_token_probability = float(token_id)

        def squeeze(x, factor):
            return x**factor

        actual_token_probability = squeeze(actual_token_probability, contrast_factor)

        if gradient:
            anomaly_hex_value = f"{round((1 - actual_token_probability) * 255):02x}"
            normal_hex_value = f"{round((actual_token_probability) * 255):02x}"

        if print_output == "terminal":
            RESET = "\033[0m"

            if gradient:
                chosen_color = f"#ff{normal_hex_value}ff"
            else:
                chosen_color = None
                for limit, hex_code in color_mapped_thresholds:
                    if actual_token_probability < limit:
                        chosen_color = hex_code
                        break

            if chosen_color:
                color_code = get_terminal_color(chosen_color)
                log_string += f"{color_code}{tokens[indx]}{RESET}"

        elif print_output == "html":
            token_text = tokens[indx].replace(" ", "&nbsp;")  # Preserve spaces in HTML
            if gradient:
                color = f"#{anomaly_hex_value}00{normal_hex_value}"
                log_string += f'<span style="color: {color};">{token_text}</span>'
            else:
                colors = {
                    "low": "#ff6b6b",  # Soft Red
                    "mid": "#ff8000",  # Warm Yellow
                    "high": "#ffff00",  # White
                    "default": "#000000",  # black
                }

                if actual_token_probability < thresholds[0]:
                    color = colors["low"]
                elif actual_token_probability < thresholds[1]:
                    color = colors["mid"]
                elif actual_token_probability < thresholds[2]:
                    color = colors["high"]
                else:
                    color = colors["default"]

                log_string += f'<span style="color: {color};">{token_text}</span>'

        elif print_output == "latex":
            act_token = tokens[indx]

            chars_to_escape = ["_", "$", "%", "&"]

            for c in chars_to_escape:
                if c in act_token:
                    act_token = act_token.replace(c, f"\\{c}")

            if gradient:
                chosen_color = f"ff{normal_hex_value}ff"
            else:
                chosen_color = None
                for limit, hex_code in color_mapped_thresholds:
                    if actual_token_probability < limit:
                        chosen_color = hex_code
                        break

            if chosen_color:
                log_string += f"|\\hlhex{{{chosen_color}}}{{{act_token}}}|"
            else:
                log_string += f"{act_token}"

    if print_output == "terminal":
        log_string += "\033[0m"
    elif print_output == "html":
        log_string += "</div>"

    log_string = log_string.replace("Ġ", " ")  # Replace special character with space
    return log_string


def explain_masked(
    in_string: str,
    model,
    tokenizer,
    print_output="terminal",
    batch_size=20,
    suppress_print=False,
    tokenizer_args=tokenizer_base_args_predict,
    gradient=True,
) -> tuple[str, float]:
    """
    Make a prediction and explain it visually by printing on terminal.

    :param in_string: the string to make the prediction of and to explain
    :param model: the model used to make the prediction
    :param tokenizer: the tokenizer associated with that model
    :param print_output: the output of the print "terminal" or "latex" format
    :param batch_size: the batch_size to use to process the input
    :param suppress_print: Set to true to avoid this function to print to terminal the explanation
    :param tokenizer_args: the tokenizer arguments to be used

    :return a tuple containing the explained string and the loss value
    """
    encodings = tokenizer(in_string, **tokenizer_args).to("cuda")

    tokens = tokenizer.convert_ids_to_tokens(encodings["input_ids"][0])
    seq_len = len(encodings["input_ids"][0])

    input_ids = []
    attention_mask = []
    labels = []

    # TODO: add option to use strided masking (all token masked == stride=1)
    # All the tokens are masked one time iteratively for most accurate result
    for i in range(seq_len):
        current_token_id = encodings["input_ids"][0][i]
        current_attention_mask_value = encodings["attention_mask"][0][i]

        if current_attention_mask_value != 1:
            # it is a padding or something to not consider
            continue

        if (
            current_token_id == tokenizer.cls_token_id
            or current_token_id == tokenizer.sep_token_id
        ):
            # It is the first or last token, skip
            continue

        input_ids.append(encodings["input_ids"][0])
        attention_mask.append(encodings["attention_mask"][0])
        label = torch.full(size=(seq_len,), fill_value=-100)
        label[i] = current_token_id
        labels.append(label)

    dataset = TensorDataset(
        torch.stack(input_ids), torch.stack(attention_mask), torch.stack(labels)
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    single_tokens_loss = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        # Unpack and move to device
        b_input_ids, b_attn_mask, b_labels = [t.to("cuda") for t in batch]
        b_inputs = {
            "input_ids": b_input_ids,
            "attention_mask": b_attn_mask,
            "labels": b_labels,
        }

        with torch.no_grad():
            outputs = model(**b_inputs)

            # Calculate loss per token for the batch
            _, _, per_sample_loss, _ = get_loss_masked_batch(
                b_inputs, outputs.logits, return_not_sum=True
            )

            # 3. Vectorized index extraction
            mask = b_labels != -100
            # Find index of the masked token for each item in the batch
            indices = (
                mask.float().argmax(dim=1, keepdim=True).to(per_sample_loss.device)
            )

            # Gather the specific losses
            batch_losses = per_sample_loss.gather(1, indices).squeeze(1)
            single_tokens_loss.extend(batch_losses.cpu().tolist())

    # group results
    final_losses = np.array(single_tokens_loss)
    token_probabilities = np.exp(-final_losses).tolist()

    avg_prob = np.array(token_probabilities).mean()

    log_string = prepare_explained_string(
        token_probabilities, tokens, print_output, gradient=gradient
    )

    if not suppress_print:
        print(log_string)
    return log_string, avg_prob


def get_batch_predictions_masked_strided(
    dataset_path: str,
    model_path: str,
    output_path: str,
    batch_size: int = 10,
    stride: int = 5,
    log_times: bool = False,
    log_times_output_filename="times.csv",
    model_load_settings=model_load_settings_normal,
):
    """
    This function does the prediction of the given dataset, with the given model, and saves the anomaly scores in the
    given output path. It uses a stride masking technique.
    :param dataset_path: the path of the dataset to be predicted
    :param model_path: the path of the folder containing the model to be used to do the prediction (should support MLM)
    :param output_path: the path of where to save the output
    :param batch_size: the batch size to use
    :param stride: the stride value to use in stride masking
    :param log_times: Set to true to log time taken to predict
    :param log_times_output_filename: The filename where to save the logged times
    :param model_load_settings: Optional settings to use to load the model
    :return: the anomaly scores as dict
    """
    model = AutoModelForMaskedLM.from_pretrained(model_path, **model_load_settings).to(
        "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    df = pandas.read_csv(dataset_path)

    # create many copy of the samples that are the strides
    df_list = []
    for s in range(stride):
        temp_df = df.copy()
        temp_df["shift_index"] = s  # the stride offset
        df_list.append(temp_df)

    # move from a pandas dataset to a temporary hf dataset to improve ram usage
    df_expanded = pandas.concat(df_list, ignore_index=True)
    dataset = Dataset.from_pandas(df_expanded)
    dataset.save_to_disk("./tmp_dataset")
    del df_expanded
    del dataset
    dataset = load_from_disk("./tmp_dataset")

    def tokenize_and_stride_mask(examples):
        inputs = tokenizer(
            examples["request"],
            **tokenizer_base_args_predict,
            return_special_tokens_mask=True,  # needed for later
        )
        input_ids = np.array(inputs["input_ids"])
        labels = input_ids.copy()

        # Get special tokens mask (1 for special tokens, 0 for regular)
        # tokenizer returns special_token_mask
        special_tokens_mask = np.array(inputs.get("special_tokens_mask", []))

        # If special_tokens_mask wasn't generated, create it manually
        if len(special_tokens_mask) == 0:
            special_tokens_mask = np.isin(input_ids, tokenizer.all_special_ids)

        shifts = np.array(examples["shift_index"])

        for i in range(len(input_ids)):
            # Create a mask for the current stride positions: True where (idx % stride == shift)
            seq_len = len(input_ids[i])
            indices = np.arange(seq_len)
            stride_mask = indices % stride == shifts[i]

            # Mask input_ids where stride_mask is True AND it's not a special token
            mask_condition = stride_mask & ~special_tokens_mask[i].astype(bool)
            input_ids[i][mask_condition] = tokenizer.mask_token_id

            # Set labels to -100 everywhere EXCEPT where we actually masked for later be ignored by loss calculation
            labels[i][~mask_condition] = -100

        return {
            "input_ids": input_ids.tolist(),
            "labels": labels.tolist(),
            "attention_mask": inputs["attention_mask"].tolist(),
        }

    tokenized_dataset = dataset.map(
        tokenize_and_stride_mask,
        batched=True,
        remove_columns=["request"],
        num_proc=get_cpu_count(),
    )
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "original_index"],
    )
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)

    try:
        shutil.rmtree("./tmp_dataset")
    except FileNotFoundError:
        print("Directory not found ./tmp_dataset")
    except PermissionError:
        print("Permission denied ./tmp_dataset")
    except Exception as e:
        print(f"Error occurred: {e}")

    results_accumulator = []
    times = []

    for batch in tqdm(dataloader, desc="Predicting.."):
        if log_times:
            start_t = perf_counter()

        indices = batch.pop("original_index")
        inputs = {k: v.to("cuda") for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            per_sample_loss, per_sample_perplexity = get_loss_masked_batch(
                inputs, logits
            )

            if log_times:
                t_taken = perf_counter() - start_t

            # for every sample in batch
            for i in range(len(indices)):
                results_accumulator.append(
                    {
                        "original_index": indices[i].item(),
                        "loss": per_sample_loss[i].item(),
                        "perplexity": per_sample_perplexity[i].item(),
                    }
                )
                if log_times:
                    times.append(
                        {
                            "original_index": indices[i].item(),
                            "time_taken": t_taken / len(indices),
                            "tokens_count": (inputs["attention_mask"][i] == 1)
                            .sum()
                            .item(),
                        }
                    )

    if log_times:
        df_times = pandas.DataFrame(times)
        df_times = df_times.groupby("original_index", as_index=False).sum()
        df_times.to_csv(log_times_output_filename, index=False)

    # calculate average results based on original_index
    df_res = pandas.DataFrame(results_accumulator)
    final_df = df_res.groupby("original_index", as_index=False).mean()
    final_df.to_csv(output_path, index=False)

    return final_df.to_dict(orient="records")


def get_loss_perplexity_causal_batch(
    inputs, logits, pad_token_id, return_not_sum=False
) -> tuple:
    """
    Get the loss and perplexity values from a causal model prediction output. This function works with batches, and
    returns a tuple, with a list of the losses and perplexity, one for each batch sample.
    :param inputs: the tokenized inputs of current batch
    :param logits: the logits obtained from the prediction output of the model given the inputs
    :param pad_token_id: the padding token id used by this model (it will be ignored in the CrossEntropyLoss)
    :param return_not_sum: Return the per-token loss instead of per-sample
    :return: a tuple (per_sample_loss, per_sample_perplexity)
    """
    loss_fct = nn.CrossEntropyLoss(
        reduction="none", ignore_index=-100
    )  # Define loss function

    labels = inputs["input_ids"].clone()

    # Ignore padding tokens
    labels[labels == pad_token_id] = -100

    # Shift logits and labels for Causal LM
    shift_logits = logits[..., :-1, :].contiguous()  # take all logits except last one
    shift_labels = labels[..., 1:].contiguous()  # tale all labels from the second one

    # Flatten the tensors to compute the loss
    batch_size, seq_len = shift_logits.shape[:2]
    vocab_size = shift_logits.shape[2]

    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)

    # Get the per-token loss
    per_token_loss = loss_fct(flat_logits, flat_labels)

    # Reshape the per-token loss to (batch_size, seq_len - 1)
    per_token_loss = per_token_loss.view(batch_size, seq_len)

    # Sum the losses of every token
    per_sample_loss_sum = per_token_loss.sum(dim=1)

    # Count the number of valid tokens in each sample
    valid_mask = (shift_labels != -100).float()
    valid_token_count = valid_mask.sum(dim=1)

    per_sample_loss_all = per_token_loss * valid_mask  # TODO: not needed?

    # Add a small epsilon to avoid division by zero
    per_sample_loss = per_sample_loss_sum / (valid_token_count + 1e-9)

    # Calculate Perplexity
    per_sample_perplexity = torch.exp(per_sample_loss)
    per_sample_perplexity_all = torch.exp(per_sample_loss_all)

    if return_not_sum:
        # This code manages the missing first token to align correctly the output length for explainability
        # Create a column of zeros with shape (batch_size, 1) to match the batch dimension
        zero_padding = torch.zeros((batch_size, 1), device=per_token_loss.device)
        one_padding = torch.ones((batch_size, 1), device=per_token_loss.device)

        # Prepend to the sequence dimension (dim=1)
        per_sample_loss_all = torch.cat(
            [zero_padding, per_token_loss * valid_mask], dim=1
        )
        per_sample_perplexity_all = torch.cat(
            [one_padding, torch.exp(per_token_loss * valid_mask)], dim=1
        )

        return (
            per_sample_loss,
            per_sample_perplexity,
            per_sample_loss_all,
            per_sample_perplexity_all,
        )

    # Note that per_sample_loss and perplexity are not influenced by extra loss values added for padding
    return per_sample_loss, per_sample_perplexity


def get_batch_predictions_causal(
    dataset_path: str,
    model_path: str,
    output_path: str,
    batch_size: int = 20,
    log_times: bool = False,
    log_times_output_filename="times.csv",
    model_load_settings=model_load_settings_normal,
):
    """
    Get batched predictions of the given dataset and model and save them to output_path.
    :param dataset_path: The path to the dataset to get the predictions from
    :param model_path: The path of the model folder
    :param output_path: The output file path
    :param batch_size: the batch size to be used
    :param log_times: whether to log time taken by each prediction step
    :param log_times_output_filename: the output filename for the time logs
    :param model_load_settings: (optional) specify custom model load settings
    """

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_load_settings).to(
        "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    df = pandas.read_csv(dataset_path)

    results = []
    times = []

    model.eval()  # Set the model in evaluation mode to have consistent results

    # Process the DataFrame rows in batches
    for row_index in tqdm(range(0, len(df), batch_size), desc="Running Predictions"):
        if log_times:
            start_t = perf_counter()

        batch = df.iloc[row_index : row_index + batch_size]

        # Get data for the batch
        texts = batch["request"].tolist()
        indices = batch["original_index"].tolist()

        # Tokenize the batch of texts
        inputs = tokenizer(texts, **tokenizer_base_args)

        # Move tokenized inputs to the GPU
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
            logits = outputs.logits

        pad_token_id = model.config.pad_token_id

        # Handle models like GPT-2 that might not have a pad_token_id
        if pad_token_id is None:
            pad_token_id = model.config.eos_token_id

        # calculate loss and perplexity for each sample in current batch
        per_sample_loss, per_sample_perplexity, all_loss, all_perplexity = (
            get_loss_perplexity_causal_batch(
                inputs, logits, pad_token_id, return_not_sum=True
            )
        )

        if log_times:
            t_taken = perf_counter() - start_t

        # append results
        for index, sample_logit in enumerate(logits):
            results.append(
                {
                    "original_index": indices[index],
                    "perplexity": per_sample_perplexity[index].item(),
                    "loss": per_sample_loss[index].item(),
                }
            )

            if log_times:
                times.append(
                    {
                        "original_index": indices[index],
                        "time_taken": t_taken / len(logits),
                        "tokens_count": (inputs["attention_mask"][index] == 1)
                        .sum()
                        .item(),
                    }
                )

    if log_times:
        df_times = pandas.DataFrame(times)
        df_times.to_csv(log_times_output_filename, index=False)

    df_out = pandas.DataFrame(data=results)
    df_out.to_csv(output_path, index=False)

    return results


def explain_causal(
    in_string: str,
    model,
    tokenizer,
    print_output="terminal",
    suppress_print=False,
    gradient=True,
    tokenizer_args=tokenizer_base_args_predict,
) -> tuple[str, float]:
    """
    Make a prediction and explain it visually by printing on terminal.

    :param in_string: the string to make the prediction of and to explain
    :param model: the model used to make the prediction
    :param tokenizer: the tokenizer associated with that model
    :param print_output: the output of the print "terminal" or "latex" format
    :param suppress_print: Set to true to avoid this function to print to terminal the explanation
    :param tokenizer_args: the tokenizer arguments to be used
    :param gradient: Set to true to print an highlight which is based to a gradient, otherwise will be percentile

    :return a tuple containing the explained string and the loss value
    """
    inputs = tokenizer(in_string, **tokenizer_args).to("cuda")

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
        logits = outputs.logits

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    pad_token_id = model.config.pad_token_id

    # Handle models that might not have a pad_token_id
    if pad_token_id is None:
        pad_token_id = model.config.eos_token_id

    avg_loss, avg_perplexity, all_loss, all_perplexity = (
        get_loss_perplexity_causal_batch(
            inputs, logits, pad_token_id, return_not_sum=True
        )
    )

    token_probabilities = np.exp(-all_loss.to("cpu"))[0]
    avg_prob = np.array(token_probabilities).mean()
    log_string = prepare_explained_string(
        token_probabilities, tokens, print_output, gradient=gradient
    )

    if not suppress_print:
        print(log_string)
    return log_string, avg_prob


def estimate_threshold_from_predictions(
    predict_file_path: str,
    method="gaussian",
    percentile=99,
    k=0.3,
    plot=False,
    eval_metric="loss",
) -> float:
    """
    This function estimates the anomaly detection theshold from the predictions contained in a file from one of the
    predict functions. There are two estimation methods available, gaussian and percentile.
    :param predict_file_path: The path of the file containing the predictions. (only normal samples predicted expected)
    :param method: The method to use (gaussian or percentile)
    :param percentile: the percentile value to use as a threshold
    :param k: The k value for the gaussian estimation
    :param plot: Set to true to plot the probability density function.
    :param eval_metric: The metric from the prediction file to be used, default loss (explainability available)
    :return: the estimated threshold
    """
    df = pandas.read_csv(predict_file_path)

    if method == "gaussian":
        mu, std = norm.fit(df[eval_metric].tolist())
        thres = mu + k * std
        print(f"Estimated threshold gaussian: {thres}")

        if plot:
            plt.figure(figsize=(7, 5))
            # density=True to match the scale of the Gaussian curve
            plt.hist(df[eval_metric].tolist(), bins=1500, density=True, alpha=0.6)

            # Create the Gaussian Curve
            xmin, xmax = plt.xlim()  # Get the range of the x-axis
            x = np.linspace(xmin, xmax, 1000)  # Generate 100 smooth x-points

            p = norm.pdf(x, mu, std)  # Calculate y-values for the Gaussian

            plt.plot(
                x,
                p,
                "k",
                linewidth=2,
                label=rf"Fit results: $\mu$={mu:.4f}, $\sigma$={std:.4f}",
            )

            plt.xlabel("Loss")
            plt.ylabel("Probability density")
            plt.xlim([-0.01, 0.2])
            plt.legend()
            plt.savefig(os.path.join(IMAGES_FOLDER_PATH, "threshold_est.pdf"))
            plt.show()

    elif method == "percentile":
        thres = np.percentile(df[eval_metric].tolist(), percentile)
        print(f"Estimated threshold percentile: {thres}")

        if plot:
            percentiles = [25, 50, 75, 95, 99]
            percentile_values = {
                p: np.percentile(df[eval_metric].tolist(), p) for p in percentiles
            }

            plt.figure(figsize=(7, 5))

            # Histogram with KDE (Kernel Density Estimate) curve
            sns.histplot(df[eval_metric].tolist(), bins=100, kde=True, alpha=0.6)

            # add vertical lines for each percentile
            colors = {25: "green", 50: "blue", 75: "green", 95: "orange", 99: "red"}
            styles = {25: "--", 50: "-", 75: "--", 95: "-.", 99: ":"}

            for p in percentiles:
                val = percentile_values[p]
                plt.axvline(
                    x=val,
                    color=colors[p],
                    linestyle=styles[p],
                    linewidth=2,
                    label=f"{p}th Percentile: {val:.1f}",
                )

            plt.xlabel("Loss")
            # plt.xlim([-0.01, 1])
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()
            print(os.getcwd())
            plt.savefig(os.path.join(IMAGES_FOLDER_PATH, "threshold_est.pdf"))
            plt.show()

    return thres
