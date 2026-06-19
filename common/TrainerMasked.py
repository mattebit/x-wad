"""
Original Copyright (c) 2022 Kathrin Seßler and Vadim Borisov. Licensed under the MIT License.
Part of code is adapted from the GReaT repository (https://github.com/kathrinse/be_great/tree/main)
Modifications Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Further modifications Copyright 2026 Matteo Bitussi
"""

from typing import Optional, Union

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer

from common.predict import get_loss_masked_batch


class TrainerMasked(Trainer):
    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        data_collator = self.data_collator

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        # Use the modified eval dataloader to not remove the label column
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        self.model.eval()

        all_labels = []
        all_loss = []
        all_perplexity = []

        print("Doing evaluation...")

        # evaluation loop
        for step, inputs in enumerate(eval_dataloader):
            all_labels.extend(
                inputs.pop("anomalous")
            )  # Take the label and remove it before inference

            inputs_clear = {
                "input_ids": inputs["input_ids"],
                "labels": inputs["labels"],
                "attention_mask": inputs["attention_mask"],
            }

            inputs = self._prepare_inputs(inputs_clear)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

                per_sample_loss, per_sample_perplexity = get_loss_masked_batch(
                    inputs, logits
                )

                all_loss.extend(per_sample_loss)
                all_perplexity.extend(per_sample_perplexity)

        normal_indices = np.array([i for i, m in enumerate(all_labels) if m == 0])
        anomaly_indices = np.array([i for i, m in enumerate(all_labels) if m == 1])

        anomalies_present = False
        if anomaly_indices.size > 0:
            anomalies_present = True

        all_perplexity = np.array(all_perplexity)
        perplexity_normal = np.mean(all_perplexity[normal_indices]).item()

        if anomalies_present:
            perplexity_anomaly = np.mean(all_perplexity[anomaly_indices]).item()
        else:
            perplexity_anomaly = 0

        auc_roc = metrics.roc_auc_score(all_labels, all_perplexity)
        del all_perplexity

        all_loss = np.array(all_loss)
        eval_loss_normal = np.mean(all_loss[normal_indices]).item()
        if anomalies_present:
            eval_loss_anomaly = np.mean(all_loss[anomaly_indices]).item()
        else:
            eval_loss_anomaly = 0
        avg_loss = all_loss.mean().item()
        del all_loss

        return_metrics = {
            "loss": avg_loss,
            "auc_roc": auc_roc,
            "perplexity_normal": perplexity_normal,
            "perplexity_anomalous": perplexity_anomaly,
            "loss_normal": eval_loss_normal,
            "loss_anomalous": eval_loss_anomaly,
        }

        # Add the prefix (e.g., "eval_") to all keys
        return_metrics = {
            f"{metric_key_prefix}_{k}": v for k, v in return_metrics.items()
        }

        # Log the metrics
        self.log(return_metrics)
        self._memory_tracker.stop_and_update_metrics(return_metrics)
        self.callback_handler.on_evaluate(
            self.args, self.state, self.control, return_metrics
        )

        return return_metrics
