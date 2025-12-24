from pathlib import Path
from typing import Tuple, Dict, Any
import json
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import evaluate


def load_hf_disk_dataset(path: str | Path) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path not found: {p}")
    ds = load_from_disk(str(p))

    # load_from_disk may return Dataset or DatasetDict. Normalize to Dataset.
    if hasattr(ds, "keys") and "train" in ds:
        return ds["train"]
    return ds


def split_train_val_test(
    ds: Dataset,
    label_field: str,
    validation_split: float,
    test_split: float,
    seed: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a HF Dataset into train/val/test with stratification.

    Example: validation_split=0.1, test_split=0.1 -> train=0.8, val=0.1, test=0.1
    """
    if validation_split < 0 or test_split < 0:
        raise ValueError("validation_split and test_split must be >= 0.")
    if validation_split + test_split >= 1.0:
        raise ValueError("validation_split + test_split must be < 1.0.")

    # 1) Split off temp = val + test
    temp_size = validation_split + test_split
    split1 = ds.train_test_split(
        test_size=temp_size,
        seed=seed,
        stratify_by_column=label_field,
    )
    train_ds = split1["train"]
    temp_ds = split1["test"]

    # 2) Split temp into val and test
    # fraction of temp that should become test:
    # test / (val + test)
    if temp_size == 0:
        raise ValueError("validation_split + test_split is 0; nothing to split.")
    test_ratio_within_temp = test_split / temp_size

    split2 = temp_ds.train_test_split(
        test_size=test_ratio_within_temp,
        seed=seed,
        stratify_by_column=label_field,
    )
    val_ds = split2["train"]
    test_ds = split2["test"]

    return train_ds, val_ds, test_ds


def build_tokenizer_and_collator(pretrained_name: str):
    tok = AutoTokenizer.from_pretrained(pretrained_name, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer=tok)
    return tok, collator


def tokenize_dataset(
    ds: Dataset,
    tokenizer,
    text_field: str,
    label_field: str,
    max_length: int,
) -> Dataset:
    if text_field not in ds.column_names:
        raise KeyError(f"'{text_field}' not in dataset columns: {ds.column_names}")
    if label_field not in ds.column_names:
        raise KeyError(f"'{label_field}' not in dataset columns: {ds.column_names}")

    def _prep(batch):
        return tokenizer(batch[text_field], truncation=True, max_length=max_length)

    keep_cols = {label_field}
    drop_cols = [c for c in ds.column_names if c not in keep_cols]

    out = ds.map(_prep, batched=True, remove_columns=drop_cols)

    # Trainer expects "labels"
    if label_field != "labels":
        out = out.rename_column(label_field, "labels")
    return out


def infer_num_labels(ds: Dataset) -> int:
    labels = ds["labels"] if "labels" in ds.column_names else ds["label"]
    return len(set(labels))


def build_metrics():
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(
                predictions=preds, references=labels, average="macro"
            )["f1"],
        }

    return compute_metrics



def save_train_val_eval_splits(
    train_ds: Dataset,
    val_ds: Dataset,
    eval_ds: Dataset,
    output_dir: str | Path,
    train_name: str = "train",
    val_name: str = "val",
    eval_name: str = "test",
) -> None:
    """
    Save Hugging Face train/eval Dataset splits to disk for reproducibility
    and future debugging.

    They can later be reloaded with `datasets.load_from_disk`.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / train_name
    eval_path = out / eval_name

    train_ds.save_to_disk(str(train_path))
    val_ds.save_to_disk(str(out / val_name))
    eval_ds.save_to_disk(str(eval_path))

    print(f"Saved train split to: {train_path}")
    print(f"Saved eval split to:  {eval_path}")
    print(f"Saved val split to:   {out / val_name}")


def save_performance_metrics(
    output_dir: str | Path,
    metrics: Dict[str, Any],
    filename: str = "metrics.json",
):
    """
    Save performance metrics to a JSON file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = out / filename
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved performance metrics to: {metrics_path}")
