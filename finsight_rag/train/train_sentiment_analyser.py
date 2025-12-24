import os
from importlib import resources as impresources

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

import finsight_rag.config as config
from finsight_rag.utils import load_yaml  # your existing YAML loader

# New helper imports (from utils.py)
from finsight_rag.train.utils import (
    load_hf_disk_dataset,
    split_train_val_test,
    save_train_val_eval_splits,
    build_tokenizer_and_collator,
    tokenize_dataset,
    infer_num_labels,
    build_metrics,
    save_performance_metrics
)


def main(cfg: dict):
    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]
    tr_cfg = cfg["training"]
    rt_cfg = cfg.get("runtime", {})

    train_raw = load_hf_disk_dataset(ds_cfg["path"])
    train_raw, val_raw, test_raw = split_train_val_test(
        train_raw,
        label_field=ds_cfg["label_field"],
        validation_split=float(ds_cfg["validation_split"]),
        test_split=float(ds_cfg["test_split"]),
        seed=int(ds_cfg["seed"]),
    )
    
    # ---- DEBUG: use small subset ----
    # train_raw = train_raw.select(range(min(30, len(train_raw))))
    # test_raw = test_raw.select(range(min(30, len(test_raw))))
    # val_raw = val_raw.select(range(min(30, len(val_raw))))

    tokenizer, collator = build_tokenizer_and_collator(m_cfg["pretrained_name"])

    train_ds = tokenize_dataset(
        train_raw,
        tokenizer=tokenizer,
        text_field=ds_cfg["text_field"],
        label_field=ds_cfg["label_field"],
        max_length=int(m_cfg.get("max_length", 256)),
    )
    val_ds = tokenize_dataset(
        val_raw,
        tokenizer=tokenizer,
        text_field=ds_cfg["text_field"],
        label_field=ds_cfg["label_field"],
        max_length=int(m_cfg.get("max_length", 256)),
    )
    test_ds = tokenize_dataset(
        test_raw,
        tokenizer=tokenizer,
        text_field=ds_cfg["text_field"],
        label_field=ds_cfg["label_field"],
        max_length=int(m_cfg.get("max_length", 256)),
    )
    
    save_train_val_eval_splits(
        train_ds,
        val_ds,
        test_ds,
        output_dir=tr_cfg["output_dir"],
    )

    num_labels = infer_num_labels(train_ds)

    model = AutoModelForSequenceClassification.from_pretrained(
        m_cfg["pretrained_name"],
        num_labels=num_labels,
    )

    os.makedirs(tr_cfg["output_dir"], exist_ok=True)

    # ---- Training progress options ----
    # You can follow progress in 3 ways:
    # 1) console logs every `logging_steps`
    # 2) tqdm progress bar (default enabled)
    # 3) TensorBoard if report_to="tensorboard"
    args = TrainingArguments(
        output_dir=tr_cfg["output_dir"],
        overwrite_output_dir=bool(tr_cfg.get("overwrite_output_dir", True)),

        num_train_epochs=float(tr_cfg["num_train_epochs"]),
        learning_rate=float(tr_cfg["learning_rate"]),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),

        per_device_train_batch_size=int(tr_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tr_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(tr_cfg.get("gradient_accumulation_steps", 1)),

        warmup_ratio=float(tr_cfg.get("warmup_ratio", 0.0)),

        eval_strategy=tr_cfg.get("eval_strategy", "epoch"),
        save_strategy=tr_cfg.get("save_strategy", "epoch"),
        save_total_limit=int(tr_cfg.get("save_total_limit", 2)),

        logging_steps=int(tr_cfg.get("logging_steps", 50)),
        logging_strategy="steps",
        log_level="info",

        load_best_model_at_end=bool(tr_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=tr_cfg.get("metric_for_best_model", "f1_macro"),
        greater_is_better=bool(tr_cfg.get("greater_is_better", True)),

        dataloader_num_workers=int(rt_cfg.get("dataloader_num_workers", 0)),
        report_to=rt_cfg.get("report_to", "none"),

        seed=int(ds_cfg["seed"]),
        disable_tqdm=False,  # keep progress bar on
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=build_metrics(),
    )

    trainer.train()

    # Save final model + tokenizer (best model if load_best_model_at_end=True)
    trainer.save_model(tr_cfg["output_dir"])
    tokenizer.save_pretrained(tr_cfg["output_dir"])

    print(f"\nSaved fine-tuned model to: {tr_cfg['output_dir']}")
    
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print("\nTest set metrics:")
    print(test_metrics)
    
    save_performance_metrics(
        metrics=test_metrics,
        output_dir=tr_cfg["output_dir"],
        filename="test_metrics.json",
    )



if __name__ == "__main__":
    sentiment_cfg_path = (impresources.files(config) / "sentiment_analyser_config.yaml")
    sentiment_cfg = load_yaml(sentiment_cfg_path)
    main(sentiment_cfg)