"""Điểm vào huấn luyện chung: `python src/train.py --config configs/phobert.yaml`

Cùng một entrypoint cho cả 4 mô hình để đảm bảo giao thức so sánh công bằng
(cùng split, cùng seed, cùng tiêu chí chọn checkpoint = macro-F1 trên dev).
"""

import argparse
import json
import os
import time

import yaml

from data import (ROOT, Vocab, build_embedding_matrix, label_to_id, load_splits,
                  make_loaders, normalize, set_seed, word_segment)
from evaluate import full_report, hf_compute_metrics, plot_learning_curve
from models import BiLSTMNLI, TextCNNNLI, build_transformer, count_params

OUT_DIR = os.path.join(ROOT, "outputs")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="đường dẫn file YAML trong configs/")
    p.add_argument("--seed", type=int, default=None, help="ghi đè seed trong config")
    p.add_argument("--hypothesis-only", action="store_true",
                   help="ablation: chỉ dùng hypothesis để đo bias của dataset")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Mô hình 3 (PhoBERT) và mô hình 4 (XLM-R external)
# --------------------------------------------------------------------------- #
def train_transformer(cfg, ds, cols):
    from transformers import (DataCollatorWithPadding, EarlyStoppingCallback,
                              Trainer, TrainingArguments)

    tokenizer, model = build_transformer(cfg["model_id"], revision=cfg.get("revision"))
    prep = (lambda t: word_segment(normalize(t))) if cfg.get("word_segment") else normalize

    def tok(batch):
        premises = [""] * len(batch[cols["label"]]) if cfg.get("hypothesis_only") \
            else [prep(x) for x in batch[cols["premise"]]]
        out = tokenizer(premises, [prep(x) for x in batch[cols["hypothesis"]]],
                        truncation=True, max_length=cfg["max_length"])
        out["labels"] = [label_to_id(y) for y in batch[cols["label"]]]
        return out

    encoded = ds.map(tok, batched=True, remove_columns=ds["train"].column_names)
    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "checkpoints", cfg["run_name"]),
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg.get("grad_accum", 1),
        num_train_epochs=cfg["epochs"],
        warmup_ratio=cfg.get("warmup_ratio", 0.06),
        weight_decay=cfg.get("weight_decay", 0.01),
        fp16=cfg.get("fp16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        logging_dir=os.path.join(OUT_DIR, "logs", cfg["run_name"]),
        seed=cfg["seed"],
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=encoded["train"], eval_dataset=encoded["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.get("early_stopping_patience", 3))],
    )
    trainer.train()

    history = [
        {"epoch": int(r["epoch"]), "train_loss": r.get("loss", float("nan")),
         "val_macro_f1": r["eval_macro_f1"]}
        for r in trainer.state.log_history if "eval_macro_f1" in r
    ]
    pred = trainer.predict(encoded["test"])
    return model, pred.label_ids, pred.predictions.argmax(-1), history


# --------------------------------------------------------------------------- #
# Mô hình 1 (TextCNN) và mô hình 2 (BiLSTM)
# --------------------------------------------------------------------------- #
def train_rnn_or_cnn(cfg, ds, cols):
    import torch

    from trainer import predict, train_model

    vocab_path = os.path.join(OUT_DIR, "checkpoints", cfg["run_name"], "vocab.txt")
    vocab = Vocab.build(ds["train"][cols["premise"]] + ds["train"][cols["hypothesis"]],
                        min_freq=cfg.get("min_freq", 2), segment=cfg.get("segment", True))
    vocab.save(vocab_path)
    print(f"[vocab] {len(vocab)} token (xây từ TRAIN only) -> {vocab_path}")

    emb = None
    if cfg.get("pretrained_emb") not in (None, "none") and cfg.get("emb_path"):
        emb = build_embedding_matrix(vocab, cfg["emb_path"], cfg["emb_dim"])

    common = dict(vocab_size=len(vocab), emb_dim=cfg["emb_dim"],
                  dropout=cfg["dropout"], pretrained_emb=emb)
    if cfg["arch"] == "textcnn":
        model = TextCNNNLI(n_filters=cfg["n_filters"],
                           kernel_sizes=tuple(cfg["kernel_sizes"]), **common)
    elif cfg["arch"] == "bilstm":
        model = BiLSTMNLI(hidden=cfg["hidden"], num_layers=cfg["num_layers"], **common)
    else:
        raise ValueError(f"arch không hợp lệ: {cfg['arch']}")

    loaders = make_loaders(ds, cols, vocab, batch_size=cfg["batch_size"],
                           max_len=cfg["max_length"], segment=cfg.get("segment", True),
                           hypothesis_only=cfg.get("hypothesis_only", False),
                           num_workers=cfg.get("num_workers", 2))

    model, history = train_model(
        model, loaders, cfg,
        log_path=os.path.join(OUT_DIR, "logs", f"{cfg['run_name']}_history.json"),
    )
    device = next(model.parameters()).device
    torch.save(model.state_dict(),
               os.path.join(OUT_DIR, "checkpoints", cfg["run_name"], "best.pt"))
    y_true, y_pred = predict(model, loaders["test"], device)
    return model, y_true, y_pred, history


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.hypothesis_only:
        cfg["hypothesis_only"] = True
        cfg["run_name"] += "_hyponly"
    set_seed(cfg["seed"])

    ds, cols = load_splits(cache_dir=cfg.get("cache_dir"), seed=cfg["seed"])
    print(f"[data] cột thật: {cols}")
    print(f"[data] " + " ".join(f"{k}={len(v)}" for k, v in ds.items()))

    os.makedirs(os.path.join(OUT_DIR, "checkpoints", cfg["run_name"]), exist_ok=True)
    started = time.time()
    trainer_fn = train_transformer if cfg["family"] == "transformer" else train_rnn_or_cnn
    model, y_true, y_pred, history = trainer_fn(cfg, ds, cols)
    elapsed = time.time() - started

    report = full_report(
        y_true, y_pred,
        out_json=os.path.join(OUT_DIR, "logs", f"{cfg['run_name']}_test.json"),
    )
    if history:
        plot_learning_curve(history, cfg["run_name"],
                            os.path.join(OUT_DIR, "figures", f"{cfg['run_name']}_curve.png"))

    summary = {"run_name": cfg["run_name"], "seed": cfg["seed"],
               "accuracy": report["accuracy"], "macro_f1": report["macro_f1"],
               "weighted_f1": report["weighted_f1"],
               "params": count_params(model), "train_seconds": round(elapsed, 1)}
    with open(os.path.join(OUT_DIR, "logs", f"{cfg['run_name']}_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump({**summary, "config": cfg}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
