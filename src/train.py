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
    p.add_argument("--epochs", type=int, default=None, help="ghi đè epochs (smoke test)")
    p.add_argument("--run-name", default=None, help="ghi đè run_name")
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

    history = _hf_history(trainer.state.log_history)
    pred = trainer.predict(encoded["test"])
    return model, pred.label_ids, pred.predictions.argmax(-1), history


def _hf_history(log_history) -> list[dict]:
    """Gộp log train và log eval của HF Trainer theo epoch.

    HF ghi `loss` và `eval_*` ở HAI record riêng biệt, nên nếu chỉ đọc record có
    `eval_macro_f1` thì `train_loss` luôn là NaN và trục loss của learning curve rỗng.
    """
    by_epoch: dict[int, dict] = {}
    for r in log_history:
        if "epoch" not in r:
            continue
        e = int(round(r["epoch"]))
        row = by_epoch.setdefault(e, {"epoch": e})
        if "loss" in r:
            row["train_loss"] = r["loss"]
        if "eval_loss" in r:
            row["val_loss"] = r["eval_loss"]
        if "eval_macro_f1" in r:
            row["val_macro_f1"] = r["eval_macro_f1"]
        if "eval_accuracy" in r:
            row["val_accuracy"] = r["eval_accuracy"]
    return [by_epoch[e] for e in sorted(by_epoch) if "val_macro_f1" in by_epoch[e]]


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
        emb_path = cfg["emb_path"]
        if not os.path.isabs(emb_path):
            emb_path = os.path.join(ROOT, emb_path)
        if os.path.exists(emb_path):
            emb = build_embedding_matrix(vocab, emb_path, cfg["emb_dim"])
        else:
            # Không crash: báo cáo PHẢI ghi rõ đây là khởi tạo ngẫu nhiên.
            print(f"[embedding] CẢNH BÁO: không thấy {emb_path} -> khởi tạo NGẪU NHIÊN. "
                  f"Ghi rõ điều này trong báo cáo (bảng 4.1, dòng 'Pretrained weights').")
    cfg["pretrained_emb_used"] = emb is not None

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
# Artifact chung cho MỌI nhánh — notebook 05 phụ thuộc vào các file này
# --------------------------------------------------------------------------- #
SUMMARY_KEYS = ("run_name", "family", "arch", "model_id", "seed", "accuracy",
                "macro_f1", "weighted_f1", "params", "epochs_chạy", "train_seconds",
                "inference_ms_per_sample", "peak_vram_gb", "pretrained_emb_used")


def save_run_artifacts(run_name, history, y_true, y_pred):
    """Ghi <run>_history.json + predictions/<run>.npy + predictions/y_true.npy.

    Trước đây chỉ nhánh CNN/RNN ghi các file này, nên PhoBERT/XLM-R không có
    learning curve dạng dữ liệu và không tham gia được vào so sánh McNemar ở nb 05.
    """
    import numpy as np

    logs = os.path.join(OUT_DIR, "logs")
    preds = os.path.join(logs, "predictions")
    os.makedirs(preds, exist_ok=True)

    if history:
        with open(os.path.join(logs, f"{run_name}_history.json"), "w",
                  encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(preds, f"{run_name}.npy"), np.asarray(y_pred))
    y_true_path = os.path.join(preds, "y_true.npy")
    y_true = np.asarray(y_true)
    if os.path.exists(y_true_path):
        # y_true dùng chung cho mọi mô hình; lệch thứ tự sẽ làm hỏng toàn bộ
        # so sánh chéo mà không báo lỗi -> chặn ngay tại đây.
        old = np.load(y_true_path)
        if not np.array_equal(old, y_true):
            raise RuntimeError(
                f"y_true của run '{run_name}' khác y_true đã lưu "
                f"({len(old)} vs {len(y_true)} phần tử / thứ tự khác). "
                "Test loader phải giữ nguyên thứ tự (shuffle=False) cho mọi mô hình."
            )
    else:
        np.save(y_true_path, y_true)


def make_summary(cfg, report, model, elapsed, epochs_run=None) -> dict:
    """Summary cùng một schema cho cả 4 mô hình -> bảng ở nb 05 không bị ragged."""
    summary = {k: None for k in SUMMARY_KEYS}
    summary.update({
        "run_name": cfg["run_name"],
        "family": cfg.get("family"),
        "arch": cfg.get("arch"),
        "model_id": cfg.get("model_id"),
        "seed": cfg["seed"],
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "weighted_f1": report["weighted_f1"],
        "params": count_params(model),
        "epochs_chạy": epochs_run,
        "train_seconds": round(elapsed, 1),
        "pretrained_emb_used": cfg.get("pretrained_emb_used"),
    })
    try:
        import torch

        if torch.cuda.is_available():
            summary["peak_vram_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    except ImportError:
        pass
    return summary


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
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
    save_run_artifacts(cfg["run_name"], history, y_true, y_pred)
    if history:
        plot_learning_curve(history, cfg["run_name"],
                            os.path.join(OUT_DIR, "figures", f"{cfg['run_name']}_curve.png"))

    summary = make_summary(cfg, report, model, elapsed, epochs_run=len(history))
    with open(os.path.join(OUT_DIR, "logs", f"{cfg['run_name']}_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump({**summary, "config": cfg}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
