"""Metric dùng chung cho cả 4 mô hình — không mô hình nào được dùng metric riêng."""

import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from data import LABELS


def compute_metrics_from_preds(y_true, y_pred) -> dict:
    """Metric chính: Accuracy. Metric phụ: Macro-F1, Weighted-F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def hf_compute_metrics(eval_pred):
    """Callback cho transformers.Trainer."""
    logits, labels = eval_pred
    return compute_metrics_from_preds(labels, np.argmax(logits, axis=-1))


def full_report(y_true, y_pred, out_json: str | None = None) -> dict:
    """Báo cáo đầy đủ: metric tổng, per-class, confusion matrix."""
    report = {
        **compute_metrics_from_preds(y_true, y_pred),
        "per_class": classification_report(
            y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def plot_confusion_matrix(y_true, y_pred, title: str, out_png: str):
    """Xuất confusion matrix cho §Phân tích lỗi."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_learning_curve(history, title: str, out_png: str):
    """Vẽ train loss và val macro-F1 theo epoch (§Đường cong huấn luyện)."""
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_macro_f1"] for h in history],
             marker="s", color="tab:orange", label="val macro-F1")
    ax2.set_ylabel("Val macro-F1")
    ax1.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def error_examples(ds_test, cols, y_true, y_pred, n: int = 20) -> list[dict]:
    """Trích ví dụ dự đoán sai để phân tích lỗi thủ công."""
    wrong = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p][:n]
    return [
        {
            "premise": ds_test[cols["premise"]][i],
            "hypothesis": ds_test[cols["hypothesis"]][i],
            "gold": LABELS[y_true[i]],
            "pred": LABELS[y_pred[i]],
        }
        for i in wrong
    ]
