"""Vòng lặp huấn luyện PyTorch cho nhánh CNN/RNN.

Tiêu chí chọn checkpoint giống hệt nhánh Transformer (macro-F1 trên dev) để đảm bảo
nguyên tắc so sánh công bằng của đề bài.
"""

import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from evaluate import compute_metrics_from_preds


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (y_true, y_pred) trên toàn bộ loader."""
    model.eval()
    trues, preds = [], []
    for p_ids, h_ids, y in loader:
        logits = model(p_ids.to(device), h_ids.to(device))
        preds.append(logits.argmax(-1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def train_model(model, loaders, cfg, device=None, log_path=None):
    """Huấn luyện với early stopping theo macro-F1 dev.

    Trả về (model tốt nhất, history) — history dùng để vẽ learning curve.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg["class_weights"], dtype=torch.float, device=device)
        if cfg.get("class_weights") else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=max(1, cfg.get("early_stopping_patience", 5) // 2),
    )

    history = []
    best_score, best_state, bad_epochs = -1.0, None, 0
    started = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for p_ids, h_ids, y in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(p_ids.to(device), h_ids.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip_grad", 5.0))
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        y_true, y_pred = predict(model, loaders["validation"], device)
        metrics = compute_metrics_from_preds(y_true, y_pred)
        scheduler.step(metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        history.append(row)
        print(f"epoch {epoch:>3} | loss {row['train_loss']:.4f} | "
              f"val_acc {metrics['accuracy']:.4f} | val_macroF1 {metrics['macro_f1']:.4f}")

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.get("early_stopping_patience", 5):
                print(f"Early stopping ở epoch {epoch} (best macro-F1 dev = {best_score:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"history": history, "best_val_macro_f1": best_score,
                       "train_seconds": time.time() - started},
                      f, ensure_ascii=False, indent=2)

    return model, history
