"""Bốn kiến trúc theo yêu cầu đề bài (CNN / RNN / Transformer / External)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import NUM_LABELS


def pair_features(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Ghép biểu diễn hai câu: [u; v; |u-v|; u*v] — chuẩn của dòng ESIM/InferSent."""
    return torch.cat([u, v, (u - v).abs(), u * v], dim=-1)


class TextCNNNLI(nn.Module):
    """Mô hình 1 — CNN-based. Conv1D đa kernel + max-pool cho từng câu."""

    def __init__(self, vocab_size, emb_dim=300, n_filters=128,
                 kernel_sizes=(2, 3, 4, 5), dropout=0.5, pretrained_emb=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.emb.weight.data.copy_(torch.as_tensor(pretrained_emb))
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, n_filters, k, padding=k - 1) for k in kernel_sizes]
        )
        enc_dim = n_filters * len(kernel_sizes)
        self.head = nn.Sequential(
            nn.Linear(enc_dim * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, NUM_LABELS),
        )

    def encode(self, ids):
        x = self.emb(ids).transpose(1, 2)
        return torch.cat([F.relu(c(x)).max(dim=-1).values for c in self.convs], dim=-1)

    def forward(self, premise_ids, hypothesis_ids):
        return self.head(pair_features(self.encode(premise_ids),
                                       self.encode(hypothesis_ids)))


class BiLSTMNLI(nn.Module):
    """Mô hình 2 — RNN/LSTM/GRU-based. BiLSTM chia sẻ trọng số + attention pooling."""

    def __init__(self, vocab_size, emb_dim=300, hidden=256, num_layers=1,
                 dropout=0.5, pretrained_emb=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.emb.weight.data.copy_(torch.as_tensor(pretrained_emb))
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, NUM_LABELS),
        )

    def encode(self, ids):
        h, _ = self.lstm(self.emb(ids))
        mask = (ids != 0).unsqueeze(-1)
        # Câu rỗng (toàn PAD) xuất hiện ở ablation hypothesis-only và khi hypothesis
        # bị cắt hết. Dùng -1e4 thay -inf: -inf trên cả hàng làm softmax ra NaN và
        # đầu độc toàn bộ gradient (đây chính là lỗi của run bilstm_hyponly cũ).
        scores = self.attn(h).masked_fill(~mask, -1e4)
        weights = scores.softmax(dim=1)
        empty = ~mask.any(dim=1, keepdim=True)  # (B, 1, 1)
        weights = weights.masked_fill(empty, 0.0)  # câu rỗng -> vector 0
        return (h * weights).sum(dim=1)

    def forward(self, premise_ids, hypothesis_ids):
        return self.head(pair_features(self.encode(premise_ids),
                                       self.encode(hypothesis_ids)))


def build_transformer(model_id="vinai/phobert-base-v2", revision=None):
    """Mô hình 3 — Transformer-based: fine-tune PhoBERT dạng cross-encoder.

    Mô hình 4 — External: gọi hàm này với 'xlm-roberta-large' hoặc
    'joeddav/xlm-roberta-large-xnli' (nhớ ghi URL, revision, license vào báo cáo).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=NUM_LABELS, ignore_mismatched_sizes=True, **kwargs
    )
    return tokenizer, model


def count_params(model) -> int:
    """Số tham số huấn luyện — điền vào bảng kết quả chính."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
