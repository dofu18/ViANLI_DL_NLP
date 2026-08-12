"""Nguồn sự thật duy nhất về dữ liệu ViANLI cho cả 4 mô hình.

Mọi mô hình phải gọi `load_splits()` từ đây để đảm bảo nguyên tắc so sánh công bằng:
cùng split, cùng seed, cùng label mapping.

Ghi chú: file được đặt tên `data.py` (thay vì `datasets.py` như mẫu repo trong đề bài)
để tránh che khuất (shadow) package `datasets` của Hugging Face khi chạy
`python src/train.py`.
"""

import os
import random
import re
import unicodedata

import numpy as np

LABELS = ["entailment", "neutral", "contradiction"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

HF_DATASET_ID = "uitnlp/ViANLI"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(ROOT, "data", "splits")

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1


# --------------------------------------------------------------------------- #
# Seed & chuẩn hóa văn bản
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    """Cố định seed cho random/numpy/torch. Seed phải được ghi trong báo cáo.

    Ghi chú: PYTHONHASHSEED phải được đặt TRƯỚC khi interpreter khởi động mới có tác
    dụng, nên không đặt ở đây (đặt cũng vô ích). Khi cần, chạy:
    `PYTHONHASHSEED=42 python src/train.py ...`
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def normalize(text: str) -> str:
    """Chuẩn hóa Unicode NFC + gộp khoảng trắng. Áp dụng cho mọi nhánh mô hình."""
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()


_SEGMENTER = None  # cache backend đã dò được: ("underthesea", fn) hoặc ("none", None)


def _get_segmenter():
    """Dò backend tách từ một lần duy nhất và in ra backend đang dùng.

    Việc in ra là bắt buộc: bản chạy Kaggle trước đây rơi vào nhánh fallback im lặng
    nên PhoBERT được huấn luyện trên văn bản CHƯA tách từ mà không ai biết, khiến
    ablation `no_wordseg` trùng khít baseline.
    """
    global _SEGMENTER
    if _SEGMENTER is not None:
        return _SEGMENTER
    try:
        from underthesea import word_tokenize

        _SEGMENTER = ("underthesea", lambda t: word_tokenize(t, format="text"))
    except ImportError:
        _SEGMENTER = ("none", None)
    print(f"[word_segment] backend = {_SEGMENTER[0]}")
    return _SEGMENTER


def word_segment(text: str, strict: bool = True) -> str:
    """Tách từ tiếng Việt (bắt buộc trước khi đưa vào PhoBERT).

    strict=True (mặc định): thiếu backend thì raise, KHÔNG âm thầm trả nguyên văn.
    strict=False: cho phép fallback — chỉ dùng cho nhánh CNN/RNN, nơi tách từ là
    tùy chọn chứ không phải yêu cầu của tokenizer.
    """
    name, fn = _get_segmenter()
    if fn is None:
        if strict:
            raise RuntimeError(
                "Không có backend tách từ tiếng Việt. PhoBERT-base-v2 yêu cầu đầu vào "
                "đã tách từ. Cài bằng: pip install underthesea"
            )
        return text
    return fn(text)


def tokenize(text: str, segment: bool = True) -> list[str]:
    """Tokenizer word-level cho nhánh CNN/RNN (nhánh A)."""
    text = normalize(text).lower()
    if segment:
        # strict=False: nhánh CNN/RNN vẫn chạy được khi thiếu underthesea, khác với
        # nhánh PhoBERT (tokenizer của nó giả định đầu vào đã tách từ).
        text = word_segment(text, strict=False)
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


# --------------------------------------------------------------------------- #
# Load split
# --------------------------------------------------------------------------- #
def _resolve_columns(split) -> dict:
    """Dò tên cột thật của dataset thay vì hard-code."""
    cols = {c.lower(): c for c in split.column_names}

    def pick(*candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        raise KeyError(f"Không tìm thấy cột nào trong {candidates}; có: {split.column_names}")

    return {
        "premise": pick("premise", "sentence1", "context", "text_a"),
        "hypothesis": pick("hypothesis", "sentence2", "text_b"),
        "label": pick("label", "gold_label", "labels"),
    }


def load_local_splits():
    """Đọc split ĐÃ CHỐT trong data/splits/*.csv (do 01_eda.ipynb sinh ra).

    Trả về None nếu chưa có file. Đây là nguồn duy nhất đảm bảo CLI và notebook
    chạy trên cùng một split — yêu cầu "so sánh công bằng" của đề bài.
    """
    paths = {name: os.path.join(SPLIT_DIR, f"{name}.csv")
             for name in ("train", "validation", "test")}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    import pandas as pd

    frames = {name: pd.read_csv(p, encoding="utf-8") for name, p in paths.items()}
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        # Nhánh CNN/RNN không cần Hugging Face datasets — chỉ nhánh Transformer mới cần.
        return {name: _CsvSplit(df) for name, df in frames.items()}
    return DatasetDict({name: Dataset.from_pandas(df, preserve_index=False)
                        for name, df in frames.items()})


class _CsvSplit:
    """Split đọc từ CSV với API tối thiểu mà nhánh CNN/RNN dùng tới."""

    def __init__(self, df):
        self._df = df
        self.column_names = list(df.columns)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._df[key].tolist()
        return self._df.iloc[key].to_dict()


def load_splits(cache_dir: str | None = None, seed: int = 42, prefer_local: bool = True):
    """Trả về (DatasetDict, dict tên cột) với các split gốc của ViANLI.

    Ưu tiên `data/splits/*.csv` nếu có; nếu không thì tải từ Hugging Face.
    Nếu không có split validation, tách 10% từ train (stratify theo nhãn, seed cố
    định). Test set không bao giờ bị đụng tới ở đây.
    """
    if prefer_local:
        local = load_local_splits()
        if local is not None:
            print(f"[data] dùng split cố định trong {SPLIT_DIR}")
            return local, _resolve_columns(local["train"])

    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, cache_dir=cache_dir)
    if "dev" in ds and "validation" not in ds:
        ds["validation"] = ds.pop("dev")
    if "validation" not in ds:
        cols = _resolve_columns(ds["train"])
        try:
            split = ds["train"].train_test_split(
                test_size=0.1, seed=seed, stratify_by_column=cols["label"]
            )
        except (ValueError, TypeError):  # label không phải ClassLabel
            split = ds["train"].train_test_split(test_size=0.1, seed=seed)
        ds["train"], ds["validation"] = split["train"], split["test"]
    return ds, _resolve_columns(ds["train"])


def describe(ds) -> None:
    """In schema thật của dataset. LUÔN chạy trước khi viết code phía sau."""
    for name, split in ds.items():
        print(f"[{name}] n={len(split)} columns={split.column_names}")
        print(f"  features: {split.features}")
        print(f"  ví dụ  : {split[0]}")


def label_to_id(value) -> int:
    """Chuẩn hóa nhãn về 0/1/2 dù dataset lưu dạng int hay string."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return LABEL2ID[str(value).strip().lower()]


def check_leakage(ds, cols) -> dict:
    """Đếm premise/cặp câu xuất hiện ở cả train và test (§Phòng tránh rò rỉ)."""

    def pairs(split):
        return {
            (normalize(p), normalize(h))
            for p, h in zip(split[cols["premise"]], split[cols["hypothesis"]])
        }

    def premises(split):
        return {normalize(p) for p in split[cols["premise"]]}

    train, test = ds["train"], ds["test"]
    return {
        "overlap_pairs_train_test": len(pairs(train) & pairs(test)),
        "overlap_premises_train_test": len(premises(train) & premises(test)),
    }


# --------------------------------------------------------------------------- #
# Vocab (chỉ xây từ TRAIN — không nhìn dev/test)
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        ids = [self.stoi.get(t, UNK_ID) for t in tokens[:max_len]]
        return ids + [PAD_ID] * (max_len - len(ids))

    @classmethod
    def build(cls, texts, min_freq: int = 2, segment: bool = True) -> "Vocab":
        from collections import Counter

        counter = Counter()
        for text in texts:
            counter.update(tokenize(text, segment=segment))
        itos = [PAD, UNK] + sorted(
            [t for t, c in counter.items() if c >= min_freq],
            key=lambda t: (-counter[t], t),
        )
        return cls(itos)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.itos))

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            return cls(f.read().split("\n"))


def build_embedding_matrix(vocab: Vocab, w2v_path: str, dim: int = 300) -> np.ndarray:
    """Nạp PhoW2V/fastText và ánh xạ sang vocab. Từ ngoài từ điển → khởi tạo ngẫu nhiên.

    Trả về ma trận (len(vocab), dim); in tỉ lệ hit để báo cáo trong phần ablation.
    """
    rng = np.random.default_rng(42)
    matrix = rng.normal(0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    matrix[PAD_ID] = 0.0

    hits = 0
    with open(w2v_path, encoding="utf-8") as f:
        first = f.readline().split()
        if len(first) > 2:  # không có header -> xử lý lại dòng đầu
            f.seek(0)
        for line in f:
            parts = line.rstrip().split(" ")
            token = parts[0]
            idx = vocab.stoi.get(token)
            if idx is not None and len(parts) == dim + 1:
                matrix[idx] = np.asarray(parts[1:], dtype=np.float32)
                hits += 1
    print(f"[embedding] phủ {hits}/{len(vocab)} token ({hits / len(vocab):.1%})")
    return matrix


# --------------------------------------------------------------------------- #
# torch Dataset cho nhánh CNN/RNN
# --------------------------------------------------------------------------- #
class NLIPairDataset:
    """Trả về (premise_ids, hypothesis_ids, label) — dùng cho TextCNN và BiLSTM."""

    def __init__(self, split, cols, vocab: Vocab, max_len: int = 128,
                 segment: bool = True, hypothesis_only: bool = False):
        self.premise = split[cols["premise"]]
        self.hypothesis = split[cols["hypothesis"]]
        self.labels = [label_to_id(y) for y in split[cols["label"]]]
        self.vocab, self.max_len, self.segment = vocab, max_len, segment
        self.hypothesis_only = hypothesis_only  # nhánh ablation đo bias dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        import torch

        p = "" if self.hypothesis_only else self.premise[i]
        p_ids = self.vocab.encode(tokenize(p, self.segment), self.max_len)
        h_ids = self.vocab.encode(tokenize(self.hypothesis[i], self.segment), self.max_len)
        return (
            torch.tensor(p_ids, dtype=torch.long),
            torch.tensor(h_ids, dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
        )


def _worker_init(worker_id: int):
    """Mỗi worker có seed dẫn xuất cố định — bắt buộc để tái lập với num_workers>0."""
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)


def make_loaders(ds, cols, vocab, batch_size=64, max_len=128, segment=True,
                 hypothesis_only=False, num_workers=2, seed=42):
    """Tạo DataLoader cho train/validation/test với cùng cấu hình."""
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders = {}
    for name in ("train", "validation", "test"):
        dataset = NLIPairDataset(ds[name], cols, vocab, max_len, segment, hypothesis_only)
        loaders[name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=num_workers, drop_last=(name == "train"),
            worker_init_fn=_worker_init, generator=generator,
        )
    return loaders
