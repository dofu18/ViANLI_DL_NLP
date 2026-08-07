# ViANLI — Suy luận ngôn ngữ tự nhiên tiếng Việt

Đồ án thực hành Deep Learning (AIN501 / FSB) — **bài cá nhân**.
Phân loại 3 lớp cho cặp (premise, hypothesis): `entailment` / `neutral` / `contradiction`.

- **Dataset:** [uitnlp/ViANLI](https://huggingface.co/datasets/uitnlp/ViANLI) — bộ adversarial NLI tiếng Việt của UIT-NLP.
- **Nền tảng huấn luyện:** Kaggle Notebooks (GPU T4 x2 / P100).
- **Kế hoạch chi tiết + checklist:** [`PLAN.md`](PLAN.md).
- **Hướng dẫn chạy trên Kaggle:** [`KAGGLE.md`](KAGGLE.md).

## Bốn mô hình (theo yêu cầu đề bài)

| # | Họ kiến trúc | Mô hình | Config |
|---|---|---|---|
| 1 | CNN-based | TextCNN kiểu ESIM-lite, ghép `[u; v; abs(u-v); u*v]` | `configs/textcnn.yaml` |
| 2 | RNN/LSTM/GRU | BiLSTM chia sẻ trọng số + attention pooling | `configs/bilstm.yaml` |
| 3 | Transformer | Fine-tune `vinai/phobert-base-v2` (cross-encoder) | `configs/phobert.yaml` |
| 4 | External | `xlm-roberta-large` (+ so sánh zero-shot XNLI) | `configs/xlmr_external.yaml` |

## Cấu trúc thư mục

```
.
|-- README.md
|-- PLAN.md                 # kế hoạch + toàn bộ checklist
|-- requirements.txt
|-- configs/                # siêu tham số của 4 mô hình
|-- data/
|   |-- raw/                # dữ liệu & embedding PhoW2V tải về (không commit)
|   |-- splits/             # split cố định dùng chung cho cả 4 mô hình
|-- notebooks/              # 01_eda … 05_analysis (chạy trên Kaggle)
|-- src/
|   |-- data.py             # load split, chuẩn hóa, tách từ, Vocab, DataLoader, seed
|   |-- models.py           # 4 kiến trúc
|   |-- trainer.py          # vòng lặp PyTorch + early stopping (nhánh CNN/RNN)
|   |-- train.py            # entrypoint chung cho cả 4 mô hình
|   |-- evaluate.py         # metric, confusion matrix, learning curve, ví dụ lỗi
|-- outputs/
|   |-- logs/ checkpoints/ figures/
|-- reports/                # main.tex + ảnh nền + PDF
```

> `src/data.py` được đặt tên khác mẫu repo trong đề bài (`datasets.py`) để tránh che
> khuất package `datasets` của Hugging Face khi chạy `python src/train.py`.

## Chuẩn bị dữ liệu

```bash
pip install -r requirements.txt
python -c "import sys; sys.path.insert(0,'src'); from data import load_splits, describe, check_leakage; ds, cols = load_splits(); describe(ds); print(cols); print(check_leakage(ds, cols))"
```

In schema thật (tên cột, số mẫu, label mapping) **trước khi** chạy train. `load_splits()`
tự dò tên cột (`premise`/`sentence1`/…), nhưng vẫn nên xác nhận bằng mắt.

Nếu Kaggle Notebook không bật được Internet: tải dataset về, upload thành một Kaggle Dataset
private rồi truyền `cache_dir: /kaggle/input/<tên-dataset>` trong config.

Embedding cho TextCNN/BiLSTM: tải [PhoW2V 300 chiều](https://github.com/datquocnguyen/PhoW2V)
vào `data/raw/`, hoặc đặt `pretrained_emb: none` để chạy nhánh ablation random init.

## Huấn luyện & đánh giá

```bash
python src/train.py --config configs/textcnn.yaml
python src/train.py --config configs/bilstm.yaml
python src/train.py --config configs/phobert.yaml
python src/train.py --config configs/xlmr_external.yaml

# ablation: đo bias của dataset bằng hypothesis-only baseline
python src/train.py --config configs/bilstm.yaml --hypothesis-only

# ablation: độ ổn định theo seed
python src/train.py --config configs/phobert.yaml --seed 1337
```

Mỗi lần chạy sinh ra:

| File | Nội dung |
|---|---|
| `outputs/logs/<run>_test.json` | accuracy, macro-F1, per-class, confusion matrix trên test |
| `outputs/logs/<run>_summary.json` | metric + số tham số + thời gian train + config đã dùng |
| `outputs/logs/<run>_history.json` | lịch sử theo epoch (nhánh CNN/RNN) |
| `outputs/figures/<run>_curve.png` | learning curve |
| `outputs/checkpoints/<run>/` | checkpoint tốt nhất + `vocab.txt` |

## Giao thức so sánh công bằng

- Cùng split train/dev/test cho cả 4 mô hình (`src/data.py:load_splits`).
- **Seed = 42** cố định cho random / numpy / torch.
- Metric chính: **Accuracy**; phụ: Macro-F1, Weighted-F1, per-class, confusion matrix.
- Chọn checkpoint tốt nhất theo **macro-F1 trên dev**; test set chỉ chạy một lần cuối.
- Không tune siêu tham số trên test set.
- Vocab của TextCNN/BiLSTM chỉ xây từ tập train.

## Trạng thái

- [x] Pipeline core: `data.py`, `models.py`, `trainer.py`, `evaluate.py`, `train.py`.
- [ ] Notebook EDA + xác nhận schema thật của ViANLI.
- [ ] Chạy 4 mô hình trên Kaggle, điền kết quả vào `reports/main.tex`.

## Trích dẫn

- ViANLI — UIT-NLP (điền trích dẫn chính thức của paper).
- PhoBERT — Nguyen & Nguyen, 2020.
- XLM-R — Conneau et al., 2020 (MIT license).
- PhoW2V — Nguyen et al., 2020.
