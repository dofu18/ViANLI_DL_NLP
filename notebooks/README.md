# Notebooks (chạy trên Kaggle, theo thứ tự)

| Notebook | Nội dung | Xuất ra |
|---|---|---|
| `01_eda.ipynb` | In schema thật, phân bố nhãn, histogram độ dài, top n-gram theo nhãn, kiểm tra leakage, **hypothesis-only baseline** | hình vào `outputs/figures/`, split cố định vào `data/splits/` |
| `02_cnn_rnn.ipynb` | Vocab từ train-only, nạp PhoW2V, vòng lặp train TextCNN + BiLSTM, early stopping theo macro-F1 dev | checkpoint + log |
| `03_phobert.ipynb` | Fine-tune `vinai/phobert-base-v2` (gọi `configs/phobert.yaml`) | checkpoint + log |
| `04_xlmr_external.ipynb` | `xlm-roberta-large`: zero-shot XNLI **và** fine-tune trên ViANLI | checkpoint + log |
| `05_analysis.ipynb` | Bảng so sánh 4 mô hình, learning curve, confusion matrix, ablation, phân tích lỗi 15–20 ví dụ | hình + bảng cho báo cáo |

Ghi chú Kaggle:
- Bật GPU + Internet; dùng **Save Version → Save & Run All** để chạy nền, tránh mất session 12h.
- Ghi checkpoint ra `/kaggle/working`, sau đó tải về `outputs/checkpoints/`.
- Không hard-code Kaggle/HF token trong notebook — dùng Kaggle Secrets.
