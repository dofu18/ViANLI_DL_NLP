# TO-DO — ViANLI (cập nhật 2026-08-07, tối)

## Đang chạy trên Kaggle qua đêm

- [ ] `03_phobert.ipynb` — Save & Run All
- [ ] `04_xlmr_external.ipynb` — Save & Run All

Sáng mai mở tab **Versions** của từng notebook xem đã xong chưa / có lỗi không.

---

## VIỆC ĐẦU TIÊN SÁNG MAI — kiểm tra LayerNorm của PhoBERT

Warning `unexpected keys: roberta...LayerNorm.beta/gamma` **chưa được xác minh là vô hại**.
Nếu trọng số LayerNorm không nạp được thì PhoBERT mất một phần kiến thức pretrain và mọi
kết quả của notebook 03 phải chạy lại.

### Cách 1 — nhìn log (nhanh nhất, không cần chạy gì)

Mở log notebook 03, xem `eval_macro_f1` sau epoch 1:

| Giá trị | Kết luận |
|---|---|
| rõ ràng > 0.35 và tăng dần | Nạp đúng. Bỏ qua warning, đi tiếp. |
| lẹt đẹt ~0.33 giống TextCNN | Nhiều khả năng LayerNorm hỏng → làm Cách 2 |

### Cách 2 — kiểm tra trực tiếp

```python
from transformers import AutoModel
import torch

m = AutoModel.from_pretrained("vinai/phobert-base-v2")
ln = m.encoder.layer[0].output.LayerNorm
print("là giá trị mặc định?",
      torch.allclose(ln.weight, torch.ones_like(ln.weight))
      and torch.allclose(ln.bias, torch.zeros_like(ln.bias)))
```

- `False` → ổn, đi tiếp.
- `True` → hỏng. Sửa bằng `!pip install -q "transformers==4.44.2"` rồi restart kernel
  và chạy lại notebook 03. (Bản 4.44.2 còn tự đổi tên `gamma/beta` → `weight/bias`.)

---

## Sau khi 03 và 04 xong

- [ ] Tải Output của notebook 03 và 04 về máy (Download All)
- [ ] Chép vào `D:\NLP`, giữ nguyên cấu trúc:
  - `outputs/logs/*.json` `*.csv`
  - `outputs/logs/predictions/*.npy`
  - `outputs/figures/*.png`
  - **Không** chép `outputs/checkpoints/`
- [ ] `git add outputs && git commit -m "kết quả 03, 04" && git push`
- [ ] Chạy `05_analysis.ipynb` (CPU, ~2 phút) — sinh toàn bộ bảng + hình cho báo cáo
- [ ] Tải output notebook 05 về, commit tiếp

---

## Tình hình hiện tại

### Đã xong

- Pipeline core: `src/data.py` `models.py` `trainer.py` `train.py` `evaluate.py`
- 5 notebook, đã có cell tự clone repo từ GitHub
- `01_eda.ipynb` đã chạy → `data/splits/*.csv`, `outputs/logs/eda.json`
- `02_cnn_rnn.ipynb` đã chạy → TextCNN + BiLSTM + ablation
- `max_length` chốt 128 (p95: premise 51 từ, hypothesis 28 từ)
- Xác nhận label mapping: `0=entailment, 1=neutral, 2=contradiction`
- Commit hash dataset: `0fec8d6ecb043a61c609f9b51f80401fdf1e84d3`

### Số liệu đã có

| Baseline / Mô hình | Accuracy (test) |
|---|---|
| Majority | 0.334 |
| TF-IDF hypothesis-only | 0.318 |
| TF-IDF premise-only | 0.039 |
| TF-IDF premise + hypothesis | 0.105 |
| TextCNN | ~0.30 |
| BiLSTM + attention | ~0.30 |
| PhoBERT | *đang chạy* |
| XLM-R large | *đang chạy* |

Đặc điểm dữ liệu: 8012 / 1000 / 1000, val-test cân bằng chính xác, 865/1000 premise của
test cũng có trong train (theo thiết kế, **không phải leakage**), chỉ 2 cặp
(premise, hypothesis) trùng train↔test.

### Kết luận đã chốt về CNN/RNN

Train loss xuống 0.002 nhưng val đứng ~0.33 → mô hình học thuộc lòng, không tổng quát hóa.
**Không phải bug, không cần sửa.** Đây là tư liệu cho §So sánh các họ kiến trúc.
Lưu ý khi viết: test chỉ 1000 mẫu, sai số ±1.5 điểm → viết "xấp xỉ mức ngẫu nhiên",
đừng viết "thấp hơn ngẫu nhiên".

---

## Tùy chọn nếu còn thời gian (không bắt buộc)

Xếp theo giá trị:

1. **ESIM cho mô hình 2** — BiLSTM có co-attention. Vừa tăng điểm, vừa cho luận điểm
   "cross-attention mới là yếu tố quyết định, không phải Transformer". Đáng làm nhất.
2. **Nạp PhoW2V** cho TextCNN/BiLSTM thay random init (hiện `EMB_PATH = None`).
   Nếu không làm thì phải ghi rõ trong báo cáo là dùng random init.
3. Quét learning rate PhoBERT trên dev {1e-5, 2e-5, 3e-5}.
4. Ensemble PhoBERT + XLM-R bằng cộng logit.
5. PhoBERT-large.

---

## Viết báo cáo (`reports/main.tex`)

Chưa động tới. Thứ tự nên làm:

- [ ] Điền metadata đầu file: CourseName, ReportTitle, DatasetName, tên + MSSV, ngày nộp
- [ ] Xóa bảng đóng góp thành viên (bài cá nhân)
- [ ] Chương Bộ dữ liệu — dùng số từ `outputs/logs/eda.json`
- [ ] Chương Mô hình — 4 bảng cấu hình + bảng thông tin nguồn mô hình ngoài
  (chép từ đầu `04_xlmr_external.ipynb`, nhớ điền commit hash)
- [ ] Chương Thiết kế thực nghiệm — bảng môi trường (chạy cell in thông tin ở mục 9
  của `KAGGLE.md`), bảng siêu tham số
- [ ] Chương Kết quả — bảng LaTeX do notebook 05 xuất sẵn + các hình
- [ ] Phân tích lỗi — mở `outputs/logs/error_examples.csv`, **tự điền cột "Nhóm lỗi"**
  (phủ định / số / kiến thức thế giới / đồng tham chiếu / nhãn sai). Không tự động hóa được.
- [ ] Tóm tắt 200–300 từ (viết sau cùng)
- [ ] Trích dẫn: ViANLI, PhoBERT, XLM-R, ESIM/TextCNN
- [ ] Rà 2 checklist trong `PLAN.md` (tái lập + tự đánh giá)

---

## Nhắc lại

ViANLI là bộ adversarial. Kết quả tốt nhất trong paper gốc cũng chỉ quanh 45–50%
(**tự kiểm tra con số chính xác trong paper trước khi trích dẫn**). PhoBERT đạt ~0.45
đã là tốt. Điểm số của đồ án nằm ở quy trình thực nghiệm chặt chẽ và phân tích lỗi có
chiều sâu, không phải ở accuracy cao.
