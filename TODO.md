# TO-DO — ViANLI (cập nhật 2026-08-14)

## Trạng thái: 4 mô hình chính + 8 nhánh ablation đã chạy xong

Toàn bộ kết quả Kaggle đã gộp vào `outputs/`. Đã audit chéo `*_summary.json` ↔
`*_test.json` ↔ `predictions/*.npy` cho cả 12 run — khớp 100%, `y_true.npy` giống
nhau ở mọi lần chạy nên 4 mô hình so sánh được với nhau.

### Kết quả test (1000 mẫu, sai số ±1.5 điểm)

| Run | Accuracy | Macro-F1 | Ghi chú |
|---|---|---|---|
| Majority baseline | 0.334 | — | mốc ngẫu nhiên |
| TF-IDF hypothesis-only | 0.318 | — | |
| **textcnn** | 0.322 | — | mô hình 1 |
| **bilstm_attn** | 0.374 | — | mô hình 2 |
| **phobert_base_v2** | 0.421 | 0.4209 | mô hình 3 |
| **xlmr_large_finetuned** | 0.449 | 0.3721 | mô hình 4, 5 epoch, 2807s |
| xlmr_xnli_finetuned | 0.423 | **0.4190** | ablation: khởi tạo từ checkpoint XNLI |
| xlmr_zeroshot | 0.334 | 0.3246 | ablation: không fine-tune |
| bilstm_hyponly | 0.334 | — | ablation |
| bilstm_dropout02 | 0.352 | — | ablation |
| bilstm_maxlen64 | 0.349 | — | ablation |
| phobert_hyponly | 0.375 | — | ablation |
| phobert_lr5e-5 | 0.418 | — | ablation |
| phobert_no_wordseg | 0.421 | 0.4209 | **KHÔNG hợp lệ — xem bên dưới** |

**Cảnh báo:** mọi số của TextCNN, BiLSTM và PhoBERT ở trên đều là kết quả chạy trên văn
bản **chưa tách từ** (xem mục 1 phần Việc còn lại). Chỉ 3 run XLM-R là dùng được ngay.

Điểm đáng viết vào báo cáo: `xlmr_large_finetuned` accuracy cao nhất (0.449) nhưng
macro-F1 chỉ 0.372 vì gần như bỏ hẳn lớp *contradiction* (phân bố dự đoán
465/514/**21**). `xlmr_xnli_finetuned` accuracy thấp hơn (0.423) nhưng dự đoán cân
(346/399/255) nên macro-F1 cao nhất toàn bộ thí nghiệm (0.419). Đây là luận điểm
"accuracy một mình không đủ để kết luận".

---

## Việc còn lại

### 1. Chạy lại `02_cnn_rnn.ipynb` VÀ `03_phobert.ipynb` — cả hai train trên text chưa tách từ

`underthesea` không có sẵn trên Kaggle → `word_segment()` fallback im lặng → **9 run
trong `outputs/` hiện tại đều dùng văn bản chưa tách từ**, dù `SEGMENT = True`.

Bằng chứng:

- `outputs/checkpoints/shared_vocab.txt`: 0/5365 token chứa `_` (nb 02)
- `phobert_no_wordseg` trùng **từng bit** với `phobert_base_v2`, macro-F1 khớp tới 14
  chữ số → nhánh "bỏ tách từ" thực chất không bỏ gì (nb 03)

Mức độ khác nhau giữa hai notebook:

- **nb 03 là lỗi thật.** PhoBERT-base-v2 pretrain trên văn bản đã tách từ, cho ăn text
  thô là dùng sai mô hình. Con số 0.421 đang bị ghi nhầm nhãn. Bắt buộc chạy lại.
- **nb 02 nhẹ hơn** — tách từ chỉ là *tùy chọn* cho CNN/RNN, kết quả 0.322/0.374 vẫn
  hợp lệ, chỉ bị mô tả sai. Nhưng phải chạy lại để 4 mô hình cùng điều kiện tiền xử lý,
  nếu không thì phần "nguyên tắc so sánh công bằng" của báo cáo không đứng được.
  Phương án rẻ: giữ kết quả cũ, đổi `SEGMENT = False` cho khớp sự thật và ghi rõ trong
  báo cáo. Trung thực nhưng yếu hơn.

Code đã vá xong (commit `52b429a`): assert chặn ở cả hai notebook, `segmented` tường
minh trong summary, `save_preds()` ghi npy ngay sau từng run. Chạy lại là chạy đúng.

Khi chạy, kiểm hai dòng đầu trước khi để nó train tiếp:

```
[word_segment] backend = underthesea          <- không được là none
Tách từ: Tọa_đàm được tổ_chức tại Hà_Nội      <- phải có dấu _
```

**Lưu ý:** mọi số của TextCNN/BiLSTM/PhoBERT sẽ đổi, kể cả hai mô hình chính. Đừng viết
chương Kết quả trước khi có bộ số mới. nb 01 (EDA) và nb 04 (XLM-R) không ảnh hưởng —
nb 01 gọi `tokenize(segment=False)` tường minh, nb 04 dùng SentencePiece.

Thứ tự: push code → Kaggle Save & Run All nb 02 (~1-2h) → nb 03 (~2-3h) → tải về
`results/5`, `results/6`.

### 2. Chạy `05_analysis.ipynb` (CPU, ~2 phút)

Chưa chạy lần nào với bộ dữ liệu đầy đủ. Sinh ra:
`main_comparison.png`, `ablation.csv` + `ablation.png`, `per_class.csv`,
`accuracy_by_group.csv`, `cost_vs_accuracy.png`, `error_examples.csv`, McNemar test.
Notebook đã tự nhận `xlmr_xnli_finetuned` trong bảng ablation, không cần sửa code.

### 3. Viết báo cáo (`reports/main.tex`) — chưa động tới

- [ ] Metadata đầu file: CourseName, ReportTitle, DatasetName, tên + MSSV, ngày nộp
- [ ] Xóa bảng đóng góp thành viên (bài cá nhân)
- [ ] Chương Bộ dữ liệu — số liệu từ `outputs/logs/eda.json`
- [ ] Chương Mô hình — 4 bảng cấu hình + bảng nguồn mô hình ngoài
      (chép từ đầu `04_xlmr_external.ipynb`, điền commit hash)
- [ ] Chương Thiết kế thực nghiệm — bảng môi trường (mục 9 `KAGGLE.md`), bảng siêu tham số
- [ ] Chương Kết quả — bảng LaTeX do notebook 05 xuất + các hình
- [ ] Phân tích lỗi — mở `outputs/logs/error_examples.csv`, **tự điền cột "Nhóm lỗi"**
      (phủ định / số / kiến thức thế giới / đồng tham chiếu / nhãn sai). Không tự động hóa được.
- [ ] Ghi rõ TextCNN/BiLSTM dùng embedding **random init** (`EMB_PATH = None`), chưa nạp PhoW2V
- [ ] Tóm tắt 200–300 từ (viết sau cùng)
- [ ] Trích dẫn: ViANLI, PhoBERT, XLM-R, ESIM/TextCNN
- [ ] Rà 2 checklist trong `PLAN.md` (tái lập + tự đánh giá)

---

## Quy ước gộp kết quả từ Kaggle

Kaggle push notebook đã chạy (JSON 1 dòng, kèm output, ~200KB) thẳng lên `origin/main`.
Sau khi pull:

1. Khôi phục bản source sạch của notebook: `git checkout <commit trước> -- notebooks/XX.ipynb`
2. Tải Output của Version về, giải nén vào `results/<n>/` (thư mục này đã gitignore)
3. Chép sang `outputs/`: `logs/*.json` `logs/*.csv` `logs/predictions/*.npy` `figures/*.png`.
   **Không** chép `outputs/checkpoints/`.
4. Chạy audit trước khi commit — summary/test/npy phải khớp:
   ```python
   # so accuracy trong *_summary.json với (predictions/<run>.npy == y_true.npy).mean()
   ```
   Đã từng bắt được lỗi thật: `xlmr_large_finetuned_summary.json` còn giữ số của lần
   chạy cũ (0.455 / 4 epoch) trong khi predictions là của lần mới (0.449 / 5 epoch).
5. `*_results.csv` sinh lại từ các `*_summary.json` chứ đừng chép đè — mỗi Version
   Kaggle chỉ chứa các run của riêng nó.

---

## Đã chốt, không cần làm lại

- Label mapping: `0=entailment, 1=neutral, 2=contradiction`
- `max_length` = 128 (p95: premise 51 từ, hypothesis 28 từ)
- Split cố định 8012 / 1000 / 1000, val-test cân bằng chính xác
- 865/1000 premise của test cũng có trong train — **theo thiết kế, không phải leakage**;
  chỉ 2 cặp (premise, hypothesis) trùng train↔test
- Commit hash dataset: `0fec8d6ecb043a61c609f9b51f80401fdf1e84d3`
- CNN/RNN học thuộc lòng (train loss 0.002, val đứng ~0.33): **không phải bug**, là tư liệu
  cho §So sánh các họ kiến trúc. Viết "xấp xỉ mức ngẫu nhiên", đừng viết "thấp hơn ngẫu nhiên".
- PhoBERT LayerNorm `beta/gamma` warning: vô hại, `eval_macro_f1` đạt 0.42 ≫ 0.33.
- XLM-R Part A và Part B phải chạy ở **hai Kaggle Version riêng** (gộp một session thì OOM).

## Tùy chọn nếu còn thời gian

1. **ESIM** cho mô hình 2 — BiLSTM có co-attention. Luận điểm "cross-attention mới là
   yếu tố quyết định, không phải Transformer". Đáng làm nhất.
2. Nạp **PhoW2V** cho TextCNN/BiLSTM thay random init.
3. Ensemble PhoBERT + XLM-R bằng cộng logit.
4. PhoBERT-large.

---

**Nhắc lại:** ViANLI là bộ adversarial, kết quả tốt nhất trong paper gốc cũng chỉ quanh
45–50% (tự kiểm tra con số chính xác trước khi trích dẫn). Điểm của đồ án nằm ở quy trình
thực nghiệm chặt chẽ và phân tích lỗi có chiều sâu, không phải ở accuracy cao.
