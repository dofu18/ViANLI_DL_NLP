# TO-DO — ViANLI (cập nhật 2026-08-14)

## Trạng thái: 4 mô hình chính + 8 nhánh ablation đã chạy xong

Toàn bộ kết quả Kaggle đã gộp vào `outputs/`. Đã audit chéo `*_summary.json` ↔
`*_test.json` ↔ `predictions/*.npy` cho cả 12 run — khớp 100%, `y_true.npy` giống
nhau ở mọi lần chạy nên 4 mô hình so sánh được với nhau.

### Kết quả test (1000 mẫu, sai số ±1.5 điểm)

| Run | Accuracy | Macro-F1 | Ghi chú |
|---|---|---|---|
| Majority baseline | 0.334 | — | mốc ngẫu nhiên |
| TF-IDF hypothesis-only | 0.318 | — | < majority → không có bias hypothesis-only |
| TF-IDF premise-only | 0.039 | — | dưới ngẫu nhiên **do thiết kế**, xem `PLAN.md` |
| **textcnn** | 0.334 | 0.3274 | mô hình 1 |
| **bilstm_attn** | 0.379 | 0.3659 | mô hình 2 |
| **phobert_base_v2** | **0.470** | **0.4659** | mô hình 3 — **tốt nhất** |
| **xlmr_large_finetuned** | 0.449 | 0.3721 | mô hình 4, 5 epoch, 2807s |
| xlmr_xnli_finetuned | 0.423 | 0.4190 | ablation: khởi tạo từ checkpoint XNLI |
| xlmr_zeroshot | 0.334 | 0.3246 | ablation: không fine-tune |
| bilstm_hyponly | 0.351 | 0.3405 | ablation |
| bilstm_dropout02 | 0.403 | 0.3963 | ablation — **cao hơn baseline**, xem dưới |
| bilstm_maxlen64 | 0.380 | 0.3621 | ablation |
| phobert_hyponly | 0.416 | 0.4093 | ablation |
| phobert_lr5e-5 | 0.450 | 0.4479 | ablation |
| phobert_no_wordseg | 0.421 | 0.4209 | ablation — Δ = **−0.049** so với base |

Tất cả 12 run đều dùng cùng split, cùng seed 42, cùng `max_length=128`, và (trừ nhánh
`no_wordseg` cố ý) đều tách từ bằng `underthesea`. `shared_vocab.txt`: 5247/9503 token
có dấu `_` — xác nhận tách từ chạy thật.

### Bốn điểm đáng viết vào báo cáo

1. **Tách từ đáng giá 4,9 điểm với PhoBERT.** `phobert_base_v2` 0.470 vs `no_wordseg`
   0.421. Con số 0.421 trùng khít kết quả của lần chạy hỏng trước đó — bằng chứng sạch
   rằng lần đó thực chất là chạy không tách từ. Đây là nhánh ablation mạnh nhất của đồ án.
2. **PhoBERT (135M, 0.470) và XLM-R large (560M, 0.449) — chênh lệch KHÔNG có ý nghĩa
   thống kê.** McNemar: chỉ PhoBERT đúng 213 mẫu, chỉ XLM-R đúng 192 mẫu, **p = 0.32**.
   Phải viết là "ngang nhau về accuracy" chứ **không được** viết "PhoBERT tốt hơn".
   Luận điểm đúng và vẫn rất mạnh: PhoBERT đạt cùng mức chất lượng với **1/4 số tham số,
   1/4 thời gian train (10,9 vs 46,8 phút) và 1/5 thời gian suy luận (3,02 vs 16,23
   ms/mẫu)**. Đó mới là nội dung cho §Chi phí tính toán.
   Khác biệt thật sự nằm ở macro-F1 (0.466 vs 0.372) — xem điểm 3.
3. **Accuracy một mình không đủ.** `xlmr_large_finetuned` accuracy 0.449 nhưng macro-F1
   chỉ 0.372 vì gần như bỏ hẳn lớp *contradiction* (dự đoán 465/514/**21**).
   `xlmr_xnli_finetuned` accuracy thấp hơn (0.423) nhưng dự đoán cân (346/399/255) nên
   macro-F1 cao hơn (0.419). Lý do bắt buộc phải báo cáo cả hai metric.
4. **`bilstm_dropout02` (0.403) cao hơn baseline `bilstm_attn` (0.379)** → dropout 0.5
   quá mạnh cho mô hình 5M tham số. Đáng bàn, **nhưng giữ `bilstm_attn` làm mô hình 2
   chính thức** — không được chọn nhánh tốt nhất post-hoc rồi gọi đó là baseline.

---

## Việc còn lại

### 1. ~~Chạy lại nb 02 và nb 03~~ ✅ XONG (2026-08-14 chiều)

Đã chạy lại, tách từ hoạt động (`backend = underthesea`, vocab 5247/9503 token có `_`),
kết quả đã gộp vào `outputs/`, audit 12/12 khớp. Xem bảng ở đầu file.

Một chi tiết đã sửa tay khi gộp: hai notebook chạy bằng code **trước** commit `52b429a`
nên `phobert_hyponly_summary.json` ghi `word_segment: false` (lỗi suy bằng
`enc is encoded`) và các summary của nb 02 thiếu hẳn trường này. Đã sửa thành `true` —
đúng với thực tế, vì cả hai log đều xác nhận backend là `underthesea`. Lần chạy sau sẽ
tự ghi đúng nhờ tham số `segmented`.

### 2. ~~Chạy `05_analysis.ipynb`~~ ✅ XONG — chạy local, không cần Kaggle

Đã sinh đủ: `main_comparison.png`, `ablation.csv` + `ablation.png`, `per_class.csv`,
`accuracy_by_group.csv`, `cost_vs_accuracy.png`, `error_examples.csv`, bảng LaTeX của
bảng kết quả chính, ma trận đồng thuận và McNemar test.

Chạy lại bất cứ lúc nào bằng (cần cwd là `notebooks/`):

```
jupyter nbconvert --to notebook --execute --inplace 05_analysis.ipynb
```

**Số liệu chính lấy được:**

- **Ablation xếp theo mức ảnh hưởng** (`ablation.csv`): zero-shot XNLI −0.115 | PhoBERT
  chỉ-hypothesis −0.054 | **bỏ tách từ −0.049** | BiLSTM chỉ-hypothesis −0.028 |
  khởi tạo từ XNLI −0.026 (nhưng macro-F1 **+0.047**) | lr 5e-5 −0.020 |
  max_len 64 +0.001 | dropout 0.2 **+0.024**.
- **F1 theo lớp** (`per_class.csv`): lớp *contradiction* — TextCNN 0.329, BiLSTM 0.264,
  PhoBERT 0.449, **XLM-R large 0.040**. XLM-R gần như không dự đoán lớp này.
- **21,2% mẫu (212/1000) mọi mô hình đều sai**, 78,8% có ít nhất một mô hình đúng.
  Con số 212 này là nhân của phần thảo luận về tính adversarial.
- **Đồng thuận giữa các mô hình** thấp: PhoBERT ↔ XLM-R chỉ 0.501, TextCNN ↔ XLM-R 0.382.
  Bốn mô hình sai ở những chỗ khác nhau → chúng học các tín hiệu khác nhau.
- **Accuracy theo nhóm** (`accuracy_by_group.csv`): mọi mô hình đều kém nhất ở nhóm
  overlap **cao** giữa premise và hypothesis (PhoBERT 0.431 vs 0.499 ở nhóm overlap thấp).
  Đúng đặc trưng adversarial: câu giống nhau về từ ngữ nhưng khác nhau về nghĩa.
- Test có 16,1% hypothesis chứa từ phủ định, 37,7% chứa số.
- **TextCNN 0.334 = đúng bằng majority baseline 0.334.** Ô "Mọi mô hình vượt majority"
  in ra `False` là vì vậy — không phải lỗi. Phải nói thẳng trong báo cáo: TextCNN không
  học được gì vượt mức đoán nhãn đa số.

### 3. Chạy lại nb 02 (~7 phút GPU) để lấy thời gian suy luận — tùy chọn nhưng nên làm

Bảng §Chi phí tính toán đang trống ô `inference_ms_per_sample` cho TextCNN và BiLSTM
(nb 02 chưa đo, nb 03/04 thì có). Đã thêm đoạn đo vào `run()` của nb 02, cùng cách đo
với hai notebook kia để bốn mô hình so được với nhau.

Rẻ: 5 run của nb 02 chỉ mất ~390 giây train. Và pipeline đã chứng minh là tất định
(`phobert_no_wordseg` chạy lại ra predictions trùng từng bit), nên **mọi con số khác sẽ
giữ nguyên**, chỉ thêm cột còn thiếu.

Nếu không chạy lại thì phải để dấu "—" ở hai ô đó trong báo cáo và ghi rõ lý do.

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
