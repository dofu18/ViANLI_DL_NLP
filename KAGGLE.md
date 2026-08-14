# Hướng dẫn chạy dự án trên Kaggle

## 0. Chuẩn bị tài khoản (làm một lần)

1. Đăng ký [kaggle.com](https://www.kaggle.com) → **Settings → Phone Verification**.
   Chưa xác minh phone thì **không bật được GPU và Internet** trong notebook.
2. Kiểm tra quota: Settings → Accelerator. Mỗi tuần có **30 giờ GPU**, reset vào thứ Bảy.

## 1. Đưa code lên Kaggle

Có hai cách. **Cách A** tiện hơn nhiều nếu bạn sửa code thường xuyên.

### Cách A — qua GitHub (khuyến nghị)

```bash
cd D:\NLP
git init
git add .
git commit -m "ViANLI: pipeline + notebooks"
git branch -M main
git remote add origin https://github.com/<user>/ViANLI_DL_NLP.git
git push -u origin main
```

Trong notebook Kaggle, cell đầu tiên:

```python
!git clone -q https://github.com/<user>/ViANLI_DL_NLP.git /kaggle/working/repo
!cp -r /kaggle/working/repo/src /kaggle/working/
!cp -r /kaggle/working/repo/configs /kaggle/working/
!ls /kaggle/working
```

Sửa code ở máy → `git push` → chạy lại cell clone. Không phải upload tay.

### Cách B — upload thành Kaggle Dataset

1. Nén thư mục `src/` và `configs/` thành `vianli-code.zip`.
2. [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset** → upload zip →
   đặt tên `vianli-code` → **Create**.
3. Trong notebook: **Add Input** → Datasets → chọn `vianli-code`.
   Code sẽ nằm ở `/kaggle/input/vianli-code/`.
4. Vì `/kaggle/input` là chỉ đọc, copy sang working:
   ```python
   !cp -r /kaggle/input/vianli-code/src /kaggle/working/
   !cp -r /kaggle/input/vianli-code/configs /kaggle/working/
   ```

Mỗi lần sửa code phải upload phiên bản mới (Dataset → New Version).

## 2. Tạo notebook

1. [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. File → **Import Notebook** → upload `notebooks/01_eda.ipynb`.
3. Panel bên phải (nếu ẩn thì bấm mũi tên góc phải):

| Mục | Giá trị |
|---|---|
| **Accelerator** | `None` cho notebook 01; `GPU T4 x2` hoặc `GPU P100` cho 02–04 |
| **Internet** | **On** (cần để tải dataset và model từ Hugging Face) |
| **Persistence** | `Files only` — giữ `/kaggle/working` giữa các session |
| **Environment** | `Always use latest` |

> Notebook 01 không cần GPU. Đừng bật GPU cho nó — phí quota.

## 3. Thứ tự chạy

```
01_eda.ipynb          CPU,  ~5-10 phút   -> data/splits/*.csv, outputs/logs/eda.json
02_cnn_rnn.ipynb      GPU,  ~1-2 giờ     -> TextCNN + BiLSTM + 4 ablation
03_phobert.ipynb      GPU,  ~2-3 giờ     -> PhoBERT + 3 ablation
04_xlmr_external.ipynb GPU, ~4-6 giờ     -> zero-shot + fine-tune XLM-R large
05_analysis.ipynb     CPU,  ~2 phút      -> bảng + hình cho báo cáo
```

Ước lượng thời gian là phỏng đoán — đo lại bằng chính log của bạn.
Tổng khoảng 8–11 giờ GPU, vừa trong quota 30h/tuần, còn dư để chạy lại khi lỗi.

**Quan trọng:** notebook 02–05 đều cần `data/splits/*.csv` do notebook 01 sinh ra.
Xem mục 5 để chuyển file giữa các notebook.

## 4. Chạy nền bằng Save Version

Session tương tác bị ngắt sau 12 giờ, hoặc sớm hơn nếu mất mạng. Với notebook 03 và 04:

1. Bấm **Save Version** (góc trên phải).
2. Chọn **Save & Run All (Commit)** — notebook chạy trên máy chủ Kaggle, bạn tắt trình duyệt được.
3. Theo dõi ở tab **Versions**; xong thì mở version đó để xem output và tải file.

Chế độ này an toàn hơn hẳn việc ngồi canh notebook chạy.

## 5. Chuyển dữ liệu giữa các notebook

`/kaggle/working` không tự chia sẻ giữa các notebook. Ba lựa chọn:

### Cách 1 — Output của notebook làm Input (gọn nhất)

Sau khi notebook 01 chạy xong bằng **Save & Run All**, ở notebook 02:
**Add Input → Notebook Output → chọn notebook 01**, rồi:

```python
!mkdir -p /kaggle/working/data/splits
!cp /kaggle/input/<slug-notebook-01>/data/splits/*.csv /kaggle/working/data/splits/
```

### Cách 2 — Tải về rồi upload thành Dataset

Tải `data/splits/*.csv` từ tab Output của notebook 01, upload thành dataset `vianli-splits`,
rồi Add Input vào các notebook sau.

### Cách 3 — Gộp tất cả vào một notebook

Nối nội dung 01–05 thành một notebook duy nhất. Đơn giản nhất về mặt dữ liệu, nhưng
một lỗi ở giữa là mất toàn bộ. Chỉ nên làm khi đã chạy trơn từng phần.

## 6. Tải kết quả về máy

Sau mỗi lần chạy, vào tab **Output** của notebook (hoặc của version) → **Download All**.
Giải nén **nguyên cục** vào `results/<số tiếp theo>/` — không cần sắp xếp hay lọc gì:

```
D:\NLP\results\1\   <- output notebook 02
D:\NLP\results\2\   <- output notebook 03
D:\NLP\results\3\   <- output notebook 04 Part A
D:\NLP\results\4\   <- output notebook 04 Part B
```

`results/` đã nằm trong `.gitignore` nên đây là vùng đệm thô, đổ vào bao nhiêu lần cũng
được. Từ đó mới chép sang `outputs/` — nơi chứa dữ liệu chính thức đi vào báo cáo:

```
outputs/logs/*.json    outputs/logs/*.csv    outputs/logs/predictions/*.npy
outputs/figures/*.png
```

**Không** chép `outputs/checkpoints/`. Hình trong `outputs/figures/` chép sang Overleaf
để chèn vào `reports/main.tex`.

Bước từ vùng đệm sang chính thức là chỗ dễ sai nhất — luôn chạy audit trước khi commit:

```python
# accuracy trong *_summary.json phải khớp (predictions/<run>.npy == y_true.npy).mean()
# và khớp accuracy trong *_test.json; y_true.npy phải giống nhau ở MỌI notebook
```

Đã bắt được lỗi thật bằng cách này: `xlmr_large_finetuned_summary.json` còn giữ số của
lần chạy cũ (0.455 / 4 epoch) trong khi `predictions/` là của lần mới (0.449 / 5 epoch).

Lưu ý: mỗi Kaggle Version chỉ chứa các run của riêng nó, nên `*_results.csv` phải **sinh
lại** từ toàn bộ `*_summary.json` chứ không chép đè.

**Đừng commit checkpoint** (`.pt`, `.safetensors`) lên GitHub — `.gitignore` đã chặn sẵn.

### Kaggle tự push notebook lên GitHub

Nếu notebook được tạo từ GitHub, Kaggle sẽ commit ngược bản **đã chạy** lên `origin/main`
(JSON một dòng ~200KB kèm toàn bộ output, và giữ nguyên giá trị biến lúc chạy — ví dụ
`PART = "B"`). Sau khi `git pull`, khôi phục lại bản source sạch:

```bash
git checkout <commit trước đó> -- notebooks/04_xlmr_external.ipynb
```

Kết quả thật lấy từ `results/<n>/` chứ không lấy từ output nhúng trong notebook.

## 7. Xử lý sự cố thường gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `ModuleNotFoundError: data` | Chưa copy `src/` sang `/kaggle/working`, hoặc `sys.path.insert` chạy trước khi copy |
| `Connection error` khi `load_dataset` | Internet đang Off — bật ở panel phải, notebook sẽ restart |
| `CUDA out of memory` | Hạ `batch_size`, tăng `grad_accum` để giữ effective batch. XLM-R large: `batch_size=4, grad_accum=8` |
| Hết dung lượng `/kaggle/working` | Giới hạn 20GB. Đã đặt `save_total_limit=1`; xóa bớt checkpoint cũ giữa các lần chạy |
| Notebook chạy quá 12h rồi bị kill | Dùng Save & Run All (mục 4), hoặc giảm số epoch/nhánh ablation |
| Hết quota GPU giữa tuần | Ưu tiên PhoBERT (notebook 03). XLM-R có thể hạ xuống `xlm-roberta-base` — **ghi rõ lý do trong báo cáo** |
| `underthesea` cài chậm/lỗi | Cài lại rồi **restart kernel**. Notebook 02 và 03 có assert chặn ngay ở cell tách từ — cứ để nó dừng, ĐỪNG chạy tiếp. Xem mục 10. |
| Kết quả khác nhau giữa các lần chạy | GPU có phần không tất định. Đã cố định seed; nếu cần chặt hơn thì chạy 3 seed và báo cáo mean ± std |

## 8. Bảo mật

- **Không** hard-code Hugging Face token hay Kaggle API key trong notebook.
  Dùng **Add-ons → Secrets**:
  ```python
  from kaggle_secrets import UserSecretsClient
  hf_token = UserSecretsClient().get_secret("HF_TOKEN")
  ```
- Notebook mặc định là Private. Chỉ chuyển sang Public khi nộp bài, và kiểm tra lại
  không có thông tin cá nhân trong output.

## 9. Ghi lại cho phần Môi trường thực nghiệm

Chạy cell này ở một notebook bất kỳ và chép kết quả vào bảng "Môi trường phần cứng và
phần mềm" trong `reports/main.tex`:

```python
import sys, torch, transformers, platform
print("Python     :", sys.version.split()[0])
print("Platform   :", platform.platform())
print("PyTorch    :", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA       :", torch.version.cuda)
print("GPU        :", torch.cuda.get_device_name(0))
print("VRAM       :", f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("Seed       : 42")
!free -g | head -2
```

## 10. Bẫy tách từ tiếng Việt — đọc trước khi chạy 02 và 03

Kaggle **không cài sẵn `underthesea`**. Lần chạy đầu tiên của dự án dính đúng bẫy này:
`word_segment()` fallback im lặng trả về văn bản gốc, nên **cả 5 run của notebook 02 và
cả 4 run của notebook 03 đều huấn luyện trên text CHƯA tách từ** mà không có dấu hiệu gì.
Hậu quả nặng nhất: ablation `phobert_no_wordseg` trùng **từng bit** với baseline
(Δ = 0.000), và PhoBERT — mô hình *bắt buộc* đầu vào đã tách từ — bị dùng sai cách.

Cách phát hiện sau khi đã chạy:

```python
# vocab của notebook 02 phải có token nối bằng "_"
v = open("outputs/checkpoints/shared_vocab.txt", encoding="utf-8").read().split("\n")
print(sum("_" in t for t in v), "/", len(v))   # bằng 0 là hỏng
```

Cách phòng, đã cài sẵn trong code:

- `word_segment(strict=True)` — mặc định — **raise** thay vì fallback. Nhánh PhoBERT dùng
  cái này. Chỉ nhánh CNN/RNN mới được `strict=False`, vì ở đó tách từ là tùy chọn thật.
- `_get_segmenter()` in ra `[word_segment] backend = underthesea` hoặc `= none` ngay lần
  gọi đầu tiên.
- Notebook 02 (cell `SEGMENT`) và notebook 03 (cell `prep`) đều có assert dò thử câu
  `"Tọa đàm được tổ chức tại Hà Nội"` và dừng nếu kết quả không chứa `_`.

Khi chạy, kiểm hai dòng đầu tiên trước khi để nó train tiếp:

```
[word_segment] backend = underthesea          <- phải là underthesea, không được là none
Tách từ: Tọa_đàm được tổ_chức tại Hà_Nội      <- phải có dấu _
```

Nếu assert dừng notebook: cài lại `underthesea`, **restart kernel**, chạy lại từ đầu.
Đừng gỡ assert. Đừng đặt `SEGMENT = False` chỉ để nó chạy qua — trừ khi m thực sự muốn
nhánh không tách từ và **ghi rõ điều đó trong báo cáo**.

Notebook 04 (XLM-R) không liên quan: nó dùng SentencePiece, `word_segment: false` trong
`configs/xlmr_external.yaml` là cố ý.
