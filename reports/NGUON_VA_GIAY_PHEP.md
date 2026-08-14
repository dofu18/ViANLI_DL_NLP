# Nguồn, giấy phép và trích dẫn — soạn sẵn để chép vào `main.tex`

Tra ngày 2026-08-14 từ trang chính thức trên Hugging Face. Ngày truy cập ghi trong báo cáo
nên là ngày m thực sự tải dữ liệu/model về, không phải ngày này.

---

## 1. Bảng tóm tắt

| Đối tượng | Giấy phép | Ghi chú |
|---|---|---|
| Bộ dữ liệu `uitnlp/ViANLI` | **CC BY-NC-SA 4.0** | Phi thương mại + chia sẻ tương tự. Dùng cho đồ án học thuật là hợp lệ. |
| `vinai/phobert-base-v2` | **AGPL-3.0** | Khá chặt. Với báo cáo thì chỉ cần ghi đúng tên giấy phép. |
| `xlm-roberta-large` (FacebookAI) | **MIT** | |
| `joeddav/xlm-roberta-large-xnli` | **MIT** | |

Điểm cần nói rõ trong báo cáo: ViANLI là **NonCommercial**. Đồ án môn học không vi phạm,
nhưng phải nêu giấy phép và không được phát hành lại dữ liệu kèm sản phẩm thương mại.

---

## 2. §Nguồn và giấy phép dữ liệu — bảng ở dòng ~369 `main.tex`

Thay các `\todoText{...}` bằng:

```latex
Tên bộ dữ liệu & ViANLI --- Vietnamese Adversarial Natural Language Inference \\
Nguồn/URL & \url{https://huggingface.co/datasets/uitnlp/ViANLI} \\
Loại dữ liệu & Văn bản tiếng Việt; cặp (premise, hypothesis) \\
Số mẫu & 10.012 mẫu: train 8.012 / validation 1.000 / test 1.000 \\
Số lớp/nhãn & 3 lớp, single-label: \texttt{0=entailment}, \texttt{1=neutral}, \texttt{2=contradiction} \\
Giấy phép & CC BY-NC-SA 4.0 (Creative Commons Ghi công --- Phi thương mại --- Chia sẻ tương tự 4.0 Quốc tế) \\
Ngày truy cập & \todoText{dd/mm/yyyy --- ngày m tải về} \\
```

Thêm một dòng nữa vào bảng, không có trong mẫu nhưng nên có vì đây là mục tái lập:

```latex
Phiên bản (commit) & \texttt{0fec8d6ecb043a61c609f9b51f80401fdf1e84d3} \\
```

Đoạn văn đi kèm bảng, viết đại ý:

> Bộ dữ liệu do nhóm UIT-NLP (Trường Đại học Công nghệ Thông tin, ĐHQG-HCM) xây dựng và
> công bố, phát hành trên Hugging Face Hub dưới giấy phép CC BY-NC-SA 4.0. Đồ án sử dụng
> cho mục đích học tập, phi thương mại, đúng phạm vi giấy phép cho phép. Dữ liệu là văn
> bản báo chí tiếng Việt công khai, không chứa thông tin định danh cá nhân nhạy cảm, nên
> không cần bước ẩn danh hóa. Phiên bản sử dụng được cố định bằng commit hash để đảm bảo
> tái lập.

---

## 3. §Mô hình 4: Mô hình tham khảo bên ngoài — bảng ở dòng ~491

Bảng mẫu chỉ có chỗ cho **một** mô hình, nhưng dự án dùng hai checkpoint ngoài. Nên nhân
bảng thành hai, hoặc đổi thành bảng hai cột nội dung. Số liệu:

```latex
% Checkpoint 1 --- dùng cho fine-tune từ đầu (mô hình 4 chính)
Tên mô hình & XLM-RoBERTa large \\
Nguồn công bố & Conneau et al., \textit{Unsupervised Cross-lingual Representation
                Learning at Scale}, 2019 (arXiv:1911.02116) \\
URL & \url{https://huggingface.co/FacebookAI/xlm-roberta-large} \\
Phiên bản/commit & \texttt{c23d21b0620b635a76227c604d44e43a9f0ee389} \\
Giấy phép & MIT \\
Điều chỉnh của nhóm & Dùng trọng số pretrain nguyên bản; thay đầu ra bằng lớp phân loại
                      3 nhãn khởi tạo ngẫu nhiên; fine-tune toàn bộ trên train của ViANLI.
                      Không sửa kiến trúc hay code gốc. \\

% Checkpoint 2 --- dùng cho zero-shot và cho nhánh ablation khởi tạo-từ-XNLI
Tên mô hình & XLM-RoBERTa large XNLI (joeddav) \\
Nguồn công bố & Fine-tune cộng đồng trên Hugging Face Hub, dựa trên XLM-R large \\
URL & \url{https://huggingface.co/joeddav/xlm-roberta-large-xnli} \\
Phiên bản/commit & \texttt{b227ee8435ceadfa86dc1368a34254e2838bf242} \\
Giấy phép & MIT \\
Điều chỉnh của nhóm & (a) Chạy nguyên bản ở chế độ zero-shot, chỉ \textbf{ánh xạ lại thứ
                      tự nhãn} từ quy ước XNLI sang quy ước của ViANLI; (b) fine-tune tiếp
                      trên ViANLI cho nhánh ablation. Không sửa kiến trúc. \\
```

Lưu ý: "ánh xạ lại thứ tự nhãn" là chi tiết bắt buộc phải ghi — checkpoint XNLI dùng thứ
tự nhãn khác ViANLI, bỏ bước này thì kết quả zero-shot sai hoàn toàn. Đây đúng là loại
"thay đổi so với code gốc" mà đề bài yêu cầu khai báo.

### Mục "đảm bảo không huấn luyện trước trên test set của đề tài"

Đây là yêu cầu bắt buộc trong `requirementbox`. Lập luận viết được như sau:

> `xlm-roberta-large` là mô hình ngôn ngữ pretrain tự giám sát trên CommonCrawl 100 ngôn
> ngữ, không dùng nhãn NLI nào, nên không thể đã thấy nhãn test của ViANLI.
>
> `joeddav/xlm-roberta-large-xnli` được fine-tune trên tập MNLI (tiếng Anh) ghép với phần
> validation và test của XNLI trên 15 ngôn ngữ, trong đó có tiếng Việt. XNLI là bản dịch
> thủ công của MultiNLI --- nguồn văn bản là tiếng Anh thuộc các thể loại hư cấu, văn bản
> hành chính và hội thoại điện thoại. ViANLI là bộ dữ liệu độc lập, premise lấy từ báo chí
> tiếng Việt và hypothesis do người viết theo quy trình adversarial. Hai bộ không chung
> nguồn văn bản, nên test set của ViANLI không nằm trong dữ liệu huấn luyện của checkpoint.
>
> Bằng chứng thực nghiệm ủng hộ điều này: accuracy zero-shot của checkpoint XNLI trên test
> ViANLI chỉ đạt 0,334 --- đúng bằng mức ngẫu nhiên. Nếu checkpoint từng thấy dữ liệu này
> thì con số phải cao hơn hẳn.

Câu cuối là điểm mạnh: m không chỉ lập luận suông mà có số đo chứng minh.

---

## 4. Trích dẫn — thêm vào `thebibliography` ở dòng ~691

`main.tex` dùng `thebibliography` thủ công, **không** đọc file `.bib`, nên phải viết dạng
`\bibitem` chứ không dán BibTeX vào. Thay hai `\bibitem` đang để trống và thêm mới:

```latex
\bibitem{vianli}
T. V. Huynh, K. V. Nguyen, and N. L.-T. Nguyen,
``A New Benchmark Dataset and Mixture-of-Experts Language Models for Adversarial Natural
Language Inference in Vietnamese,''
\textit{Expert Systems with Applications}, p.~130109, 2025.
doi: 10.1016/j.eswa.2025.130109.

\bibitem{phobert}
D. Q. Nguyen and A. T. Nguyen,
``PhoBERT: Pre-trained Language Models for Vietnamese,''
in \textit{Findings of the Association for Computational Linguistics: EMNLP 2020}, 2020,
pp.~1037--1042.

\bibitem{xlmr}
A. Conneau et al.,
``Unsupervised Cross-lingual Representation Learning at Scale,''
\textit{arXiv preprint arXiv:1911.02116}, 2019.

\bibitem{textcnn}
Y. Kim, ``Convolutional Neural Networks for Sentence Classification,''
in \textit{Proceedings of EMNLP}, 2014, pp.~1746--1751.

\bibitem{esim}
Q. Chen, X. Zhu, Z. Ling, S. Wei, H. Jiang, and D. Inkpen,
``Enhanced LSTM for Natural Language Inference,''
in \textit{Proceedings of ACL}, 2017, pp.~1657--1668.
```

Hai `\bibitem` cũ tên `dataset` và `externalmodel` nên xóa hẳn, đổi mọi chỗ `\cite{dataset}`
thành `\cite{vianli}` và `\cite{externalmodel}` thành `\cite{xlmr}` cho khớp.

Bản BibTeX gốc (nếu sau này chuyển sang dùng file `.bib`):

```bibtex
@article{HUYNH2025130109,
  title   = {A New Benchmark Dataset and Mixture-of-Experts Language Models for Adversarial Natural Language Inference in Vietnamese},
  journal = {Expert Systems with Applications},
  pages   = {130109},
  year    = {2025},
  issn    = {0957-4174},
  doi     = {10.1016/j.eswa.2025.130109},
  url     = {https://www.sciencedirect.com/science/article/pii/S095741742503725X},
  author  = {Tin Van Huynh and Kiet Van Nguyen and Ngan Luu-Thuy Nguyen}
}

@inproceedings{phobert,
  title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
  author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  year      = {2020},
  pages     = {1037--1042}
}

@article{xlmr,
  author  = {Alexis Conneau and Kartikay Khandelwal and Naman Goyal and Vishrav Chaudhary
             and Guillaume Wenzek and Francisco Guzm\'{a}n and Edouard Grave and Myle Ott
             and Luke Zettlemoyer and Veselin Stoyanov},
  title   = {Unsupervised Cross-lingual Representation Learning at Scale},
  journal = {CoRR},
  volume  = {abs/1911.02116},
  year    = {2019},
  url     = {http://arxiv.org/abs/1911.02116}
}
```

---

## 5. Việc phải tự kiểm trước khi nộp

- [ ] **Đọc paper ViANLI để lấy con số SOTA chính xác.** Cả `TODO.md` lẫn `PLAN.md` đang
      ghi "quanh 45--50%" theo trí nhớ, chưa ai xác minh. Không được trích con số này khi
      chưa mở paper. Link: https://www.sciencedirect.com/science/article/pii/S095741742503725X
- [ ] Điền **ngày truy cập** thật vào bảng dữ liệu.
- [ ] Nếu bài nộp đi kèm mã nguồn công khai: cân nhắc thêm mục giấy phép cho chính repo,
      và lưu ý AGPL-3.0 của PhoBERT nếu có ý định phát hành lại trọng số (chỉ fine-tune
      rồi báo cáo kết quả thì không phát sinh nghĩa vụ gì).
- [ ] Trích dẫn thêm thư viện chính nếu giảng viên yêu cầu: PyTorch, Hugging Face
      Transformers, underthesea, scikit-learn.
