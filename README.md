# Viettel AI Race 2026 - LLM Inference Optimization Challenge (Bài 3)

Dự án này chứa mã nguồn, tài liệu hướng dẫn tối ưu hóa, tệp tin cấu hình deploy và bộ công cụ đánh giá local cho **Bài 3 - LLM Inference Optimization Challenge** thuộc cuộc thi **Viettel AI Race 2026**.

---

## 1. Tổng Quan Cuộc Thi & Mục Tiêu

Bài toán yêu cầu triển khai và tối ưu hóa hệ thống phục vụ (serving stack) bằng **vLLM** cho mô hình **LiquidAI/LFM2.5-1.2B-Instruct** trên 1 instance MiG **H200** (18GB VRAM, 3 CPU cores, 8GB RAM) sao cho đạt điểm số tối đa dựa trên:
1.  **Độ trễ thấp & Thông lượng cao (Effective Request Score - ERS):** Đo đạc dựa trên TTFT (Time-To-First-Token) và TPOT (Time-Per-Output-Token) trên một tệp lưu lượng multi-turn mô phỏng traffic production.
2.  **Độ chính xác ổn định (Accuracy Gate):** Vượt qua bài kiểm tra chất lượng bằng tập GPQA Diamond, chỉ chạy sau khi kết thúc vòng online, trên tối đa 5 submissions đội tự chọn.

Chi tiết đầy đủ và cập nhật nhất nằm ở [`docs/requirement.html`](docs/requirement.html) (cập nhật 18/07/2026) - đây là tài liệu tham chiếu chính thức.

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
viettel-bai-3/
├── README.md                  # Hướng dẫn chung toàn bộ dự án
├── docs/
│   ├── requirement.html            # Đề bài & quy định hiện hành (nguồn tham chiếu chính thức)
│   ├── PLANNING.md                 # Phân tích gap + kiến trúc refactor cho warmup/
│   ├── grading-workload-spec.json  # Thông số workload thực tế (public) dùng để sinh trace mẫu
│   └── Solution.md                 # Đã superseded - xem PLANNING.md / warmup/README.md
└── warmup/                   # Không gian làm việc của Vòng 1 (Sơ loại) - xem warmup/README.md
```

`warmup/README.md` là điểm bắt đầu vận hành thực tế: setup, benchmark local,
sweep tham số, kiểm tra accuracy gate, và quy trình nộp bài.

---

## 3. Bắt Đầu Nhanh

```bash
cd warmup
cat README.md
```

Đang chạy trên compute thuê ngoài (Vast.ai) và muốn quy trình từng bước
kèm cách đọc output/score? Xem
[`docs/RUNNING_GUIDE.md`](docs/RUNNING_GUIDE.md).

Xem thêm:
- [`docs/RUNNING_GUIDE.md`](docs/RUNNING_GUIDE.md) - hướng dẫn từng bước: chạy trên compute thuê & cách đọc output/score.
- [`warmup/docs/OPTIMIZATION_NOTES.md`](warmup/docs/OPTIMIZATION_NOTES.md) - lý do chọn từng flag tối ưu.
- [`warmup/docs/VAST_TESTING_GUIDE.md`](warmup/docs/VAST_TESTING_GUIDE.md) - hướng dẫn test trên GPU thuê qua Vast.ai, so sánh các tier GPU theo độ tin cậy (chi tiết + caveats).
- [`warmup/docs/COLAB.md`](warmup/docs/COLAB.md) - phương án test thay thế bằng Google Colab (free T4).

---

## 4. Tài Liệu Tham Khảo

*   [Đề bài & quy định hiện hành](docs/requirement.html)
*   [Kế hoạch refactor & phân tích gap](docs/PLANNING.md)
*   [Thông số workload thực tế (grading spec)](docs/grading-workload-spec.json)
