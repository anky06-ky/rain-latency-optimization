# Rainfall Forecasting under Inference Latency Constraint

[![CI Pipeline](https://github.com/anky06-ky/rain-latency-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/anky06-ky/rain-latency-optimization/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![Data License: CC BY-NC 4.0](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Optimization: Optuna TPE](https://img.shields.io/badge/Optimization-Optuna%20TPE-orange.svg)](https://optuna.org/)

> **Course Project:** Introduction to Optimization (*Nhập môn Tối ưu hóa*)  
> **Institution:** Faculty of Information Technology, Van Lang University  
> **Topic:** Supervised Time-Series Forecasting under Hardware/Latency Constraint

---

## 1. Problem

Urban flooding triggered by short-duration, high-intensity rainfall is an acute operational challenge in metropolitan areas such as Ho Chi Minh City. Early flood alert systems require reliable next-hour precipitation forecasts. However, for real-time edge or server deployment, forecast accuracy cannot come at the expense of computational burden.

**Research Questions:**  
* **RQ1 (Fair Comparison under Equal Budget):** Under an identical evaluation budget ($N = 40$ trials/seed across 5 random seeds), does Bayesian Optimization via Optuna TPE discover feasible configurations ($P_{95} \le 5.0\text{ ms}$) with lower validation RMSE and significantly higher feasible rate than Random Search?
* **RQ2 (Budget Scaling & Saturation):** As the evaluation budget scales up to $N = 100$ trials/seed $\times$ 5 seeds = 500 trials, does the optimal solution continue to improve, and where does convergence saturation occur?
* **RQ3 (Accuracy-Latency Trade-off):** Across an expanded model capacity space ($n_{\text{estimators}} \in [20, 500]$) and historical windows $L$, how does the Pareto frontier characterize the trade-off between forecast precision and execution latency?

* **Target Variable:** Total precipitation 1 hour ahead ($y_{t+1}$ in mm) at Tan Son Nhat Station (WMO ID: `48900`).
* **Raw Data Source & GO/NO-GO Quality Gate:** 
  Dữ liệu quan trắc gốc được tải từ Meteostat giai đoạn 2021–2026 (`data/raw/weather_meteostat_2021_2026.csv`). Tuy nhiên, cổng kiểm soát chất lượng dữ liệu (**GO/NO-GO Gate**) đã loại bỏ giai đoạn 2021–2022 do trong thời gian gián đoạn dịch bệnh Covid-19, trạm 48900 xuất hiện nhiều khoảng trống mất dữ liệu vượt ngưỡng cho phép (>5% số giờ thiếu). Giai đoạn 3 năm liên tục **2023–2025** (26.304 giờ) đạt tiêu chuẩn chất lượng cao nhất (tỷ lệ thiếu lượng mưa chỉ 2.13%) và được chọn làm tập dữ liệu nghiên cứu chính thức.

---

## 2. Optimization Formulation

The core of this project is formulated as a **Constrained Hyperparameter and Feature Optimization Problem**:

$$\begin{aligned}
\min_{x = (L, \theta)} \quad & f(x) = \text{RMSE}_{\text{val}}(L, \theta) \\
\text{subject to} \quad & T_{\text{e2e}}(L, \theta) \le 5.0\text{ ms} \quad (\text{Internal Safety Margin: } T_{\text{e2e}} \le 4.5\text{ ms}) \\
& L \in \{1, 2, 3, 6, 12, 24, 36, 48, 72\} \text{ (hours)} \\
& \theta \in \Theta \text{ (LightGBM hyperparameters)}
\end{aligned}$$

### Decision Variables ($x$)
1. **Historical Window Length ($L$):** Số giờ quá khứ sử dụng để trích xuất đặc trưng trễ và thống kê trượt ($L \in \{1, 2, 3, 6, 12, 24, 36, 48, 72\}$ giờ).
2. **Model Hyperparameters ($\theta$):**
   * Tree depth (`max_depth` $\in [2, 12]$)
   * Number of boosting iterations (`n_estimators` $\in [20, 500]$ — mở rộng lên 500 cây để khảo sát giới hạn suy luận)
   * Number of leaves (`num_leaves` $\in [7, 127]$)
   * Learning rate (`learning_rate` $\in [0.01, 0.25]$, lấy mẫu theo tiên nghiệm log-uniform)
   * Minimum child samples (`min_child_samples` $\in [10, 100]$)

> **Lưu ý kỹ thuật về cấu trúc cây LightGBM:** LightGBM phát triển cây theo cơ chế *leaf-wise* (ưu tiên mở rộng lá có độ giảm tổn thất lớn nhất). Với cấu hình tối ưu `num_leaves=15`, cây chỉ có tối đa 15 lá. Do đó, `max_depth=12` đóng vai trò là **ngưỡng chặn trên (upper ceiling)** để khống chế độ sâu tối đa của các nhánh bất đối xứng, chứ không phải mô hình tạo ra cây nhị phân cân bằng $2^{12} = 4.096$ lá.

### Constraint: End-to-End Latency ($T_{\text{e2e}} \le 5.0\text{ ms}$)
To ensure the latency constraint reflects real-world deployment, latency is evaluated **end-to-end** on single-sample inference requests:
$$T_{\text{e2e}} = T_{\text{feature\_transform}}(L) + T_{\text{predict}}(\theta)$$
* **Tối ưu hóa tầng suy luận:** Sử dụng `model.booster_.predict` trực tiếp trên mảng NumPy thay vì qua wrapper Scikit-learn, loại bỏ hoàn toàn overhead kiểm tra kiểu dữ liệu và cảnh báo tên cột của Python.
* **Biên an toàn nội bộ (Safety Margin):** Ràng buộc SLA hệ thống là $5.0\text{ ms}$, nhưng quy trình tối ưu đặt mục tiêu biên an toàn nội bộ $\le 4.5\text{ ms}$ nhằm đảm bảo mô hình luôn khả thi ngay cả khi chịu ảnh hưởng bởi nhiễu điều tiết xung nhịp CPU trên máy trạm dùng chung.
* **Đo lường đa lượt (Multi-run Stability):** Mô hình champion được đo lường qua 5 phiên độc lập (mỗi phiên 100 lần lặp sau 20 lần warmup) và báo cáo giá trị trung vị (median) cùng khoảng dao động $[\min, \max]$.

### Two-Tier Experimental Budget Protocol
* **Phase 1: Equal-Budget Comparison (RQ1):** Benchmark Random Search vs Optuna TPE at **40 trials per seed** across 5 random seeds (`42, 123, 456, 789, 2026`) — total 200 evaluations per method. Optuna achieves **59.0% feasible rate** vs **39.0%** for Random Search (**+20.0 điểm phần trăm (pp)**).
* **Phase 2: Budget Scaling & Convergence Analysis (RQ2):** Scale budget up to **100 trials per seed $\times$ 5 seeds = 500 trials** per method (1,000 total evaluations). Optuna stabilizes convergence variance by $-75.8\%$ compared to the 40-trial baseline ($\sigma = 0.0029 \to 0.0007$) and by $-57.6\%$ compared to 100-trial Random Search ($\sigma = 0.0017 \to 0.0007$). Optuna achieves **40.0% feasible rate** vs **26.2%** for Random Search (**+13.8 điểm phần trăm (pp)**).
* **Selection Rule:** Chọn nghiệm có validation RMSE thấp nhất trong tập các cấu hình khả thi ($T_{\text{e2e}} \le 5.0\text{ ms}$).
* **Test Evaluation:** Đánh giá đúng 1 lần duy nhất trên tập Test (H2/2025) để đối chuẩn khách quan.

---

## 3. Method

To assess optimization efficacy, we compare two distinct approaches under identical budgets:

| Method | Role | Strategy | Constraint Handling |
| :--- | :--- | :--- | :--- |
| **Random Search** | Baseline | Uniform random sampling in search space $\Omega$ (log-uniform for lr) | Post-hoc feasible filtering during Pareto selection |
| **Optuna / TPE** | Main Method | Tree-structured Parzen Estimator (Bayesian Optimization) | Constrained sampling via `constraints_func` ($P_{95} - 5.0 \le 0$) |

### Splitting Protocol (Chronological Split by Target Timestamp)
Để phản ánh chính xác bài toán chuỗi thời gian thực tế và **triệt tiêu hoàn toàn rò rỉ dữ liệu (data leakage)**:

1. **Causal Imputation (Không dùng dữ liệu tương lai):**
   - Các biến thời tiết liên tục (`temp, dwpt, rhum, pres, wspd`) được điền khuyết bằng phương pháp **forward-fill nhân quả** (`ffill(limit=2)`). Tuyệt đối **không** sử dụng phép nội suy tuyến tính hai chiều vốn dùng giá trị tương lai để điền cho quá khứ.
   - Lượng mưa phục vụ tạo biến trễ dùng `ffill(limit=1).fillna(0.0)`. Nhãn mục tiêu $y_{t+1}$ được lấy nguyên bản từ dữ liệu gốc; mọi dòng thiếu nhãn đều bị loại bỏ chứ không tự ý điền.
2. **Loại bỏ rò rỉ tại biên bằng `target_time`:**
   - Mục tiêu dự báo là mưa tại $t+1$, nên việc phân chia được thực hiện nghiêm ngặt dựa trên thời điểm mục tiêu $\text{target\_time} = t + 1\text{h}$:
     * **Tập Huấn luyện (Train - 66.7%):** $\text{target\_time} \le \text{2024-12-31 23:59:59}$ (17.544 giờ — năm 2024 là năm nhuận 366 ngày). Mẫu cuối cùng có đặc trưng tại $22:00$ dự đoán mục tiêu tại $23:00$ (hoàn toàn nằm trong 2024).
     * **Tập Thẩm định (Validation - 16.5%):** $\text{2025-01-01 00:00:00} \le \text{target\_time} \le \text{2025-06-30 23:59:59}$ (4.344 giờ). Mẫu đầu tiên có đặc trưng tại thời điểm $2024-12-31\text{ }23:00$ (dữ liệu lịch sử đã biết) để dự đoán mục tiêu đầu tiên của năm 2025.
     * **Tập Kiểm thử (Test - 16.8%):** $\text{2025-07-01 00:00:00} \le \text{target\_time} \le \text{2025-12-31 23:59:59}$ (4.416 giờ, đỉnh mùa mưa). Chỉ được đánh giá ở bước cuối cùng.
3. **Tính Nhất quán về Số mẫu Đánh giá qua các Giá trị Cửa sổ $L$ (Sample Consistency Across $L$):**
   - Do hàm `build_features` gọi `dropna()` để loại bỏ các hàng đầu tiên chưa đủ $L$ giờ trễ lịch sử, chỉ có $L$ giờ đầu tiên của **tháng 01/2023 (đầu tập Train)** bị cắt bỏ (tập Train có 17.286 mẫu ở $L=1$ và 17.215 mẫu ở $L=72$, độ lệch chỉ 71 mẫu / 0.4%).
   - **Tập Thẩm định (Validation - 4.057 mẫu) và Tập Kiểm thử (Test - 4.399 mẫu) có chính xác 100% số lượng mẫu và tem thời gian giống hệt nhau trên MỌI giá trị $L \in \{1, 2, 3, 6, 12, 24, 36, 48, 72\}$**. Do đó, các chỉ số Val RMSE và Test RMSE giữa các $L$ được so sánh trên cùng một tập quan sát tuyệt đối công bằng.

![Chronological Split Timeline](results/figures/train_val_test_split.png)

---

## 4. Results & Multi-Model Benchmark

### 1. Phân tích Thống kê Dịch chuyển Phân phối theo Mùa (Seasonal Distribution Shift)

| Phân chia | Giai đoạn quan trắc | Số mẫu ($N$) | Mean (mm/h) | Std (mm/h) | Tỷ lệ giờ mưa ($y > 0$) | Max mưa (mm/h) | Đặc điểm khí hậu trạm Tân Sơn Nhất |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | 01/01/2023 – 31/12/2024 | 17.286 | 0.2326 | 1.0038 | 19.25% | 19.4 | 2 chu kỳ năm trọn vẹn (năm 2024 nhuận) |
| **Validation** | 01/01/2025 – 30/06/2025 | 4.057 | 0.1896 | 0.9298 | 15.23% | 16.4 | Nửa đầu năm: mùa khô và giai đoạn chuyển mùa |
| **Test** | 01/07/2025 – 31/12/2025 | 4.399 | 0.3549 | 1.3256 *(+42.6% std)* | 25.82% | 23.4 | Nửa cuối năm: cao điểm mùa mưa gió mùa Tây Nam |

> **Giải thích khoa học về khoảng cách Train–Test:** Độ lệch chuẩn của lượng mưa tập Test cao hơn tập Validation tới **+42.6%** với tỷ lệ giờ mưa tăng lên **25.82%** (gấp 1.7 lần Validation) và xuất hiện các trận mưa dông cực đoan lên tới $23.4\text{ mm/h}$. Do đó, việc Test RMSE của toàn bộ mô hình tăng lên ~1.18–1.22 mm/h so với 0.83 mm/h ở Validation là hệ quả tự nhiên của sự dịch chuyển khí hậu theo mùa, chứ không phải do mô hình mất khả năng khái quát.

### 2. Multi-Model Benchmark Comparison (Train vs. Validation vs. Test)

Bảng đối chuẩn 7 mô hình được đồng bộ từ nguồn dữ liệu duy nhất (`results/multi_model_benchmark.csv` và `results/final_test_metrics.json`), bổ sung đường cơ sở **Climatology**, chỉ số dự báo giờ có mưa (`Wet_Test_RMSE`), chỉ số cảnh báo ngập (`Rain_Event_F1`), độ trễ P95 (ms) và các chỉ số kỹ năng dự báo (**Skill Score** so với Persistence và Climatology):

| STT | Mô hình so sánh | Cửa sổ $L$ | Cấu hình / Siêu tham số $\theta$ | Train RMSE | Val RMSE | Test RMSE | Test MAE | Wet Test RMSE ($y>0$) | Rain Event $F_1$ ($\ge 0.5$mm) | Độ trễ $P_{95}$ (ms) | Ràng buộc $\le 5.0$ ms | Skill Score vs. Pers (%) | Skill Score vs. Clim (%) |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Persistence (Ngây thơ)** | $L=1$ | $\hat{y}_{t+1} = y_t$ (mưa giờ trước) | 1.1615 | 1.0699 | 1.5674 | 0.4543 | 3.0420 | 0.5570 | **0.050 ms** | Khả thi | 0.00% | -9.72% |
| 2 | **Climatology (Tháng-Giờ)** | $L=1$ | $\mathbb{E}[y \mid \text{month}, \text{hour}]$ | 0.9319 | 0.9626 | 1.4285 | 0.6228 | 2.6167 | 0.1088 | **0.080 ms** | Khả thi | +8.86% | 0.00% |
| 3 | **Linear Regression** | $L=6$ | Hồi quy tuyến tính chuẩn (OLS) | 0.9128 | 0.8468 | 1.2203 | 0.4712 | 2.3411 | 0.5112 | 1.480 ms | Khả thi | +22.14% | +14.57% |
| 4 | **Ridge Regression** | $L=6$ | Tuyến tính điều chuẩn L2 ($\alpha=1.0$) | 0.9128 | 0.8467 | 1.2203 | 0.4713 | 2.3411 | 0.5112 | 1.650 ms | Khả thi | +22.14% | +14.57% |
| 5 | **LightGBM (Mặc định)** | $L=6$ | 100 trees, `max_depth` tự do | **0.5825** *(Quá khớp)* | 0.8499 | 1.2036 | 0.4021 | 2.3136 | 0.5617 | 1.521 ms | Khả thi | +23.21% | +15.74% |
| 6 | **Random Search Best** | $L=6$ | 200 trees, depth=12, lr=0.0171 | 0.7667 | 0.8290 | 1.1891 | 0.3993 | 2.2937 | 0.5778 | **1.207 ms** | Khả thi | +24.14% | +16.76% |
| 7 | **Optuna TPE (Đề tài tối ưu)** | **$L=1$** | **150 trees, depth=12, lr=0.0215** | **0.8243** | **0.8318** | **1.1774** | **0.3998** | **2.2735** | **0.5882** | **1.660 ms** *(range [1.19, 2.45])* | **Khả thi & Biên an toàn** | **+24.88%** | **+17.58%** |

#### Phân tích Chuyên sâu về Quá trình Tối ưu & Đánh giá Tính Khái quát hóa:
1. **Quy ước Tính Validation RMSE giữa Trial Thăm dò và Retrain:**
   - Trong nhật ký thăm dò (`optimization_history_optuna.csv`), trial tốt nhất của Optuna ghi nhận Val RMSE = `0.8297` (seed 456, L=1, 150 cây, lr 0.0215). Nhật ký Random Search ghi nhận `0.8290` (seed 42, L=6, 200 cây, lr 0.0171).
   - Khi tái huấn luyện cố định trên cùng seed 42 để kiểm chứng chéo (`evaluate_final.py`), Optuna đạt Val RMSE = `0.8318` và Random Search đạt `0.8290`. Bề mặt mục tiêu Validation tương đối phẳng giữa các cấu hình hàng đầu (chênh lệch $\le 0.0028\text{ mm/h}$, $p = 0.6742 > 0.05$ — không có ý nghĩa thống kê về RMSE trên tập Val).
   - **Lợi ích cốt lõi của Optuna TPE:** Thay vì chỉ tập trung vào khác biệt biên nhỏ trên Val, Optuna TPE đã tìm ra giải pháp tối ưu kiến trúc với **$L=1$ (chỉ 10 đặc trưng)** so với $L=6$ (45 đặc trưng) của Random Search. Điều này giúp giảm 6 lần chi phí trích xuất đặc trưng, đạt **tỷ lệ khả thi cao hơn +13.8 điểm phần trăm**, và vượt trội trên tập Test mùa mưa về cả Test RMSE (`1.1774` vs `1.1891`), sai số giờ mưa dông (`2.2735` vs `2.2937`) và chỉ số cảnh báo ngập ($F_1 = 0.5882$ vs $0.5778$).
2. **Khái quát hóa & Khoảng cách Sai số (Generalization vs. Overfitting):**
   - **Hiện tượng Quá khớp ở LightGBM Mặc định:** Mô hình mặc định (Dòng 5) không khống chế độ sâu cây và tiêu chí dừng, dẫn đến Train RMSE giảm sâu xuống `0.5825 mm/h` (ghi nhớ nhiễu lịch sử), khiến Test RMSE tăng lên `1.2036 mm/h` và $F_1$ chỉ đạt `0.5617`.
   - **Độ chặt chẽ Train–Val của Optuna:** Mô hình Optuna TPE duy trì khoảng cách Train–Val rất nhỏ (`0.8243` vs `0.8318`, gap = $0.0075\text{ mm/h}$), chứng minh mô hình không bị quá khớp vào tập huấn luyện.
   - **Năng lực Dự báo Thực chất (Skill Scores):** Dù Test RMSE tăng do phân phối mùa mưa cực đoan, mô hình Optuna TPE vẫn đạt **Skill Score +24.88%** so với dự báo ngây thơ (Persistence) và **+17.58%** so với dự báo khí hậu trung bình theo tháng–giờ (Climatology), chứng minh mô hình nắm bắt được động lực biến thiên vật lý thực sự.

![Actual vs Predicted Test](results/figures/actual_vs_predicted_test.png)

### 3. Comparative Optimization Performance Across Seeds & Statistical Significance

Báo cáo thống kê trung bình $\pm$ độ lệch chuẩn trên 5 random seeds (mỗi seed 100 trials, tổng 500 lượt thử nghiệm/phương pháp):

| Tiêu chí so sánh | Baseline: Random Search | Main Method: Optuna (TPE) | Chênh lệch / Đánh giá |
| :--- | :---: | :---: | :---: |
| **Tổng số lượt đánh giá (Trials)** | 500 | 500 | Ngân sách 100 trials x 5 seeds |
| **Số nghiệm khả thi ($T_{\text{e2e}} \le 5.0\text{ ms}$)** | 131 (26.2%) | **200 (40.0%)** | **Tỷ lệ khả thi cao hơn +13.8 điểm phần trăm (pp)** |
| **Tỷ lệ khả thi ở ngân sách 40 trials (Phase 1)** | 39.0% (78/200) | **59.0% (118/200)** | **Tỷ lệ khả thi cao hơn +20.0 điểm phần trăm (pp)** |
| **Validation RMSE tốt nhất trong Trial History** | **0.8290 mm/h** (seed 42) | 0.8297 mm/h (seed 456) | Tương đương (chênh lệch $\le 0.0007$) |
| **Validation RMSE của Mô hình Retrain xác nhận** | **0.8290 mm/h** (seed 42) | 0.8318 mm/h (seed 42) | Tương đương (chênh lệch $\le 0.0028$) |
| **Validation RMSE trung bình qua 5 seeds** | **$0.8306 \pm 0.0017$** | **$0.8301 \pm 0.0007$** | **Độ ổn định cao hơn rõ rệt** |
| **Độ lệch chuẩn qua các seeds (Std)** | Std = 0.0017 | **Std = 0.0007** | **Giảm phương sai 57.6% (và -75.8% so với baseline 40 trials)** |
| **Kiểm định Paired Student's t-test (5 seeds)** | — | — | **$t = 0.4527, p = 0.6742$** (chấp nhận $H_0$) |
| **Cửa sổ tối ưu lựa chọn ($L^*$)** | $L=6$ giờ (45 đặc trưng) | **$L=1$ giờ (10 đặc trưng)** | $L=1$ tối giản đường ống trích xuất đặc trưng |
| **Độ trễ End-to-End $P_{95}$ (Trung vị)** | 1.207 ms | **1.660 ms** *(range [1.19, 2.45])* | **Thỏa mãn SLA 5.0 ms & Biên an toàn 4.5 ms** |
| **Chỉ số F1 phát hiện mưa dông trên Test** | 0.5778 | **0.5882** | Optuna cao hơn +1.04 điểm phần trăm |

*Ghi chú khoa học về Kiểm định & Cơ chế Ràng buộc:* Khi cả hai phương pháp đều được cung cấp cùng không gian tìm kiếm và tiên nghiệm lấy mẫu log-uniform (`learning_rate`), giá trị Validation RMSE trung bình qua 5 seeds đạt mức xấp xỉ nhau ($0.8306$ vs $0.8301$, $t = 0.4527, p = 0.6742$ — không có khác biệt có ý nghĩa thống kê về RMSE). Bề mặt hàm mục tiêu Validation quanh vùng tối ưu khá phẳng với nhiều cấu hình ($L=1, 36, 48$ và `max_depth` khác nhau) đều cho sai số trong khoảng 0.8297–0.8318.
Vì vậy, việc tìm ra $L=1$ là một phát hiện thực nghiệm giá trị giúp giảm nhẹ khối lượng tính toán biên cho thiết bị nhúng/edge device, kết hợp với ưu thế **tỷ lệ khả thi cao hơn (+13.8 pp)** và **khả năng tổng quát hóa tốt hơn trên mùa mưa ($F_1 = 0.5882$)**.

### 4. Visualizations & Engineering Analysis

* **Biên Pareto Khả thi (`results/figures/pareto_frontier.png`):** Thể hiện rõ sự đánh đổi giữa sai số dự báo và độ trễ. Các nghiệm có độ trễ vượt 5.0 ms bị đánh dấu vi phạm.
* **Đường cong Hội tụ (`results/figures/convergence_comparison.png`):** Optuna TPE với cơ chế lấy mẫu có điều kiện đạt giá trị RMSE trung bình ổn định hơn và độ lệch chuẩn nhỏ hơn rõ rệt qua 5 seeds.
* **Phân chia Tập dữ liệu (`results/figures/train_val_test_split.png`):** Minh họa trực quan dòng thời gian 2023–2025 không có hiện tượng rò rỉ dữ liệu.
* **Đối chiếu Thực tế vs Dự báo (`results/figures/actual_vs_predicted_test.png`):** Minh họa khả năng bám sát lượng mưa của mô hình tối ưu trên các trận mưa dông đỉnh điểm tháng 10/2025.

---

## 5. How to Run

### 1. Prerequisites & Environment
* **Python Version:** `3.10+` (khuyến nghị Python 3.10, 3.11 hoặc 3.13)
* Khởi tạo và kích hoạt môi trường ảo:
  ```bash
  python -m venv venv
  # Windows PowerShell:
  .\venv\Scripts\Activate.ps1
  # Linux/macOS:
  source venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Reproduce Experiments
1. **Tải dữ liệu khí tượng gốc (Meteostat Tân Sơn Nhất 48900):**
   ```bash
   python src/data/download.py
   ```
2. **Chạy Baseline (Random Search - 100 trials x 5 seeds = 500 trials):**
   ```bash
   python src/models/optimize_random.py --trials 100
   ```
3. **Chạy Phương pháp chính (Optuna TPE - 100 trials x 5 seeds = 500 trials):**
   ```bash
   python src/models/optimize_optuna.py --trials 100
   ```
4. **Huấn luyện mô hình độc lập & lưu artifact:**
   ```bash
   python src/models/train.py --L 1 --n-estimators 150 --max-depth 12 --learning-rate 0.0215 --min-child-samples 20
   ```
5. **Chạy Demo suy luận thời gian thực & cảnh báo ngập:**
   ```bash
   python src/demo_predict.py
   ```
6. **Đánh giá Đa mô hình & Xuất toàn bộ biểu đồ:**
   ```bash
   python src/evaluation/evaluate_final.py
   ```
7. **Kiểm thử tự động Pipeline (Unit Tests):**
   ```bash
   python -m pytest -v tests/
   ```
8. **Khởi chạy Web Dashboard Tương tác Trực quan (Interactive UI):**
   ```bash
   python src/server.py
   # Mở trình duyệt tại: http://127.0.0.1:8000
   ```

---

## 6. Limitations & Hardware Metadata

### 1. Hardware Metadata for Benchmark Reproducibility
Để đảm bảo khả năng tái lập độ trễ suy luận, các thực nghiệm đo lường được tiến hành trên môi trường phần cứng chuẩn:
* **CPU:** Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz (4 Physical Cores, 8 Logical Processors)
* **RAM:** 16.0 GB DDR4
* **Operating System:** Windows 11 Home (64-bit)
* **Runtime Stack:** Python 3.13.15, LightGBM 4.7.0 (`n_jobs=1`), Scikit-learn 1.9.1, NumPy 2.5.4, Optuna 5.0.0

### 2. Technical Nuances & Limitations
1. **Phụ thuộc vào Cách cài đặt Hàm trích xuất Đặc trưng:**
   - Trong quá trình thực nghiệm ban đầu, việc sử dụng vòng lặp truy cập hàng bằng `.iloc` của Pandas khiến thời gian trích xuất đặc trưng tăng phi tuyến theo $L$ (ở $L=72$, riêng việc lấy lag tốn hơn $6.3\text{ ms}$, lập tức vi phạm ngân sách $5.0\text{ ms}$).
   - Khi được tối ưu hóa sang dạng mảng NumPy vector hóa (`arr[::-1].ravel()`), thời gian tiền xử lý cho $L=72$ giảm xuống dưới $0.5\text{ ms}$. Điều này cho thấy rằng trong các bài toán tối ưu độ trễ thời gian thực, **tối ưu hóa luồng dữ liệu (data engineering pipeline) có vai trò quyết định tương đương với việc tinh chỉnh mô hình**.
2. **Nhiễu đo lường Độ trễ trên Phần cứng Dùng chung (CPU Jitter):**
   - Độ trễ đơn mẫu ở mức mili-giây chịu ảnh hưởng bởi hiện tượng điều tiết xung nhịp (thermal throttling) và chuyển ngữ cảnh CPU (context switching). Do đó, dự án sử dụng phân vị 95 ($P_{95}$) qua hàng trăm lượt đo để đảm bảo tính ổn định.
3. **Tính thưa thớt của Lượng mưa (Zero-Inflation) & Bài toán Cảnh báo Ngập:**
   - Dữ liệu lượng mưa theo giờ tại Tân Sơn Nhất có tới $>74\%$ số giờ không mưa ($0\text{ mm}$). Việc chỉ nhìn vào RMSE toàn cục có thể bị chi phối bởi các giờ khô hạn. Vì vậy, báo cáo đã bổ sung `Wet_Test_RMSE` ($2.2735\text{ mm/h}$) và $F_1\text{-score}$ phát hiện mưa dông ($0.5882$) nhằm phục vụ trực tiếp cho mục tiêu cảnh báo ngập lụt.
4. **Phạm vi Trạm Đơn lẻ:**
   - Dữ liệu thu thập từ trạm Tân Sơn Nhất (`48900`) chưa phản ánh tính phân hóa không gian (spatial heterogeneity) lượng mưa trên toàn địa bàn TP.HCM.

---

## Data Provenance & Licensing

* **Mã nguồn dự án (Code License):** Phát hành theo giấy phép [MIT License](LICENSE).
* **Dữ liệu thời tiết (Data Source & License):** Dữ liệu quan trắc trạm Tân Sơn Nhất (ID: `48900`) được cung cấp bởi [Meteostat](https://meteostat.net/) và tuân thủ giấy phép [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
* **Chính sách dữ liệu:** Để đảm bảo tính gọn nhẹ của repository và tuân thủ quy định môn học, **các file CSV dữ liệu thô và trung gian không được commit lên Git** (đã cấu hình trong `.gitignore`). Dữ liệu có thể được tái lập hoàn toàn tự động thông qua `python src/data/download.py`.

---

## Citation

```bibtex
@misc{meteostat2026,
  author = {Meteostat Developers},
  title = {Hourly Weather Data for Tan Son Nhat Station (48900)},
  year = {2026},
  url = {https://meteostat.net/}
}
```
