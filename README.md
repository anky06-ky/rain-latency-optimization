# Rainfall Forecasting under Inference Latency Constraint

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Optimization: Optuna TPE](https://img.shields.io/badge/Optimization-Optuna%20TPE-orange.svg)](https://optuna.org/)

> **Course Project:** Introduction to Optimization (*Nhập môn Tối ưu hóa*)  
> **Institution:** Faculty of Information Technology, Van Lang University  
> **Topic:** Supervised Time-Series Forecasting under Hardware/Latency Constraint

---

## 1. Problem

Urban flooding triggered by short-duration, high-intensity rainfall is a critical challenge in metropolitan areas such as Ho Chi Minh City. Early warning systems require accurate next-hour precipitation forecasts. However, for real-time edge or server deployment, forecast accuracy cannot come at the expense of computational burden.

**Core Research Question:**  
Given historical hourly weather observations, what historical window length $L$ and model configuration $\theta$ yield the lowest forecasting error while strictly satisfying an **end-to-end inference latency constraint $\le 5.0\text{ ms}$**?

* **Target Variable:** Total precipitation 1 hour ahead ($y_{t+1}$ in mm) at Tan Son Nhat Station (WMO ID: 48900).
* **Dataset:** Meteostat hourly historical weather data (Station 48900, 2015–2025).

---

## 2. Optimization Formulation

The core of this project is formulated as a **Constrained Hyperparameter and Feature Optimization Problem**:

$$\begin{aligned}
\min_{x = (L, \theta)} \quad & f(x) = \text{RMSE}_{\text{val}}(L, \theta) \\
\text{subject to} \quad & T_{\text{e2e}}(L, \theta) \le 5.0\text{ ms} \\
& L \in \{1, 2, 3, 6, 12, 24, 36, 48, 72\} \text{ (hours)} \\
& \theta \in \Theta \text{ (Model hyperparameters)}
\end{aligned}$$

### Decision Variables ($x$)
1. **Historical Window Length ($L$):** Number of past hours utilized to compute lag features and temporal statistics ($L \in [1, 72]$).
2. **Model Hyperparameters ($\theta$):**
   * Tree depth (`max_depth` $\in [2, 12]$)
   * Number of boosting iterations (`n_estimators` $\in [10, 500]$)
   * Learning rate (`learning_rate` $\in [0.01, 0.3]$)
   * Minimum child weight / leaves configuration

### Constraint: End-to-End Latency ($T_{\text{e2e}} \le 5.0\text{ ms}$)
To ensure the latency constraint is realistically **binding** ("cắn thật"), latency is evaluated **end-to-end** on single-sample requests:
$$T_{\text{e2e}} = T_{\text{feature\_transform}}(L) + T_{\text{predict}}(\theta)$$
* **Warm-up:** 20 unmeasured warmup runs before benchmarking.
* **Measurement:** 500 repetitions using high-precision `time.perf_counter_ns()`; reported as the 95th percentile ($P_{95}$) or mean latency on CPU.
* Large windows (e.g., $L = 72$) combined with deep ensembles produce latency $> 5.0\text{ ms}$, rendering them **infeasible** and forcing a real trade-off between lookback fidelity and computation speed.

### Decision Rule & Experimental Budget
* **Budget:** **40 trials per method**, repeated across **5 random seeds** (total 200 trial evaluations per method).
* **Selection Rule:** Select the candidate configuration with the lowest validation RMSE among all feasible configurations ($T_{\text{e2e}} \le 5.0\text{ ms}$).
* **Final Test Evaluation:** Evaluated on the held-out Test set **exactly once** using the selected best configuration to strictly avoid test data leakage.

---

## 3. Method

To assess optimization efficacy, we compare two distinct approaches under identical budgets:

| Method | Role | Strategy | Constraint Handling |
| :--- | :--- | :--- | :--- |
| **Random Search** | Baseline | Uniform random sampling in search space $\Omega$ | Rejection / penalty for infeasible trials ($T_{\text{e2e}} > 5.0\text{ ms}$) |
| **Optuna / TPE** | Main Method | Tree-structured Parzen Estimator (Bayesian Optimization) | Native constrained optimization via constraint evaluation function |

### Splitting Protocol (Chronological Time-Series Split)
Để phản ánh chính xác bài toán dự báo chuỗi thời gian trong thực tế và ngăn ngừa rò rỉ dữ liệu (data leakage), dữ liệu khí tượng Tân Sơn Nhất (2023–2025) được phân chia nghiêm ngặt theo trình tự thời gian (chronological split) thành 3 tập riêng biệt:

1. **Tập Huấn luyện (Train Set - 70%):** Từ `2023-01-01` đến `2024-12-31` (17.520 giờ). Dùng để huấn luyện các mô hình dự báo.
2. **Tập Thẩm định (Validation Set - 15%):** Từ `2025-01-01` đến `2025-06-30` (4.344 giờ). Dùng làm mục tiêu đánh giá hàm $f(x)$ cho các thuật toán tìm kiếm tối ưu hóa (Random Search và Optuna TPE) và đo đạc kiểm tra ràng buộc độ trễ $T_{\text{e2e}}(x) \le 5.0\text{ ms}$.
3. **Tập Kiểm thử (Test Set - 15%):** Từ `2025-07-01` đến `2025-12-31` (4.416 giờ). Tập dữ liệu held-out độc lập, rơi đúng vào đỉnh mùa mưa bão tại TP.HCM. **Chỉ được đánh giá DUY NHẤT MỘT LẦN** sau khi đã chọn xong nghiệm tối ưu nhất trên Validation, đảm bảo không có thiên kiến lựa chọn (selection bias).

![Chronological Split Timeline](results/figures/train_val_test_split.png)

---

## 4. Results & Multi-Model Benchmark

### 1. Multi-Model Benchmark Comparison (Train vs. Validation vs. Test)

Để đánh giá mô hình tối ưu hóa bằng Optuna TPE cải thiện ra sao so với các giải pháp khác, chúng tôi thiết lập bảng đối sánh toàn diện gồm 6 mô hình trên cả 3 tập dữ liệu cùng độ trễ suy luận End-to-End ($P_{95}$):

| STT | Mô hình so sánh | Cửa sổ $L$ | Cấu hình / Siêu tham số $\theta$ | Train RMSE (mm/h) | Val RMSE (mm/h) | Test RMSE (mm/h) | Test MAE (mm/h) | End-to-End Latency $P_{95}$ (ms) | Ràng buộc $\le 5.0$ ms |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Persistence (Ngây thơ)** | $L=1$ | $\hat{y}_{t+1} = y_t$ (lấy mưa giờ trước) | 1.1615 | 1.0699 | 1.5676 | 0.4544 | **0.05 ms** | Khả thi |
| 2 | **Linear Regression** | $L=6$ | Hồi quy tuyến tính chuẩn (OLS) | 0.9128 | 0.8468 | 1.2204 | 0.4713 | 3.33 ms | Khả thi |
| 3 | **Ridge Regression** | $L=6$ | Tuyến tính có điều chuẩn L2 ($\alpha=1.0$) | 0.9128 | 0.8467 | 1.2205 | 0.4713 | 4.41 ms | Khả thi |
| 4 | **LightGBM (Mặc định)** | $L=6$ | 100 trees, `max_depth` tự do (không giới hạn) | **0.5825** *(Quá khớp)* | 0.8499 | 1.2038 | 0.4022 | 4.13 ms | Khả thi |
| 5 | **Random Search Best** | $L=2$ | 200 trees, `max_depth`=2, `learning_rate`=0.019 | 0.8582 | 0.8382 | 1.1896 | 0.4129 | 2.78 ms | Khả thi |
| 6 | **Optuna TPE (Đề tài tối ưu)** | **$L=1$** | **200 trees, depth=6, leaves=15, lr=0.0235** | **0.8111** | **0.8301** | **1.1762** | **0.3975** | **2.48 ms** | **Khả thi (Thấp nhất)** |

#### Phân tích chuyên sâu về Quá trình Tối ưu:
* **Tối ưu cái gì?** Bài toán tối ưu đồng thời biến không gian kết hợp $x = (L, \theta)$:
  1. *Độ dài cửa sổ trích xuất đặc trưng ($L$):* Nếu $L$ quá lớn ($L \ge 24$), hàm trích xuất lag/rolling tiêu tốn hơn 10–25 ms, lập tức vi phạm ngân sách thời gian thực ($T_{\text{e2e}} > 5.0\text{ ms}$). Optuna tự động phát hiện cửa sổ tối ưu $L=1$ giờ, vừa nắm bắt được tín hiệu mưa tức thời vừa giữ độ trễ tiền xử lý ở mức siêu thấp.
  2. *Siêu tham số mô hình ($\theta$):* Cân bằng giữa độ phức tạp cây (`max_depth`, `num_leaves`, `min_child_samples`) và tốc độ hội tụ (`learning_rate`, `n_estimators`).
* **Khắc phục hiện tượng Quá khớp (Overfitting):**
  - Hãy quan sát mô hình **LightGBM Mặc định (Dòng 4)**: Train RMSE đạt tới `0.5825` (rất thấp), nhưng sang Test RMSE lại vọt lên `1.2038` $\rightarrow$ Hiện tượng Overfitting rất nặng do cây phân nhánh sâu tự do ghi nhớ nhiễu.
  - Ngược lại, **Optuna TPE (Dòng 6)** kiểm soát chặt khoảng cách khái quát hóa (*generalization gap* giữa Train `0.8111` và Val `0.8301`), giúp mô hình đạt sai số thấp nhất trên tập Test (`1.1762 mm/h`, MAE `0.3975 mm/h`).
* **Độ trễ đáp ứng thời gian thực:**
  - Mô hình đạt $P_{95} = 2.48\text{ ms}$, nằm sâu trong ngưỡng cho phép ($5.0\text{ ms}$), đảm bảo khả năng triển khai trực tiếp vào các trạm quan trắc cảnh báo ngập lụt đô thị TP.HCM.

![Actual vs Predicted Test](results/figures/actual_vs_predicted_test.png)

### 2. Comparative Optimization Performance (40 Trials x 5 Seeds = 200 Evaluations / Method)

| Metric / Dimension | Baseline: Random Search | Main Method: Optuna (TPE) | Improvement |
| :--- | :---: | :---: | :---: |
| **Total Evaluations** | 200 | 200 | — |
| **Feasible Trials ($T_{\text{e2e}} \le 5.0\text{ ms}$)** | 124 (62.0%) | **146 (73.0%)** | **+11.0% feasibility** |
| **Constraint Violations ($T_{\text{e2e}} > 5.0\text{ ms}$)** | 76 (38.0%) | **54 (27.0%)** | Constrained sampling bias |
| **Best Historical Window ($L^*$)** | 2 hours | **1 hour** | Faster inference |
| **Best Model Configuration ($\theta^*$)** | depth=2, trees=200 | depth=6, trees=200, leaves=15 | Higher capacity |
| **Best Validation RMSE** | 0.8331 mm/h | **0.8297 mm/h** | **Lower forecasting error** |
| **Validation Latency ($P_{95}$)** | 1.284 ms | **1.558 ms** | Strictly $\le 5.0$ ms |

### 3. Visualizations & Engineering Analysis

* **Biên Pareto Khả thi (`results/figures/pareto_frontier.png`):** Thể hiện rõ sự đánh đổi giữa sai số dự báo và độ trễ. Các nghiệm có $L \ge 24$ bị loại bỏ hoàn toàn do vượt vách ngăn 5.0 ms.
* **Đường cong Hội tụ (`results/figures/convergence_comparison.png`):** Optuna TPE liên tục khám phá các vùng không gian nghiệm khả thi có chất lượng cao hơn và hội tụ nhanh hơn hẳn Random Search trên cả 5 seeds.
* **Phân chia Tập dữ liệu (`results/figures/train_val_test_split.png`):** Minh họa trực quan dòng thời gian 2023–2025 không có hiện tượng rò rỉ dữ liệu.
* **Đối chiếu Thực tế vs Dự báo (`results/figures/actual_vs_predicted_test.png`):** Minh họa khả năng bám sát lượng mưa của mô hình tối ưu trên các trận mưa dông đỉnh điểm tháng 10/2025.


---

## 5. How to Run

### 1. Prerequisites & Environment
* **Python Version:** `3.10+` (recommended: Python 3.10 or 3.11)
* Create and activate a virtual environment:
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
1. **Download Raw Climate Data (Meteostat):**
   ```bash
   python src/data/download.py
   ```
2. **Run Baseline (Random Search):**
   ```bash
   python src/models/optimize_random.py --trials 40 --seeds 5
   ```
3. **Run Main Method (Optuna TPE):**
   ```bash
   python src/models/optimize_optuna.py --trials 40 --seeds 5
   ```
4. **Evaluate on Final Test Set & Generate Figures:**
   ```bash
   python src/evaluation/evaluate_final.py
   ```

---

## 6. Limitations

1. **Single-Station Scope:** Observations are sourced exclusively from Station 48900 (Tan Son Nhat). Spatial rainfall variability across Ho Chi Minh City is not captured.
2. **Hardware Dependency of Latency:** End-to-end inference latency is measured on specific CPU hardware. Absolute millisecond values may vary across host architectures, though relative ordering of configurations remains stable.
3. **Rainfall Sparsity (Zero-Inflation):** Hourly precipitation contains high zero-value proportions ($>85\%$), requiring careful evaluation beyond global RMSE (e.g., monitoring wet-hour metrics).

---

## Data Provenance & Licensing

* **Data Source:** [Meteostat](https://meteostat.net/) Hourly Weather Observations (Station ID: 48900 - Ho Chi Minh / Tan Son Nhat).
* **License:** Licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
* **Data Policy:** In accordance with course guidelines and repository best practices, **raw and processed CSV files are excluded from Git version control** (specified in `.gitignore`). Datasets can be deterministically reproduced via `src/data/download.py`.

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
