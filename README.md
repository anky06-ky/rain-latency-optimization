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

### Splitting Protocol (Time-Series Split)
To preserve temporal causality without data leakage:
* **Train Set:** Chronological 70%
* **Validation Set:** Chronological 15% (used for HPO objective $f(x)$ and constraint verification)
* **Test Set:** Final chronological 15% (evaluated once on final selected model)

---

## 4. Results

*(Experiment logs, metrics tables, and Pareto plots will be populated here upon running optimization).*

### Expected Deliverables:
* **Pareto Frontier Plot:** Scatter plot of $\text{RMSE}_{\text{val}}$ vs. $T_{\text{e2e}}$ displaying feasible vs. infeasible regions separated by the $5\text{ ms}$ threshold.
* **Search Convergence:** Best-so-far validation error across 40 trials comparing Random Search vs. Optuna TPE (mean $\pm$ std across 5 seeds).
* **Final Test Performance:** RMSE, MAE, and End-to-End Latency of the selected optimal model.

```
results/
├── figures/
│   ├── pareto_frontier.png
│   └── convergence_comparison.png
├── optimization_history_random.csv
├── optimization_history_optuna.csv
└── final_test_metrics.json
```

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
