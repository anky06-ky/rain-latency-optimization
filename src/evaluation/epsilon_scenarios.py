"""
Epsilon-Constraint Scenario Analysis (decision per latency budget).

The project problem is an epsilon-constraint formulation of the Accuracy-Latency trade-off:
    min Val RMSE(x)  subject to  P95 latency(x) <= eps,   with eps = 5.0 ms.
This module re-solves it for several deployment budgets eps using only the trials already logged in
results/optimization_history_*.csv. No model is retrained and the Test set is never touched, so the
table is a pure re-reading of the search history and can be regenerated in seconds.

Caveat: the logged P95 values come from the search phase (scikit-learn wrapper, commit 1da4ab6) and sit
above the deployed booster_.predict path (champion P95 ~0.68 ms). An empty scenario therefore means
"no logged trial met eps under search-phase measurement", not that eps is unreachable in deployment.

Output: results/epsilon_scenarios.csv (one row per eps x method).
"""

from pathlib import Path
import sys
from typing import List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.latency import LATENCY_CONSTRAINT_MS, SAFETY_MARGIN_MS


RESULTS_DIR = Path("results")
RANDOM_CSV = RESULTS_DIR / "optimization_history_random.csv"
OPTUNA_CSV = RESULTS_DIR / "optimization_history_optuna.csv"
SCENARIOS_CSV = RESULTS_DIR / "epsilon_scenarios.csv"

EPSILON_SCENARIOS_MS = [1.0, 2.0, 3.0, SAFETY_MARGIN_MS, LATENCY_CONSTRAINT_MS]
CONFIG_COLS = ["L", "n_estimators", "max_depth", "num_leaves", "learning_rate", "min_child_samples"]
# Nullable ints: empty scenarios would otherwise turn these into floats (789.0) in the CSV
INT_COLS = ["seed", "trial", "L", "n_estimators", "max_depth", "num_leaves", "min_child_samples"]


def feasible_mask(df: pd.DataFrame, eps_ms: float) -> pd.Series:
    """
    Trials meeting P95 <= eps. At the official threshold the logged is_feasible flag is used: it was
    computed on the unrounded P95 the optimizer actually saw, whereas the 3-decimal p95_latency_ms column
    misclassifies trials sitting exactly at 5.000 ms.
    """
    if eps_ms == LATENCY_CONSTRAINT_MS:
        return df["is_feasible"].astype(bool)
    return df["p95_latency_ms"] <= eps_ms


def compute_epsilon_scenarios(
    df_random: pd.DataFrame,
    df_optuna: pd.DataFrame,
    eps_list: List[float] = EPSILON_SCENARIOS_MS,
) -> pd.DataFrame:
    """
    For each budget eps and each method, report the feasible count/rate and the configuration that the
    project's selection rule (lowest Val RMSE among feasible trials) would pick. Rows with no feasible
    trial keep the counts and leave the configuration columns empty.
    """
    rows = []
    for eps in eps_list:
        for method, df in (("RandomSearch", df_random), ("OptunaTPE", df_optuna)):
            feas = df[feasible_mask(df, eps)]
            row = {
                "epsilon_ms": eps,
                "method": method,
                "feasible_count": int(len(feas)),
                "total_trials": int(len(df)),
                "feasible_rate_pct": round(len(feas) / len(df) * 100, 1),
            }
            if not feas.empty:
                best = feas.loc[feas["val_rmse"].idxmin()]
                row.update({
                    "seed": int(best["seed"]),
                    "trial": int(best["trial"]),
                    **{c: best[c] for c in CONFIG_COLS},
                    "val_rmse": float(best["val_rmse"]),
                    "p95_latency_ms": float(best["p95_latency_ms"]),
                })
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.astype({c: "Int64" for c in INT_COLS if c in df.columns})


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not RANDOM_CSV.exists() or not OPTUNA_CSV.exists():
        raise FileNotFoundError("Missing optimization logs. Please run optimize_random.py and optimize_optuna.py first.")

    scenarios = compute_epsilon_scenarios(pd.read_csv(RANDOM_CSV), pd.read_csv(OPTUNA_CSV))
    scenarios.to_csv(SCENARIOS_CSV, index=False)
    print(scenarios.to_string(index=False))
    print(f"\n[+] Saved epsilon-constraint scenarios to {SCENARIOS_CSV}")
