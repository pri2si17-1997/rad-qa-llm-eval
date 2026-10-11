"""
Condition-level significance tests across models: for every pair of conditions
(k=0,1,2,3), a Wilcoxon signed-rank test and a sign test on the 13 model means,
for token F1 and RadGraph F1.

Input: results/qlevel_eval_main.csv (scripts/run_question_level_eval.py).
Output: results/shot_level_wilcoxon.csv
"""

from pathlib import Path

import pandas as pd
from scipy import stats
from itertools import combinations

RESULTS = str(Path(__file__).resolve().parent.parent / "results")

# ── Shot-level Wilcoxon for Token F1 and RadGraph F1 ─────────────────────────

df = pd.read_csv(f"{RESULTS}/qlevel_eval_main.csv")
test = df[df["subset"] == "test"].copy()

rows = []
for metric, col in [("token_f1", "f1"), ("radgraph_f1", "radgraph_f1")]:
    if col not in test.columns:
        print(f"[skip] {col} not in qlevel_eval_main.csv (run without --no-radgraph)")
        continue
    pivot = test.pivot(index="model", columns="shot", values=col)
    for a, b in combinations([0, 1, 2, 3], 2):
        d = pivot[b] - pivot[a]
        n_up = (d > 0).sum()
        n_dn = (d < 0).sum()
        stat, p_wilc = stats.wilcoxon(pivot[b], pivot[a])
        p_sign = (
            stats.binomtest(min(n_up, n_dn), n_up + n_dn, 0.5).pvalue
            if (n_up + n_dn) > 0
            else 1.0
        )
        rows.append(
            {
                "metric": metric,
                "shot_a": a,
                "shot_b": b,
                "mean_a": round(pivot[a].mean(), 4),
                "mean_b": round(pivot[b].mean(), 4),
                "delta": round(pivot[b].mean() - pivot[a].mean(), 4),
                "n_improved": int(n_up),
                "n_declined": int(n_dn),
                "wilcoxon_stat": round(stat, 1),
                "wilcoxon_p": round(p_wilc, 4),
                "sign_p": round(p_sign, 4),
                "significant_05": p_wilc < 0.05,
            }
        )

shot_df = pd.DataFrame(rows)
shot_df.to_csv(f"{RESULTS}/shot_level_wilcoxon.csv", index=False)
print("=== Shot-level Wilcoxon (Token F1) ===")
print(shot_df[shot_df["metric"] == "token_f1"].to_string(index=False))
