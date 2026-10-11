"""
Statistical analyses for RadQA, on question-level scores (614 questions; one
prediction per question scored against all annotator golds — see
src/eval/question_level.py).

  1. Error category breakdown  — 7 mutually exclusive categories per model × shot
  2. Taxonomy validation       — RadGraph F1 on genuine_failure / paraphrase questions

All use the same 13-model set and results-{k}shot/ directory structure.

Usage (from project root):
    source radqa_venv/bin/activate
    python scripts/run_statistical_analyses.py                   # error (taxonomy only on request: it can load RadGraph)
    python scripts/run_statistical_analyses.py --analysis error
    python scripts/run_statistical_analyses.py --analysis taxonomy
    python scripts/run_statistical_analyses.py --shots 0 1 2 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.question_level import load_gold, load_question_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. ERROR CATEGORY BREAKDOWN
# ---------------------------------------------------------------------------

CATEGORIES = [
    "correct_em", "paraphrase", "partial", "genuine_failure",
    "correct_abstain", "abstention_error", "hallucination",
]

CATEGORY_LABELS = {
    "correct_em":       "Correct (EM=1)",
    "paraphrase":       "Paraphrase (F1≥0.5)",
    "partial":          "Partial (0.1≤F1<0.5)",
    "genuine_failure":  "Genuine Failure (F1<0.1)",
    "correct_abstain":  "Correct Abstain",
    "abstention_error": "Abstention Error",
    "hallucination":    "Hallucination",
}


def classify_questions(df: pd.DataFrame) -> pd.Series:
    """Category per question from question-level EM/F1 and answerability."""
    gold, pred = df["gold_impossible"].astype(bool), df["pred_impossible"].astype(bool)
    cat = np.select(
        [gold & pred, gold & ~pred, ~gold & pred,
         df["em"] >= 1.0, df["f1"] >= 0.5, df["f1"] >= 0.1],
        ["correct_abstain", "hallucination", "abstention_error",
         "correct_em", "paraphrase", "partial"],
        default="genuine_failure",
    )
    return pd.Series(cat, index=df.index)


def run_error_breakdown(shots: List[int], subset: str):
    print("\n" + "="*70)
    print("1. ERROR CATEGORY BREAKDOWN")
    print("="*70)

    per_q = load_question_table(shots, subset)
    per_q["category"] = classify_questions(per_q)

    all_rows = []
    for k in shots:
        print(f"\n  k={k}")
        for disp, mdf in per_q[per_q["shot"] == k].groupby("model"):
            cats  = mdf["category"].value_counts()
            total = len(mdf)

            row = {"shot": k, "model": disp, "n_total": total}
            for cat in CATEGORIES:
                cnt = int(cats.get(cat, 0))
                row[cat]            = cnt
                row[f"{cat}_pct"]   = round(100 * cnt / total, 2)
            all_rows.append(row)

            # inline print
            print(
                f"    {disp:<28}"
                f"  EM={row['correct_em_pct']:5.1f}%"
                f"  Para={row['paraphrase_pct']:5.1f}%"
                f"  Part={row['partial_pct']:5.1f}%"
                f"  GenFail={row['genuine_failure_pct']:5.1f}%"
                f"  Abstain={row['correct_abstain_pct']:5.1f}%"
                f"  OverAbst={row['abstention_error_pct']:5.1f}%"
                f"  Halluc={row['hallucination_pct']:5.1f}%"
            )

    df = pd.DataFrame(all_rows)
    out = OUTPUT_DIR / f"error_category_breakdown_{subset}.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out} ({len(df)} rows)")

    # Summary: mean across models at k=0
    k0 = df[df["shot"]==0]
    if not k0.empty:
        print(f"\n  Mean across all models at k=0:")
        for cat in CATEGORIES:
            print(f"    {CATEGORY_LABELS[cat]:<35} {k0[f'{cat}_pct'].mean():.1f}%")

    # Shot-level stability (mean pct across models)
    print(f"\n  Key categories by shot level (all-model mean %):")
    shot_mean = df.groupby("shot")[[f"{c}_pct" for c in CATEGORIES]].mean().round(1)
    shot_mean.columns = [CATEGORY_LABELS[c.replace("_pct","")] for c in shot_mean.columns]
    print(shot_mean.to_string())

    return df


# ---------------------------------------------------------------------------
# 2. TAXONOMY VALIDATION
# ---------------------------------------------------------------------------

def run_taxonomy_validation(shots: List[int] = None, subset: str = "test"):
    """Validate error taxonomy: RadGraph F1 on genuine_failure and paraphrase
    questions to quantify clinical contamination in lexical categories.

    genuine_failure (Token F1<0.1) may contain clinically correct extractions
    (zero token overlap but same clinical meaning). RadGraph F1 > 0 on these
    questions reveals the contamination rate. RadGraph F1 is max over annotator
    golds vs the joined prediction (same as run_question_level_eval.py); it is
    reused from results/qlevel_per_question.csv when present.

    Runs on k=0 only by default. Questions with an empty prediction are skipped.
    """
    from src.eval.question_level import pred_list
    from src.eval.radqa_metrics_enhanced import RadGraphScorer

    if shots is None:
        shots = [0]

    print("\n" + "=" * 70)
    print("2. TAXONOMY RADGRAPH VALIDATION")
    print("=" * 70)

    per_q = load_question_table(shots, subset)
    per_q["category"] = classify_questions(per_q)
    per_q = per_q[per_q["category"].isin(["genuine_failure", "paraphrase"])].copy()
    per_q["pred_list"] = per_q["pred"].apply(lambda x: pred_list(json.loads(x)))
    per_q = per_q[per_q["pred_list"].str.len() > 0]
    golds = load_gold(subset).set_index("question_id")["gold_answers"]
    per_q["gold_answers"] = per_q["question_id"].map(golds)

    if "radgraph_f1" not in per_q.columns or per_q["radgraph_f1"].isna().any():
        print("  Computing RadGraph F1 (loads a transformer model)...")
        scorer = RadGraphScorer()
        per_q["radgraph_f1"] = [
            max(scorer.score(g, " ".join(pl)) for g in gl)
            for gl, pl in zip(per_q["gold_answers"], per_q["pred_list"])
        ]
    else:
        print("  Using per-question RadGraph F1 from results/qlevel_per_question.csv")

    for (k, disp), g in per_q.groupby(["shot", "model"]):
        vc = g["category"].value_counts()
        print(f"  {disp:<28} k={k}: genuine_failure={vc.get('genuine_failure', 0):3d}  "
              f"paraphrase={vc.get('paraphrase', 0):4d}")

    if per_q.empty:
        print("  No questions found.")
        return pd.DataFrame()

    all_rows = pd.DataFrame({
        "shot": per_q["shot"], "model": per_q["model"], "question_id": per_q["question_id"],
        "category": per_q["category"], "token_f1": per_q["f1"].round(4),
        "radgraph_f1": per_q["radgraph_f1"].round(4),
        "gold": per_q["gold_answers"].apply(lambda gl: " || ".join(gl)[:200]),
        "pred": per_q["pred_list"].apply(lambda pl: " ".join(pl)[:200]),
    })
    df = pd.DataFrame(all_rows)

    out_path = OUTPUT_DIR / f"taxonomy_radgraph_validation_{subset}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Per-question results saved to {out_path}")

    # Summary by category
    print("\n  === Summary by category (pooled across all models × shots) ===")
    for cat in ["genuine_failure", "paraphrase"]:
        sub = df[df["category"] == cat].dropna(subset=["radgraph_f1"])
        if sub.empty:
            continue
        n = len(sub)
        mean_token = sub["token_f1"].mean()
        mean_rg = sub["radgraph_f1"].mean()
        pct_rg_gt0 = (sub["radgraph_f1"] > 0).mean() * 100
        pct_rg_gt02 = (sub["radgraph_f1"] > 0.2).mean() * 100
        print(f"\n  {cat.upper()} (n={n}):")
        print(f"    Mean Token F1:    {mean_token:.4f}")
        print(f"    Mean RadGraph F1: {mean_rg:.4f}")
        print(f"    RadGraph F1 > 0:  {pct_rg_gt0:.1f}% of questions  ← clinical contamination")
        print(f"    RadGraph F1 > 0.2:{pct_rg_gt02:.1f}% of questions  ← substantial clinical overlap")

    # Per-model summary for genuine_failure
    print("\n  === Genuine failure RadGraph breakdown per model (k=0) ===")
    gf = df[(df["category"] == "genuine_failure") & (df["shot"] == 0)].dropna(subset=["radgraph_f1"])
    if not gf.empty:
        per_model = gf.groupby("model").agg(
            n=("radgraph_f1", "count"),
            mean_rg=("radgraph_f1", "mean"),
            pct_rg_gt0=("radgraph_f1", lambda x: (x > 0).mean() * 100),
        ).round(3)
        print(per_model.to_string())

    # Aggregate summary CSV
    summary_rows = []
    for (cat, shot), grp in df.groupby(["category", "shot"]):
        grp_valid = grp.dropna(subset=["radgraph_f1"])
        summary_rows.append({
            "category": cat,
            "shot": shot,
            "n_questions": len(grp_valid),
            "mean_token_f1": round(grp_valid["token_f1"].mean(), 4),
            "mean_radgraph_f1": round(grp_valid["radgraph_f1"].mean(), 4),
            "radgraph_gt0_pct": round((grp_valid["radgraph_f1"] > 0).mean() * 100, 1),
            "radgraph_gt02_pct": round((grp_valid["radgraph_f1"] > 0.2).mean() * 100, 1),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / f"taxonomy_radgraph_validation_summary_{subset}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved to {summary_path}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error-category analyses for RadQA.")
    parser.add_argument("--analysis", choices=["error", "taxonomy", "all"], default="all")
    parser.add_argument("--shots",    nargs="+", type=int, default=[0,1,2,3])
    parser.add_argument("--subset",   default="test", choices=["test","dev"])
    args = parser.parse_args()

    if args.analysis in ["error", "all"]:
        run_error_breakdown(args.shots, args.subset)

    if args.analysis == "taxonomy":
        run_taxonomy_validation(shots=[0], subset=args.subset)
