"""
Question-level RadQA evaluation (primary evaluation unit, SQuAD v2 protocol).

The RadQA test split has 614 questions; 280 answerable questions carry two
annotator answers (``_MG`` / ``_SS``). ``get_flatten_radqa_data`` flattens at
the answer level, so model output files have 894 rows: each of those 280
questions appears twice, once per annotator answer.

This script scores one prediction per question against ALL gold answers for
that question (max over golds, then mean over questions), following SQuAD v2
(Rajpurkar et al., 2016; 2018):
  - Gold answers come from the raw split file (data/radqa/1.0.0/{subset}.json).
  - Prediction = the first row for the question_id in the model output file
    (the duplicate row is a second call on an identical input).
  - Answerability rule: answerable/unanswerable mismatch -> 0; both
    unanswerable -> 1; else best match over gold x pred.
  Scoring lives in src/eval/question_level.py (shared by all scripts/).

Outputs (saved to results/):
  qlevel_per_question.csv    — one row per model × shot × question
  qlevel_eval_main.csv       — one row per model × shot (F1/EM/RadGraph + CIs, answerability)
  qlevel_cluster_wilcoxon.csv — per-model shot comparisons, report-level Wilcoxon, BH within comparison
  qlevel_mcnemar.csv         — pairwise model McNemar (EM, F1>=0.5, answerability), BH across all tests

Usage (from project root):
    source radqa_venv/bin/activate
    python scripts/run_question_level_eval.py
    python scripts/run_question_level_eval.py --no-radgraph
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.question_level import (
    CI_SEED,
    MODEL_DISPLAY,
    N_BOOT,
    PROJECT_ROOT,
    bootstrap_ci,
    evaluate_model,
    load_gold,
)
from src.eval.radqa_metrics_enhanced import (
    RadGraphScorer,
    answerability_confusion,
    answerability_metrics,
    mcnemar_pairwise,
)

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

CLUSTER_N_BOOT = 5000
TOKEN_FIELDS = ["input_tokens", "output_tokens", "total_tokens"]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    summary = {"n_questions": len(df)}
    metrics = ["f1", "em"] + (["radgraph_f1"] if "radgraph_f1" in df.columns else [])
    for m in metrics:
        mean, lo, hi = bootstrap_ci(df[m].tolist(), N_BOOT, CI_SEED)
        summary.update({m: round(mean, 4), f"{m}_ci_lo": round(lo, 4), f"{m}_ci_hi": round(hi, 4)})
    confusion = answerability_confusion(df["gold_impossible"], df["pred_impossible"])
    summary.update(confusion)
    summary.update({k: round(v, 4) for k, v in answerability_metrics(confusion).items()})
    ans = df[~df["gold_impossible"]]
    summary["answerable_f1"] = round(ans["f1"].mean(), 4)
    for col in TOKEN_FIELDS:
        if col in df.columns and df[col].notna().any():
            summary[f"avg_{col}"] = round(df[col].mean(), 1)
    return summary


def run_cluster_wilcoxon(per_q: pd.DataFrame) -> pd.DataFrame:
    """Per model, compare shots on report-level mean F1 (Wilcoxon, BH within each shot pair)."""
    rows = []
    shots = sorted(per_q["shot"].unique())
    for model, mdf in per_q.groupby("model"):
        # round away float noise so exact ties are ties (CSV vs in-memory)
        rep = mdf.groupby(["shot", "report_id"])["f1"].mean().round(10).unstack(0)
        for a, b in combinations(shots, 2):
            if a not in rep or b not in rep:
                continue
            pair = rep[[a, b]].dropna()
            va, vb = pair[a].values, pair[b].values
            diff = vb - va
            if (diff != 0).sum() < 10:
                stat_w, p_w = np.nan, np.nan
            else:
                stat_w, p_w = stats.wilcoxon(vb, va)
            rng = np.random.default_rng(CI_SEED)
            idx = rng.integers(0, len(diff), size=(CLUSTER_N_BOOT, len(diff)))
            boot = diff[idx].mean(axis=1)
            rows.append({
                "model": model, "shot_a": a, "shot_b": b, "n_reports": len(diff),
                "mean_a": round(va.mean(), 4), "mean_b": round(vb.mean(), 4),
                "delta": round(diff.mean(), 4),
                "delta_ci_lo": round(np.percentile(boot, 2.5), 4),
                "delta_ci_hi": round(np.percentile(boot, 97.5), 4),
                "wilcoxon_stat": stat_w, "wilcoxon_p": p_w,
            })
    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    for _, grp in out.groupby(["shot_a", "shot_b"]):
        idx = grp.index[grp["wilcoxon_p"].notna()]
        if len(idx):
            out.loc[idx, "q_value"] = multipletests(out.loc[idx, "wilcoxon_p"], method="fdr_bh")[1]
    out["significant_q05"] = out["q_value"] < 0.05
    return out


def run_mcnemar(per_q: pd.DataFrame) -> pd.DataFrame:
    per_q = per_q.assign(
        em_correct=per_q["em"] >= 1.0,
        f1_correct=per_q["f1"] >= 0.5,
        ans_correct=per_q["gold_impossible"] == per_q["pred_impossible"],
    )
    rows = []
    for k, sdf in per_q.groupby("shot"):
        wide = {m: g.set_index("question_id") for m, g in sdf.groupby("model")}
        for ma, mb in combinations(sorted(wide), 2):
            a, b = wide[ma], wide[mb].loc[wide[ma].index]
            row = {"shot": k, "model_a": ma, "model_b": mb}
            for metric in ["em", "f1", "ans"]:
                res = mcnemar_pairwise(a[f"{metric}_correct"], b[f"{metric}_correct"])
                row[f"{metric}_stat"] = res["statistic"]
                row[f"{metric}_p"] = res["p_value"]
            rows.append(row)
    out = pd.DataFrame(rows)
    for metric in ["em", "f1", "ans"]:
        out[f"{metric}_q_all"] = multipletests(out[f"{metric}_p"], method="fdr_bh")[1]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(shots: List[int], subset: str, use_radgraph: bool):
    gold = load_gold(subset)
    print(f"Gold: {len(gold)} questions ({gold['gold_impossible'].sum()} unanswerable, "
          f"{(gold['n_gold'] == 2).sum()} with two annotator answers)")
    radgraph = RadGraphScorer() if use_radgraph else None

    per_q_frames, main_rows = [], []
    for k in shots:
        results_dir = PROJECT_ROOT / f"results-{k}shot"
        if not results_dir.exists():
            print(f"[k={k}] {results_dir} not found — skipping")
            continue
        for model_dir in sorted(d for d in results_dir.iterdir() if d.is_dir()):
            json_path = model_dir / f"{subset}.json"
            if model_dir.name not in MODEL_DISPLAY or not json_path.exists():
                continue
            model = MODEL_DISPLAY[model_dir.name]
            df = evaluate_model(gold, json_path, radgraph, keep_fields=TOKEN_FIELDS)
            df.insert(0, "shot", k)
            df.insert(0, "model", model)
            per_q_frames.append(df)

            summary = summarize(df)
            main_rows.append({"shot": k, "model_dir": model_dir.name, "model": model,
                              "subset": subset, **summary})
            print(f"  k={k}  {model:<20}  F1={summary['f1']:.4f} "
                  f"[{summary['f1_ci_lo']:.3f},{summary['f1_ci_hi']:.3f}]  "
                  f"risky={summary['risky_answer_rate']:.3f}", flush=True)

    per_q = pd.concat(per_q_frames, ignore_index=True).drop(columns=TOKEN_FIELDS)
    outputs = {
        "qlevel_per_question.csv": per_q,
        "qlevel_eval_main.csv": pd.DataFrame(main_rows),
        "qlevel_cluster_wilcoxon.csv": run_cluster_wilcoxon(per_q),
        "qlevel_mcnemar.csv": run_mcnemar(per_q),
    }
    print("\nSaved:")
    for name, df in outputs.items():
        df.to_csv(OUTPUT_DIR / name, index=False)
        print(f"  results/{name}  ({len(df)} rows)")

    main_df = outputs["qlevel_eval_main.csv"]
    cols = [c for c in ["model", "f1", "f1_ci_lo", "f1_ci_hi", "em", "radgraph_f1",
                        "risky_answer_rate", "over_abstention_rate"] if c in main_df.columns]
    print("\n--- Question-level F1 by shot ---")
    print(main_df.pivot(index="model", columns="shot", values="f1")
          .sort_values(shots[0], ascending=False).to_string())
    print("\n--- Zero-shot detail ---")
    print(main_df[main_df["shot"] == shots[0]].sort_values("f1", ascending=False)[cols]
          .to_string(index=False))

    cw = outputs["qlevel_cluster_wilcoxon.csv"]
    print("\n--- Significant shot comparisons (report-level Wilcoxon, BH q<0.05) ---")
    print(cw[cw["significant_q05"]][["model", "shot_a", "shot_b", "delta", "q_value"]]
          .to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Question-level RadQA eval (SQuAD v2 protocol).")
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--subset", default="test", choices=["test", "dev"])
    parser.add_argument("--no-radgraph", action="store_true", help="Skip RadGraph F1")
    args = parser.parse_args()
    main(shots=args.shots, subset=args.subset, use_radgraph=not args.no_radgraph)
