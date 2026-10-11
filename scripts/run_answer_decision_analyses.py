"""
Answer-decision analyses for LLM radiology report QA on RadQA. Everything is
computed from stored model outputs; no model is called.

Unit: question level (614 test questions; first prediction per question, max over
annotator golds; see src/eval/question_level.py).

Conditions (cumulative demonstration design, src/few_shot.py):
  k=0 : zero-shot
  k=1 : +F      (answerable Findings demo)
  k=2 : +F+I    (+ answerable Impression demo)
  k=3 : +F+I+U  (+ unanswerable demo)

Analyses:
  decompose   Split ΔF1 between conditions into an extraction term and an answerability
              term (token F1 and RadGraph F1)
  hygiene     McNemar significance counts (from qlevel_mcnemar.csv) and error-category
              threshold sensitivity
  itemsafety  Within-model predictors of false answers (report x model GEE), controlling
              report difficulty
  gaterobust  Consensus gate: keep an answer only if at least t other models answer.
              All 156 ordered model pairs, and threshold/partner chosen on half the reports
              and evaluated on the other half (cross-fitting)
  accfar      Cross-model FAR correlation with SQuAD / answerable-only / extraction-only F1
  clusterci   Report-clustered bootstrap CIs for model-level F1, FAR, OAR, EM, RadGraph F1,
              answerable F1
  farpaired   Paired FAR tests: Cochran's Q, McNemar and report-level sign-flip vs lowest FAR
  selectivity Prior F1 of newly abstained vs still-answered questions (pooled and per model,
              token F1 and RadGraph F1)
  human       LLM vs inter-annotator agreement on the 280 two-annotator answerable questions
              (token F1, EM, RadGraph F1; single-annotator references; loads RadGraph)
  spanjoin    Decomposition under token F1 on the joined prediction (RadGraph's span
              convention) -> delta_decomposition_tokenjoined.csv

Outputs: results/*.csv (file names are given in each analysis).

Usage:
    radqa_venv/bin/python scripts/run_answer_decision_analyses.py                 # all
    radqa_venv/bin/python scripts/run_answer_decision_analyses.py --only decompose gaterobust
"""

import argparse
import json
import sys
import warnings
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.question_level import (  # noqa: E402
    MODEL_DISPLAY,
    load_gold,
    load_question_table,
    pred_list,
)
from src.eval.radqa_metrics_enhanced import compute_f1  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
SUBSET = "test"
SHOTS = [0, 1, 2, 3]
COND = {0: "k=0", 1: "+F", 2: "+F+I", 3: "+F+I+U"}
N_BOOT = 5000
SEED = 42

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def bh(pvals):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    if mask.any():
        q[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return q


def cluster_boot(df: pd.DataFrame, stat_fn, cluster="report_id",
                 n_boot=N_BOOT, seed=SEED):
    """Percentile CI for stat_fn(df) under report-level cluster bootstrap."""
    groups = {k: g.index.values for k, g in df.groupby(cluster)}
    keys = np.array(list(groups.keys()))
    rng = np.random.default_rng(seed)
    point = stat_fn(df)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in sample])
        boots.append(stat_fn(df.loc[idx].reset_index(drop=True)))
    boots = np.asarray(boots)
    return point, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def gee(formula, df, family="gaussian", groups="report_id"):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    fam = sm.families.Gaussian() if family == "gaussian" else sm.families.Binomial()
    model = smf.gee(formula, groups, df, family=fam,
                    cov_struct=sm.cov_struct.Exchangeable())
    return model.fit()


def load_questions() -> pd.DataFrame:
    """Question-level table (13 models x 4 conditions x 614 questions)."""
    q = load_question_table(SHOTS, SUBSET)
    return q.rename(columns={"gold_impossible": "gold_imp", "pred_impossible": "pred_imp"})


# ---------------------------------------------------------------------------
# Decomposition of ΔF1 between conditions
# ---------------------------------------------------------------------------

def _decomp_stats(a: pd.DataFrame, b: pd.DataFrame, metric: str = "f1") -> dict:
    """a, b aligned on question order. Contributions sum to total Δmetric (mean over all questions)."""
    n = len(a)
    d = b[metric].values - a[metric].values
    ga = a["gold_imp"].values
    pa, pb = a["pred_imp"].values, b["pred_imp"].values
    ans = ~ga
    both_ans = ans & ~pa & ~pb
    new_abst = ans & ~pa & pb
    new_ansd = ans & pa & ~pb
    unans = ga
    return {
        "total_delta": d.sum() / n,
        "contrib_extraction": d[both_ans].sum() / n,
        "contrib_new_abstention": d[new_abst].sum() / n,
        "contrib_new_answered": d[new_ansd].sum() / n,
        "contrib_unanswerable": d[unans].sum() / n,
        "answerability_shift": (d[new_abst].sum() + d[new_ansd].sum() + d[unans].sum()) / n,
        "n_both_answered": int(both_ans.sum()),
        "n_new_abstention": int(new_abst.sum()),
        "n_new_answered": int(new_ansd.sum()),
        "mean_f1_change_both_answered": float(d[both_ans].mean()) if both_ans.any() else np.nan,
        "n_new_hallucination": int((unans & pa & ~pb).sum()),
        "n_new_correct_abstain": int((unans & ~pa & pb).sum()),
    }


METRIC_SUFFIX = {"f1": "", "radgraph_f1": "_radgraph"}


def run_decompose(rows: pd.DataFrame):
    """Token F1 -> delta_decomposition.csv; RadGraph F1 -> delta_decomposition_radgraph.csv.
    Column names say f1 in both files; in the RadGraph file they refer to RadGraph F1."""
    for metric in METRIC_SUFFIX:
        _run_decompose_metric(rows, metric)


def _run_decompose_metric(rows: pd.DataFrame, metric: str, suffix: Optional[str] = None):
    pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (0, 3)]
    out = []
    for a_k, b_k in pairs:
        for model in list(MODEL_DISPLAY.values()) + ["POOLED"]:
            sub = rows if model == "POOLED" else rows[rows["model"] == model]
            a = sub[sub["shot"] == a_k].sort_values(["model", "question_id"]).reset_index(drop=True)
            b = sub[sub["shot"] == b_k].sort_values(["model", "question_id"]).reset_index(drop=True)
            assert (a["question_id"].values == b["question_id"].values).all()
            s = _decomp_stats(a, b, metric)
            paired = a[["report_id"]].copy()
            paired["d"] = b[metric].values - a[metric].values
            paired["both_ans"] = (~a["gold_imp"] & ~a["pred_imp"] & ~b["pred_imp"]).values
            paired["ans"] = (~a["gold_imp"]).values
            n_all = len(paired)

            def extraction(df):
                return df.loc[df["both_ans"], "d"].sum() / len(df)

            def shift(df):
                return df.loc[~df["both_ans"], "d"].sum() / len(df)

            def total(df):
                return df["d"].mean()

            _, t_lo, t_hi = cluster_boot(paired, total, n_boot=2000)
            _, e_lo, e_hi = cluster_boot(paired, extraction, n_boot=2000)
            _, s_lo, s_hi = cluster_boot(paired, shift, n_boot=2000)
            s.update({"total_ci_lo": t_lo, "total_ci_hi": t_hi,
                      "extraction_ci_lo": e_lo, "extraction_ci_hi": e_hi,
                      "shift_ci_lo": s_lo, "shift_ci_hi": s_hi, "n_questions": n_all})
            # Paired test on extraction change (both-answered questions, report means)
            ba = paired[paired["both_ans"]].groupby("report_id")["d"].mean()
            s["extraction_wilcoxon_p"] = (float(stats.wilcoxon(ba.values).pvalue)
                                          if (ba != 0).sum() >= 10 else np.nan)
            out.append({"from": COND[a_k], "to": COND[b_k], "shot_a": a_k, "shot_b": b_k,
                        "model": model, **s})
    df = pd.DataFrame(out)
    df["extraction_q"] = np.nan
    for _, g in df[df["model"] != "POOLED"].groupby(["shot_a", "shot_b"]):
        df.loc[g.index, "extraction_q"] = bh(g["extraction_wilcoxon_p"].values)
    df = df.round(6)  # 6 dp so tables can round half-up to 3–4 dp without double rounding
    suffix = METRIC_SUFFIX[metric] if suffix is None else suffix
    df.to_csv(RESULTS / f"delta_decomposition{suffix}.csv", index=False)
    print(f"\n=== Δ{metric} decomposition (POOLED) ===")
    print(df[df["model"] == "POOLED"][["from", "to", "total_delta", "contrib_extraction",
                                       "answerability_shift", "contrib_new_abstention",
                                       "contrib_unanswerable", "n_new_abstention",
                                       "mean_f1_change_both_answered"]].to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# McNemar counts and error-threshold sensitivity
# ---------------------------------------------------------------------------

def run_hygiene(rows: pd.DataFrame):
    # (a) McNemar significance counts; BH across all 312 tests is in qlevel_mcnemar.csv
    mc = pd.read_csv(RESULTS / "qlevel_mcnemar.csv")
    summary = []
    for metric in ["f1", "em", "ans"]:
        for k in SHOTS:
            s = mc[mc["shot"] == k]
            summary.append({"metric": metric, "shot": k, "n_pairs": len(s),
                            "sig_raw_p05": int((s[f"{metric}_p"] < 0.05).sum()),
                            "sig_bh_q05": int((s[f"{metric}_q_all"] < 0.05).sum())})
    pd.DataFrame(summary).to_csv(RESULTS / "mcnemar_fdr_summary.csv", index=False)
    print("\n=== McNemar BH-FDR (all 312 tests per metric) ===")
    print(pd.DataFrame(summary).to_string(index=False))

    # (b) Error-category threshold sensitivity (answerable, both-answered questions)
    thr = []
    ans = rows[~rows["gold_imp"] & ~rows["pred_imp"]]
    for lo in [0.05, 0.1, 0.2]:
        for hi in [0.4, 0.5, 0.6]:
            for k in SHOTS:
                s = ans[ans["shot"] == k]
                n_all = len(rows[rows["shot"] == k])
                gf = (s["f1"] < lo).sum() / n_all
                para = ((s["em"] == 0) & (s["f1"] >= hi)).sum() / n_all
                partial = ((s["f1"] >= lo) & (s["f1"] < hi)).sum() / n_all
                thr.append({"genuine_fail_thr": lo, "paraphrase_thr": hi, "shot": k,
                            "genuine_failure_pct": 100 * gf, "paraphrase_pct": 100 * para,
                            "partial_pct": 100 * partial})
    thr = pd.DataFrame(thr)
    thr.round(3).to_csv(RESULTS / "error_threshold_sensitivity.csv", index=False)
    piv = thr.pivot_table(index=["genuine_fail_thr", "paraphrase_thr"], columns="shot",
                          values="genuine_failure_pct")
    print("\n=== Genuine-failure % by threshold (rows) x shot (cols) ===")
    print(piv.round(2).to_string())
    return mc, thr


# ---------------------------------------------------------------------------
# Within-model predictors of false answers
# ---------------------------------------------------------------------------

def run_itemsafety(rows: pd.DataFrame):
    out = []
    for k in SHOTS:
        s = rows[rows["shot"] == k]
        acc = (s[~s["gold_imp"]].groupby(["model", "report_id"])["f1"].mean()
               .rename("acc").reset_index())
        # Report difficulty = leave-one-model-out mean answerable F1 on the report
        tot = s[~s["gold_imp"]].groupby("report_id")["f1"].agg(["sum", "count"])
        per = s[~s["gold_imp"]].groupby(["model", "report_id"])["f1"].agg(["sum", "count"]).reset_index()
        per = per.merge(tot, left_on="report_id", right_index=True, suffixes=("", "_all"))
        per["difficulty"] = (per["sum_all"] - per["sum"]) / (per["count_all"] - per["count"])
        acc = acc.merge(per[["model", "report_id", "difficulty"]], on=["model", "report_id"])
        # Separate extraction quality (F1 on questions the model answered) from answering
        # propensity (fraction of answerable questions it answered) on the same report.
        answered = s[~s["gold_imp"] & ~s["pred_imp"]]
        ext = answered.groupby(["model", "report_id"])["f1"].mean().rename("ext").reset_index()
        prop = (s[~s["gold_imp"]].assign(ansd=lambda x: (~x["pred_imp"]).astype(float))
                .groupby(["model", "report_id"])["ansd"].mean().rename("prop").reset_index())
        acc = acc.merge(ext, on=["model", "report_id"], how="left").merge(
            prop, on=["model", "report_id"], how="left")
        for c in ["acc", "ext", "prop"]:
            acc[f"{c}_within"] = acc[c] - acc.groupby("model")[c].transform("mean")

        un = s[s["gold_imp"]].copy()
        un["halluc"] = (~un["pred_imp"]).astype(int)
        d = un.merge(acc, on=["model", "report_id"], how="inner")
        for c in ["acc", "ext", "prop"]:
            w = d[f"{c}_within"]
            d[f"{c}_z"] = (w - w.mean()) / w.std()
        d["diff_z"] = (d["difficulty"] - d["difficulty"].mean()) / d["difficulty"].std()
        d3 = d.dropna(subset=["ext_z"])

        m1 = gee("halluc ~ acc_z + C(model)", d, family="binomial")
        m2 = gee("halluc ~ acc_z + diff_z + C(model)", d, family="binomial")
        m3 = gee("halluc ~ ext_z + prop_z + diff_z + C(model)", d3, family="binomial")
        for name, m, terms in [("within_model", m1, ["acc_z"]),
                               ("within_model_plus_difficulty", m2, ["acc_z", "diff_z"]),
                               ("extraction_vs_propensity", m3, ["ext_z", "prop_z", "diff_z"])]:
            for t in terms:
                out.append({"shot": k, "condition": COND[k], "spec": name, "term": t,
                            "log_odds_per_sd": m.params[t], "odds_ratio": np.exp(m.params[t]),
                            "or_ci_lo": np.exp(m.params[t] - 1.96 * m.bse[t]),
                            "or_ci_hi": np.exp(m.params[t] + 1.96 * m.bse[t]),
                            "p": m.pvalues[t], "n_rows": int(m.nobs),
                            "n_reports": d["report_id"].nunique()})
    out = pd.DataFrame(out)
    out["q_across_shots"] = np.nan
    for _, g in out.groupby(["spec", "term"]):
        out.loc[g.index, "q_across_shots"] = bh(g["p"].values)
    out = out.round(4)
    out.to_csv(RESULTS / "item_level_accuracy_safety.csv", index=False)
    print("\n=== Item-level accuracy->hallucination (GEE logit, cluster=report) ===")
    print(out[["condition", "spec", "term", "odds_ratio", "or_ci_lo", "or_ci_hi", "p",
               "n_rows", "n_reports"]].to_string(index=False))
    return out


def _gated_f1(own_ans, keep, f1, gold_imp):
    """Model's own F1 where it abstained or the gate kept its answer; where the gate
    overrides an answer to abstain, score 1 if gold is unanswerable else 0."""
    return np.where(own_ans & ~keep, gold_imp.astype(float), f1)


# ---------------------------------------------------------------------------
# Consensus gate: all ordered pairs and a cross-fitted choice of threshold / partner
# ---------------------------------------------------------------------------

N_SPLITS = 200


def _gate_arrays(s: pd.DataFrame, models):
    imp = s.pivot(index="question_id", columns="model", values="pred_imp").astype(bool)[models]
    f1 = s.pivot(index="question_id", columns="model", values="f1")[models]
    meta = s[s["model"] == models[0]].set_index("question_id").loc[imp.index]
    return (imp.values, f1.values, meta["gold_imp"].values.astype(bool),
            meta["report_id"].astype(str).values)


def _gate_metrics(own_ans, keep, f1, gold):
    """risky rate, over-abstention, gated F1 (vectors over questions)."""
    return ((keep & gold).sum() / max(gold.sum(), 1),
            ((~keep) & ~gold).sum() / max((~gold).sum(), 1),
            _gated_f1(own_ans, keep, f1, gold).mean())


def run_gate_robust(rows: pd.DataFrame):
    models = list(MODEL_DISPLAY.values())
    M = len(models)
    pairs, xfit = [], []
    for k in SHOTS:
        imp, f1, gold, rep = _gate_arrays(rows[rows["shot"] == k], models)
        ans = ~imp
        votes = ans.sum(axis=1)
        # per-question gated F1 for every (model, t) and every (owner, partner)
        gf_t = np.empty((M, 13, len(gold)))
        for i in range(M):
            loo = votes - ans[:, i]
            for t in range(13):
                keep = ans[:, i] & (loo >= t)
                gf_t[i, t] = _gated_f1(ans[:, i], keep, f1[:, i], gold)
        gf_p = np.full((M, M, len(gold)), np.nan)
        for i in range(M):
            base_r, base_o, base_f = _gate_metrics(ans[:, i], ans[:, i], f1[:, i], gold)
            for j in range(M):
                if i == j:
                    continue
                keep = ans[:, i] & ans[:, j]
                gf_p[i, j] = _gated_f1(ans[:, i], keep, f1[:, i], gold)
                r, o, g = _gate_metrics(ans[:, i], keep, f1[:, i], gold)
                pairs.append({"shot": k, "condition": COND[k], "owner": models[i],
                              "partner": models[j], "risky_rate": r, "over_abstention": o,
                              "f1": g, "d_risky": r - base_r, "d_f1": g - base_f})

        # cross-fitting: choose on half the reports, evaluate on the other half
        reports = np.unique(rep)
        rng = np.random.default_rng(SEED)
        for s_i in range(N_SPLITS):
            fit_rep = rng.choice(reports, size=len(reports) // 2, replace=False)
            fit = np.isin(rep, fit_rep)
            ev = ~fit
            g_ev = gold[ev]
            pooled_t = int(np.argmax(gf_t[:, :, fit].mean(axis=(0, 2))))
            for i in range(M):
                base = gf_t[i, 0, ev].mean()
                a_ev = ans[ev, i]
                base_risky = (a_ev & g_ev).sum() / g_ev.sum()
                base_over = ((~a_ev) & ~g_ev).sum() / (~g_ev).sum()
                t_i = int(np.argmax(gf_t[i][:, fit].mean(axis=1)))
                with warnings.catch_warnings():  # row i (self-pair) is all-NaN
                    warnings.simplefilter("ignore", RuntimeWarning)
                    p_scores = np.nanmean(gf_p[i][:, fit], axis=1)
                p_scores[i] = -np.inf
                j = int(np.argmax(p_scores))
                loo_ev = votes[ev] - a_ev
                out = {"shot": k, "condition": COND[k], "split": s_i, "model": models[i],
                       "t_model": t_i, "t_pooled": pooled_t, "partner": models[j],
                       "base_f1": base, "base_risky": base_risky}
                for name, keep, gf in [("t_model", a_ev & (loo_ev >= t_i), gf_t[i, t_i, ev]),
                                       ("t_pooled", a_ev & (loo_ev >= pooled_t), gf_t[i, pooled_t, ev]),
                                       ("pair", a_ev & ans[ev, j], gf_p[i, j, ev])]:
                    out[f"{name}_d_f1"] = gf.mean() - base
                    out[f"{name}_d_risky"] = (keep & g_ev).sum() / g_ev.sum() - base_risky
                    out[f"{name}_d_over"] = ((~keep) & ~g_ev).sum() / (~g_ev).sum() - base_over
                    # held-out counts (evaluation half only)
                    vetoed = a_ev & ~keep
                    out[f"{name}_n_halluc_removed"] = int((vetoed & g_ev).sum())
                    out[f"{name}_n_answerable_lost"] = int((vetoed & ~g_ev).sum())
                xfit.append(out)

    pairs = pd.DataFrame(pairs)
    xfit = pd.DataFrame(xfit)
    pairs.round(4).to_csv(RESULTS / "gate_pairs.csv", index=False)

    # summary: per condition, mean over splits of the model-mean held-out change
    summ = []
    for k in SHOTS:
        x = xfit[xfit["shot"] == k]
        per_split = x.groupby("split")[[c for c in x.columns if c.endswith(("_d_f1", "_d_risky", "_d_over"))]].mean()
        # counts are summed over models within a split (one held-out half), then averaged over splits
        counts = x.groupby("split")[[c for c in x.columns if "_n_" in c]].sum().mean()
        p = pairs[pairs["shot"] == k]
        fm = p[(p["owner"] == "Gemini 3 Flash") & (p["partner"] == "LLaMA 4 Maverick")].iloc[0]
        row = {"shot": k, "condition": COND[k],
               "pairs_median_d_f1": p["d_f1"].median(), "pairs_iqr_d_f1": f"{p['d_f1'].quantile(.25):.4f}..{p['d_f1'].quantile(.75):.4f}",
               "pairs_median_d_risky": p["d_risky"].median(),
               "pairs_frac_d_f1_pos": (p["d_f1"] > 0).mean(),
               "flash_maverick_d_f1_pctile": (p["d_f1"] < fm["d_f1"]).mean(),
               "pooled_t_mode": int(x["t_pooled"].mode().iloc[0])}
        for c in per_split.columns:
            row[f"xfit_{c}"] = per_split[c].mean()
            row[f"xfit_{c}_lo"] = per_split[c].quantile(0.025)
            row[f"xfit_{c}_hi"] = per_split[c].quantile(0.975)
        for c, v in counts.items():
            row[f"xfit_{c}_per_half"] = v
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.round(6).to_csv(RESULTS / "gate_crossfit.csv", index=False)
    permodel = (xfit.groupby(["shot", "condition", "model"])
                [["t_model_d_f1", "t_model_d_risky", "t_model_d_over", "pair_d_f1", "pair_d_risky"]]
                .mean().reset_index())
    # how often each model's own chosen threshold is t=0 (no gate), and the modal t
    tchoice = xfit.groupby(["shot", "model"])["t_model"].agg(
        t_model_mode=lambda t: int(t.mode().iloc[0]), frac_t0=lambda t: (t == 0).mean()).reset_index()
    permodel = permodel.merge(tchoice, on=["shot", "model"])
    permodel.round(6).to_csv(RESULTS / "gate_crossfit_permodel.csv", index=False)
    print("\n=== Gate robustness (cross-fitted, held-out halves; model-mean change) ===")
    print(summ.round(4).T.to_string())
    return summ


# ---------------------------------------------------------------------------
# Cross-model F1 vs FAR without the mechanical coupling
# ---------------------------------------------------------------------------

def _model_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Per model x shot: SQuAD F1, answerable-only F1, extraction-only F1, FAR, OAR."""
    g = rows.groupby(["model", "shot"])
    ans = rows[~rows["gold_imp"]]
    out = pd.DataFrame({
        "squad_f1": g["f1"].mean(),
        "answerable_f1": ans.groupby(["model", "shot"])["f1"].mean(),
        "extraction_f1": ans[~ans["pred_imp"]].groupby(["model", "shot"])["f1"].mean(),
        "far": rows[rows["gold_imp"]].assign(a=lambda x: (~x["pred_imp"]).astype(float))
                   .groupby(["model", "shot"])["a"].mean(),
        "oar": ans.assign(a=lambda x: x["pred_imp"].astype(float)).groupby(["model", "shot"])["a"].mean(),
    })
    return out.reset_index()


def _boot_spearman(x, y, n_boot=10_000, seed=SEED):
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        if np.ptp(x[idx]) == 0 or np.ptp(y[idx]) == 0:
            continue
        rhos.append(stats.spearmanr(x[idx], y[idx])[0])
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def run_accfar(rows: pd.DataFrame):
    """SQuAD v2 F1 credits every correct abstention with 1, so F1 = (sum answerable F1 +
    n_unans * (1 - FAR)) / n and falls mechanically as FAR rises. Correlate FAR with
    accuracy measures that exclude the unanswerable questions."""
    summ = _model_summary(rows)
    summ.round(4).to_csv(RESULTS / "model_summary.csv", index=False)
    out = []
    for k in SHOTS:
        s = summ[summ["shot"] == k].reset_index(drop=True)
        for measure, y in [("squad_f1", "far"), ("answerable_f1", "far"),
                           ("extraction_f1", "far"), ("oar", "far")]:
            x, f = s[measure].values, s[y].values
            rho, p = stats.spearmanr(x, f)
            r, pr = stats.pearsonr(x, f)
            lo, hi = _boot_spearman(x, f)
            loo_p = [stats.spearmanr(np.delete(x, i), np.delete(f, i))[1] for i in range(len(x))]
            out.append({"shot": k, "condition": COND[k], "x": measure, "y": y,
                        "spearman_rho": rho, "spearman_p": p, "rho_ci_lo": lo, "rho_ci_hi": hi,
                        "pearson_r": r, "pearson_p": pr,
                        "loo_p_min": min(loo_p), "loo_p_max": max(loo_p),
                        "loo_n_sig": int(sum(pp < 0.05 for pp in loo_p)), "n_models": len(x)})
    out = pd.DataFrame(out).round(4)
    out.to_csv(RESULTS / "accuracy_far_correlation.csv", index=False)
    print("\n=== Cross-model correlation with FAR (Spearman) ===")
    print(out.pivot_table(index="x", columns="condition", values="spearman_rho", sort=False).to_string())
    print(out.pivot_table(index="x", columns="condition", values="spearman_p", sort=False).to_string())
    return out


# ---------------------------------------------------------------------------
# Report-clustered CIs for model-level F1 / FAR / OAR / EM / RadGraph F1 / answerable F1
# ---------------------------------------------------------------------------

def _cluster_ci_ratio(num, den, n_boot=10_000, seed=SEED):
    """Percentile CI for sum(num)/sum(den) resampling reports (num/den are per-report sums)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def run_clusterci(rows: pd.DataFrame):
    out = []
    for (model, k), s in rows.groupby(["model", "shot"]):
        rec = {"model": model, "shot": k, "condition": COND[k]}
        s = s.assign(one=1.0, ans=(~s["pred_imp"]).astype(float), abst=s["pred_imp"].astype(float))
        for name, sub, val in [("f1", s, "f1"), ("far", s[s["gold_imp"]], "ans"),
                               ("oar", s[~s["gold_imp"]], "abst"), ("em", s, "em"),
                               ("radgraph_f1", s, "radgraph_f1"),
                               ("answerable_f1", s[~s["gold_imp"]], "f1")]:
            per = sub.groupby("report_id")[[val, "one"]].sum()
            rec[name] = per[val].sum() / per["one"].sum()
            rec[f"{name}_ci_lo"], rec[f"{name}_ci_hi"] = _cluster_ci_ratio(per[val].values, per["one"].values)
        out.append(rec)
    out = pd.DataFrame(out).round(6)
    out.to_csv(RESULTS / "clustered_ci.csv", index=False)
    print("\n=== Report-clustered 95% CIs, k=0 ===")
    print(out[out["shot"] == 0].sort_values("f1", ascending=False).to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Paired FAR comparisons (same 154 unanswerable questions)
# ---------------------------------------------------------------------------

def run_farpaired(rows: pd.DataFrame, n_perm=10_000):
    """Omnibus Cochran's Q across models, then each model vs the lowest-FAR model:
    exact McNemar (paired, question-level) and a report-level sign-flip permutation
    test (paired and clustered)."""
    from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
    un = rows[rows["gold_imp"]]
    models = list(MODEL_DISPLAY.values())
    out, omni = [], []
    for k in SHOTS:
        s = un[un["shot"] == k]
        A = s.pivot(index="question_id", columns="model", values="pred_imp")[models].astype(bool)
        A = (~A).astype(int)  # 1 = false answer
        rep = s[s["model"] == models[0]].set_index("question_id")["report_id"].loc[A.index].values
        q = cochrans_q(A.values)
        omni.append({"shot": k, "condition": COND[k], "cochran_q": q.statistic, "df": q.df, "p": q.pvalue})
        far = A.mean()
        ref = far.idxmin()
        rng = np.random.default_rng(SEED)
        for m in models:
            if m == ref:
                continue
            a, b = A[m].values, A[ref].values
            n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
            p_mc = mcnemar([[0, n01], [n10, 0]], exact=True).pvalue
            d = pd.Series(a - b).groupby(rep).sum().values
            obs = d.sum()
            flips = rng.choice([-1, 1], size=(n_perm, len(d)))
            p_perm = float((np.abs((flips * d).sum(axis=1)) >= abs(obs)).mean())
            out.append({"shot": k, "condition": COND[k], "reference": ref, "model": m,
                        "ref_far": far[ref], "far": far[m], "discordant_model_only": n10,
                        "discordant_ref_only": n01, "mcnemar_p": p_mc, "cluster_perm_p": p_perm})
    out = pd.DataFrame(out)
    for p_col, q_col in [("mcnemar_p", "mcnemar_q"), ("cluster_perm_p", "cluster_perm_q")]:
        out[q_col] = np.nan
        for _, g in out.groupby("shot"):
            out.loc[g.index, q_col] = bh(g[p_col].values)
    out.round(4).to_csv(RESULTS / "far_paired.csv", index=False)
    pd.DataFrame(omni).round(6).to_csv(RESULTS / "far_cochran.csv", index=False)
    print("\n=== FAR: Cochran's Q across 13 models ===")
    print(pd.DataFrame(omni).round(5).to_string(index=False))
    print("\n=== FAR vs lowest-FAR model: significant (q<0.05) ===")
    for k, g in out.groupby("shot"):
        print(f"  {COND[k]} ref={g['reference'].iloc[0]}: McNemar {g[g['mcnemar_q'] < .05]['model'].tolist()}"
              f" | cluster perm {g[g['cluster_perm_q'] < .05]['model'].tolist()}")
    return out


# ---------------------------------------------------------------------------
# Are new abstentions selective? (prior F1 of questions a model stops answering)
# ---------------------------------------------------------------------------

MIN_NEW_ABST = 5  # per-model GEE only when a model has at least this many new abstentions


def run_selectivity(rows: pd.DataFrame):
    """Among answerable questions answered in condition A, compare A-condition F1 of those
    abstained on in B vs those still answered in B. A selective threshold shift would drop
    low-F1 answers first. GEE clustered by report: pooled f1_A ~ new_abstention + model FE;
    per model f1_A ~ new_abstention (BH over models within each transition); leave-one-model-out
    range of the pooled estimate (loo_* columns, POOLED rows only).
    Token F1 -> abstention_selectivity.csv; RadGraph F1 -> abstention_selectivity_radgraph.csv
    (column names say f1 in both)."""
    for metric in METRIC_SUFFIX:
        _run_selectivity_metric(rows, metric)


def _run_selectivity_metric(rows: pd.DataFrame, metric: str):
    out = []
    for a_k, b_k in [(0, 1), (1, 2), (2, 3), (0, 2), (0, 3)]:
        A = rows[rows["shot"] == a_k].set_index(["model", "question_id"])
        B = rows[rows["shot"] == b_k].set_index(["model", "question_id"]).loc[A.index]
        d = A[~A["gold_imp"] & ~A["pred_imp"]][["report_id", metric]].copy()
        d = d.rename(columns={metric: "f1"})
        d["new_abst"] = B.loc[d.index, "pred_imp"].astype(int).values
        d = d.reset_index()
        for model in list(MODEL_DISPLAY.values()) + ["POOLED"]:
            s = d if model == "POOLED" else d[d["model"] == model]
            na, kept = s[s["new_abst"] == 1], s[s["new_abst"] == 0]
            rec = {"from": COND[a_k], "to": COND[b_k], "model": model,
                   "n_new_abstention": len(na), "n_still_answered": len(kept),
                   "prior_f1_new_abstention": na["f1"].mean() if len(na) else np.nan,
                   "prior_f1_still_answered": kept["f1"].mean(),
                   "frac_new_abst_prior_f1_ge_0.5": (na["f1"] >= 0.5).mean() if len(na) else np.nan}
            if model == "POOLED" or len(na) >= MIN_NEW_ABST:
                formula = "f1 ~ new_abst + C(model)" if model == "POOLED" else "f1 ~ new_abst"
                res = gee(formula, s)
                c, se = res.params["new_abst"], res.bse["new_abst"]
                rec.update(gee_diff=c, gee_ci_lo=c - 1.96 * se, gee_ci_hi=c + 1.96 * se,
                           gee_p=res.pvalues["new_abst"])
            if model == "POOLED":
                # Leave-one-model-out: is the pooled difference driven by a single model?
                loo = []
                for m in MODEL_DISPLAY.values():
                    r_ = gee("f1 ~ new_abst + C(model)", s[s["model"] != m])
                    loo.append((m, r_.params["new_abst"], r_.pvalues["new_abst"]))
                loo = pd.DataFrame(loo, columns=["dropped", "diff", "p"])
                worst = loo.loc[loo["p"].idxmax()]
                rec.update(loo_diff_min=loo["diff"].min(), loo_diff_max=loo["diff"].max(),
                           loo_p_min=loo["p"].min(), loo_p_max=loo["p"].max(),
                           loo_n_ns=int((loo["p"] >= 0.05).sum()),
                           loo_worst_model=worst["dropped"], loo_worst_diff=worst["diff"])
            out.append(rec)
    out = pd.DataFrame(out)
    out["gee_q"] = np.nan
    per = out[out["model"] != "POOLED"]
    for _, g in per.groupby(["from", "to"]):
        out.loc[g.index, "gee_q"] = bh(g["gee_p"].values)
    out = out.round(6)
    out.to_csv(RESULTS / f"abstention_selectivity{METRIC_SUFFIX[metric]}.csv", index=False)
    print(f"\n=== Prior {metric} of newly abstained vs still-answered questions (POOLED) ===")
    print(out[out["model"] == "POOLED"].to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Human agreement reference under token F1, EM and RadGraph F1
# ---------------------------------------------------------------------------

def run_human(rows: pd.DataFrame):
    """Inter-annotator agreement vs LLM agreement on the answerable questions with two
    annotator answers. Every score is against ONE annotator's answer, averaged over the two
    annotators: human = annotator A vs B (both directions); LLM = prediction vs A and vs B
    (best over predicted spans for F1/EM, joined prediction for RadGraph, as in score_question).
    Scopes: 'all' (LLM abstention scores 0; the subset is selected on both annotators
    answering, so humans have no abstentions by construction) and 'answered' (questions the
    LLM answered; span agreement only, paired with human agreement on the same questions).
    Per model x condition x scope x metric: report-clustered 95% CI (10,000 reps) for LLM,
    human and LLM − human; Wilcoxon on report-mean differences, BH over 13 models within
    condition x scope x metric. -> human_agreement.csv (6 dp). Loads RadGraph (~20 min)."""
    from src.eval.radqa_metrics_enhanced import RadGraphScorer, compute_exact
    gold = load_gold(SUBSET)
    gold = gold[~gold["gold_impossible"] & (gold["n_gold"] == 2)].set_index("question_id")
    rg, cache = RadGraphScorer(), {}

    def rgs(g, p):
        if (g, p) not in cache:
            cache[(g, p)] = rg.score(g, p)
        return cache[(g, p)]

    def vs(ref, preds):
        return {"f1": max(compute_f1(ref, p) for p in preds),
                "em": max(float(compute_exact(ref, p)) for p in preds),
                "radgraph_f1": rgs(ref, " ".join(preds))}

    metrics = ["f1", "em", "radgraph_f1"]
    human = {}
    for qid, g in gold.iterrows():
        a, b = g["gold_answers"]
        ab, ba = vs(a, [b]), vs(b, [a])
        human[qid] = {m: (ab[m] + ba[m]) / 2 for m in metrics}
    human = pd.DataFrame.from_dict(human, orient="index")
    print(f"human agreement over {len(human)} questions: {human.mean().round(4).to_dict()}")

    recs = []
    s_all = rows[rows["question_id"].isin(gold.index)]
    for (model, k), s in s_all.groupby(["model", "shot"]):
        for r_ in s.itertuples():
            rec = {"model": model, "shot": k, "question_id": r_.question_id,
                   "report_id": r_.report_id, "answered": not r_.pred_imp}
            preds = pred_list(json.loads(r_.pred)) if not r_.pred_imp else []
            if not preds:
                rec.update({m: 0.0 for m in metrics})
            else:
                a, b = gold.loc[r_.question_id, "gold_answers"]
                va, vb = vs(a, preds), vs(b, preds)
                rec.update({m: (va[m] + vb[m]) / 2 for m in metrics})
            recs.append(rec)
    llm = pd.DataFrame(recs)
    llm = llm.join(human.add_prefix("human_"), on="question_id")

    def ci(df, col):
        per = df.assign(one=1.0).groupby("report_id")[[col, "one"]].sum()
        return _cluster_ci_ratio(per[col].values, per["one"].values)

    out = []
    for (model, k), s in llm.groupby(["model", "shot"]):
        for scope, sub in [("all", s), ("answered", s[s["answered"]])]:
            for m in metrics:
                sub = sub.assign(diff=sub[m] - sub[f"human_{m}"])
                rep = sub.groupby("report_id")["diff"].mean()
                p = stats.wilcoxon(rep.values).pvalue if (rep != 0).sum() >= 10 else np.nan
                (llo, lhi), (hlo, hhi), (dlo, dhi) = ci(sub, m), ci(sub, f"human_{m}"), ci(sub, "diff")
                out.append({"model": model, "shot": k, "condition": COND[k], "scope": scope,
                            "metric": m, "n_questions": len(sub), "n_reports": sub["report_id"].nunique(),
                            "llm": sub[m].mean(), "llm_ci_lo": llo, "llm_ci_hi": lhi,
                            "human": sub[f"human_{m}"].mean(), "human_ci_lo": hlo, "human_ci_hi": hhi,
                            "diff": sub["diff"].mean(), "diff_ci_lo": dlo, "diff_ci_hi": dhi,
                            "wilcoxon_p": p})
    hrep = human.join(gold["report_id"])
    for m in metrics:
        lo, hi = ci(hrep, m)
        out.append({"model": "HUMAN", "shot": np.nan, "condition": "—", "scope": "all", "metric": m,
                    "n_questions": len(hrep), "n_reports": hrep["report_id"].nunique(),
                    "human": hrep[m].mean(), "human_ci_lo": lo, "human_ci_hi": hi})
    out = pd.DataFrame(out)
    out["wilcoxon_q"] = np.nan
    per = out[out["model"] != "HUMAN"]
    for _, g in per.groupby(["shot", "scope", "metric"]):
        out.loc[g.index, "wilcoxon_q"] = bh(g["wilcoxon_p"].values)
    out = out.round(6)
    out.to_csv(RESULTS / "human_agreement.csv", index=False)
    print("\n=== LLM vs inter-annotator agreement, k=0 ===")
    print(out[(out["shot"] == 0) | (out["model"] == "HUMAN")]
          [["model", "scope", "metric", "n_questions", "llm", "human", "diff", "diff_ci_lo",
            "diff_ci_hi", "wilcoxon_q"]].to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Span convention: token F1 on the joined prediction
# ---------------------------------------------------------------------------

def run_spanjoin(rows: pd.DataFrame):
    """Token F1 with RadGraph's span convention: score the joined prediction (max over golds)
    instead of the best-matching predicted span. Isolates the span convention from the metric."""
    gold = load_gold(SUBSET).set_index("question_id")["gold_answers"]
    rows = rows.copy()

    def joined(r):
        if r["gold_imp"] or r["pred_imp"]:
            return r["f1"]  # 0/1 answerability score, identical under both conventions
        preds = pred_list(json.loads(r["pred"]) if isinstance(r["pred"], str) else "")
        golds = gold[r["question_id"]]
        return max(compute_f1(g, " ".join(preds)) for g in golds) if preds and golds else 0.0

    rows["f1_joined"] = rows.apply(joined, axis=1)
    multi = rows.apply(lambda r: not (r["gold_imp"] or r["pred_imp"])
                       and len(pred_list(json.loads(r["pred"]) if isinstance(r["pred"], str) else "")) > 1, axis=1)
    print(f"multi-span answered-answerable predictions: {multi.sum()} "
          f"({100 * multi.sum() / (~rows.gold_imp & ~rows.pred_imp).sum():.2f}%); "
          f"joined ≠ best-span on {((rows.f1_joined - rows.f1).abs() > 1e-6).sum()} questions (CSV f1 is rounded)")
    _run_decompose_metric(rows, "f1_joined", "_tokenjoined")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ANALYSES = {"decompose": run_decompose, "hygiene": run_hygiene, "itemsafety": run_itemsafety,
            "gaterobust": run_gate_robust, "accfar": run_accfar, "clusterci": run_clusterci,
            "farpaired": run_farpaired, "selectivity": run_selectivity, "human": run_human,
            "spanjoin": run_spanjoin}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(ANALYSES), default=None)
    args = ap.parse_args()
    questions = load_questions()
    for name in args.only or ANALYSES:
        ANALYSES[name](questions)
