"""
Question-level scoring for RadQA (primary evaluation unit, SQuAD v2 protocol).

The RadQA test split has 614 questions; 280 answerable questions carry two
annotator answers. ``get_flatten_radqa_data`` flattens at the answer level, so
model output files have 894 rows (one per question x annotator answer). The
two rows of a duplicated question have identical model input.

Question-level scoring:
  - Gold answers come from the raw split file (data/radqa/1.0.0/{subset}.json).
  - Prediction = the first row for the question_id in the model output file.
  - Answerability mismatch -> 0; both unanswerable -> 1; else best match
    (max EM/F1) over all gold answers x predicted spans.
  - Aggregate = mean over questions.

All analysis scripts in scripts/ build on ``build_question_table``.
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.eval.radqa_metrics_enhanced import (
    RadGraphScorer,
    best_match_em_f1,
    normalize_answer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "radqa" / "1.0.0"
RESULTS_DIR = PROJECT_ROOT / "results"
SHOTS = [0, 1, 2, 3]
COND = {0: "k=0", 1: "+F", 2: "+F+I", 3: "+F+I+U"}

MODEL_DISPLAY = {
    "deepseek.v3.2": "DeepSeek V3.2",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "mistral.ministral-3-14b-instruct": "Ministral 3 14B",
    "mistral.mistral-large-3-675b-instruct": "Mistral Large 3",
    "openai.gpt-oss-120b-1:0": "GPT-OSS 120B",
    "openai.gpt-oss-20b-1:0": "GPT-OSS 20B",
    "us.anthropic.claude-haiku-4-5-20251001-v1_0": "Claude Haiku 4.5",
    "us.anthropic.claude-opus-5": "Claude Opus 5",
    "us.anthropic.claude-sonnet-4-5-20250929-v1_0": "Claude Sonnet 4.5",
    "us.anthropic.claude-sonnet-5": "Claude Sonnet 5",
    "us.meta.llama4-maverick-17b-instruct-v1:0": "LLaMA 4 Maverick",
    "us.meta.llama4-scout-17b-instruct-v1:0": "LLaMA 4 Scout",
}

# model output field "answerable" -> predicted is_impossible
ANSWERABLE_MAP = {"true": False, "false": True, "True": False, "False": True}

NEGATION_RE = re.compile(
    r"\b(no\b|not\b|no evidence|without|absent|negative|unremarkable|"
    r"no sign|no signs|none|no finding|no findings|cannot exclude)\b",
    re.IGNORECASE,
)

N_BOOT = 10_000
CI_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_boot: int = N_BOOT, seed: int = CI_SEED,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return float(arr.mean()), lo, hi


def is_impossible_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip() == "True"


def pred_is_impossible(pred_row: dict) -> bool:
    return ANSWERABLE_MAP.get(str(pred_row.get("answerable", "")).strip(), False)


def section_of(question_id: str) -> str:
    suffix = str(question_id).rsplit("_", 1)[-1].upper()
    return "Impression" if suffix == "I" else ("Findings" if suffix == "O" else "Unknown")


def pred_list(pred) -> List[str]:
    items = pred if isinstance(pred, list) else [pred]
    return [str(p) for p in items if p is not None and str(p).strip()]


def model_output_paths(shots: Sequence[int] = SHOTS,
                       subset: str = "test") -> List[Tuple[int, str, str, Path]]:
    """(shot, model_dir, display_name, json_path) for every available model output."""
    out = []
    for k in shots:
        for model_dir, disp in MODEL_DISPLAY.items():
            path = PROJECT_ROOT / f"results-{k}shot" / model_dir / f"{subset}.json"
            if path.exists():
                out.append((k, model_dir, disp, path))
    return out


# ---------------------------------------------------------------------------
# Gold + predictions
# ---------------------------------------------------------------------------
def load_gold(subset: str = "test") -> pd.DataFrame:
    """One row per question with the list of all annotator gold answers."""
    with open(DATA_DIR / f"{subset}.json") as f:
        data = json.load(f)
    rows = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                golds = [a["text"] for a in qa["answers"] if str(a.get("text", "")).strip()]
                lens = [len(normalize_answer(g).split()) for g in golds]
                rows.append({
                    "question_id": qa["id"],
                    "report_id": qa["id"].split("_")[0],
                    "section": section_of(qa["id"]),
                    "gold_impossible": is_impossible_bool(qa["is_impossible"]),
                    "gold_answers": golds,
                    "n_gold": len(golds),
                    # negated if any annotator's answer contains a negation cue
                    "has_negation": any(NEGATION_RE.search(g) for g in golds),
                    "gold_len": float(np.mean(lens)) if lens else 0.0,
                })
    gold = pd.DataFrame(rows)
    assert gold["question_id"].is_unique
    return gold


def load_first_predictions(json_path: Path) -> Dict[str, dict]:
    """question_id -> first prediction row in file order."""
    with open(json_path) as f:
        data = json.load(f)
    preds: Dict[str, dict] = {}
    for row in data:
        qid = str(row.get("question_id", ""))
        if qid and qid not in preds:
            preds[qid] = row
    return preds


def score_question(gold_row, pred_row: dict,
                   radgraph: Optional[RadGraphScorer] = None) -> dict:
    gold_imp = gold_row["gold_impossible"]
    pred_imp = pred_is_impossible(pred_row)
    out = {"pred_impossible": pred_imp}

    if gold_imp != pred_imp:
        base = 0.0
    elif gold_imp and pred_imp:
        base = 1.0
    else:
        base = None

    golds = gold_row["gold_answers"]
    preds = pred_list(pred_row.get("answer_extracted", ""))
    if base is None and (not golds or not preds):
        base = 0.0
    if base is not None:
        out.update(f1=base, em=base)
        if radgraph is not None:
            out["radgraph_f1"] = base
        return out

    em, f1 = best_match_em_f1(golds, preds)
    out.update(f1=f1, em=float(em))
    if radgraph is not None:
        # max over annotators, each gold vs the joined prediction
        pred_str = " ".join(preds)
        out["radgraph_f1"] = max(radgraph.score(g, pred_str) for g in golds)
    return out


def evaluate_model(gold: pd.DataFrame, json_path: Path,
                   radgraph: Optional[RadGraphScorer] = None,
                   keep_fields: Iterable[str] = ()) -> pd.DataFrame:
    """Per-question scores for one model output file."""
    preds = load_first_predictions(json_path)
    missing = set(gold["question_id"]) - set(preds)
    if missing:
        raise RuntimeError(f"{json_path}: {len(missing)} questions have no prediction")
    records = []
    for g in gold.itertuples(index=False):
        g = g._asdict()
        p = preds[g["question_id"]]
        rec = {k: g[k] for k in ["question_id", "report_id", "section", "gold_impossible",
                                 "n_gold", "has_negation", "gold_len"]}
        rec.update(score_question(g, p, radgraph))
        rec["pred"] = json.dumps(p.get("answer_extracted", ""), ensure_ascii=False)
        for field in keep_fields:
            rec[field] = p.get(field)
        records.append(rec)
    return pd.DataFrame(records)


def build_question_table(shots: Sequence[int] = SHOTS, subset: str = "test",
                         radgraph: Optional[RadGraphScorer] = None,
                         keep_fields: Iterable[str] = ()) -> pd.DataFrame:
    """One row per model x shot x question (13 x 4 x 614 on the test split).

    ``keep_fields`` copies raw fields from the prediction row (e.g. "reasoning",
    "context", "question", "input_tokens")."""
    gold = load_gold(subset)
    frames = []
    for k, model_dir, disp, path in model_output_paths(shots, subset):
        df = evaluate_model(gold, path, radgraph, keep_fields)
        df.insert(0, "shot", k)
        df.insert(0, "model", disp)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_question_table(shots: Sequence[int] = SHOTS, subset: str = "test") -> pd.DataFrame:
    """Per-question table without RadGraph; reads results/qlevel_per_question.csv if present
    (written by scripts/run_question_level_eval.py), else recomputes."""
    path = RESULTS_DIR / "qlevel_per_question.csv"
    if subset == "test" and path.exists():
        df = pd.read_csv(path, dtype={"report_id": str, "question_id": str})
        df = df[df["shot"].isin(shots)].reset_index(drop=True)
        if "gold_len" in df.columns:
            return df
    return build_question_table(shots, subset)
