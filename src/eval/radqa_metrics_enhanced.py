"""
RadQA evaluation metrics
========================

Automated metrics for scoring structured LLM output against RadQA gold
annotations: SQuAD v2 token F1 / exact match, RadGraph F1, answerability
confusion counts and rates (false-answer and over-abstention), and McNemar's
test for paired model comparisons. No LLM judge is used.

Expected LLM output schema (per record), matching:

    class LLMResponse(BaseModel):
        answer_extracted: List[str]
        answerable: Literal["true", "false"]
        reasoning: str            # not scored here

Expected gold fields on the same record (joined in from the RadQA dataset):

    is_impossible: bool           # gold answerability
    answer_text: Union[str, List[str]]   # gold answer(s)

Optional dependencies (each is used only if installed):

    pip install radgraph                 # RadGraph F1
    pip install statsmodels              # McNemar's test
"""

import collections
import itertools
import re
import string
from typing import Dict, List, Tuple

import pandas as pd

try:
    from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

try:
    from radgraph import F1RadGraph
    _RADGRAPH_AVAILABLE = True
except ImportError:
    _RADGRAPH_AVAILABLE = False


ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)


# ---------------------------------------------------------------------------
# SQuAD v2-style text normalization (unchanged from the original, adopted
# from https://github.com/huggingface/evaluate/blob/main/metrics/squad_v2)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    if not s:
        return ""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, F1 is 1 if they agree, 0 otherwise.
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# List-level matching for multi-answer records
# ---------------------------------------------------------------------------

def best_match_em_f1(gold_list: List[str], pred_list: List[str]) -> Tuple[float, float]:
    """
    Lenient, SQuAD-style matching.

    Treats gold_list as acceptable variants of a single answer and pred_list
    as candidate spans; returns the single best (max) EM/F1 across all
    gold x pred pairs.
    """
    best_em, best_f1 = 0.0, 0.0
    for g, p in itertools.product(gold_list, pred_list):
        best_em = max(best_em, compute_exact(g, p))
        best_f1 = max(best_f1, compute_f1(g, p))
    return best_em, best_f1


# ---------------------------------------------------------------------------
# RadGraph F1 -- radiology-specific entity/relation overlap
# (Delbrouck et al. 2022 / Jain et al. 2021; validated against radiologists
# in the original RadGraph papers, so no new expert annotation is needed here)
# ---------------------------------------------------------------------------

class RadGraphScorer:
    """Thin wrapper around the RadGraph F1 metric. Requires `pip install radgraph`."""

    def __init__(self, reward_level: str = "partial"):
        if not _RADGRAPH_AVAILABLE:
            raise ImportError(
                "The `radgraph` package is required for RadGraph F1. "
                "Install with `pip install radgraph`."
            )
        self._scorer = F1RadGraph(reward_level=reward_level)

    def score(self, gold: str, pred: str) -> float:
        if not gold and not pred:
            return 1.0
        if not gold or not pred:
            return 0.0
        mean_f1, _, _, _ = self._scorer(hyps=[pred], refs=[gold])
        return float(mean_f1)


# ---------------------------------------------------------------------------
# Answerability metrics -- full battery, clinical-risk framing
# ---------------------------------------------------------------------------

def answerability_confusion(gold_impossible: pd.Series, pred_impossible: pd.Series) -> Dict[str, int]:
    """
    "Positive" = ANSWERABLE (not impossible), matching clinical framing:
      - risky_answer (FP)    = model says answerable, gold says unanswerable
      - over_abstention (FN) = model says unanswerable, gold says answerable
    """
    gold_answerable = ~gold_impossible.astype(bool)
    pred_answerable = ~pred_impossible.astype(bool)

    tp = int((gold_answerable & pred_answerable).sum())
    tn = int((~gold_answerable & ~pred_answerable).sum())
    fp = int((~gold_answerable & pred_answerable).sum())
    fn = int((gold_answerable & ~pred_answerable).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def answerability_metrics(confusion: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = confusion["tp"], confusion["tn"], confusion["fp"], confusion["fn"]
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / n if n else 0.0

    n_gold_unanswerable = tn + fp
    n_gold_answerable = tp + fn
    risky_answer_rate = fp / n_gold_unanswerable if n_gold_unanswerable else 0.0
    over_abstention_rate = fn / n_gold_answerable if n_gold_answerable else 0.0

    return {
        "answerability_accuracy": accuracy,
        "answerability_precision": precision,
        "answerability_recall": recall,
        "answerability_f1": f1,
        "risky_answer_rate": risky_answer_rate,
        "over_abstention_rate": over_abstention_rate,
    }


# ---------------------------------------------------------------------------
# Statistics: McNemar's test for pairwise model comparison
# ---------------------------------------------------------------------------

def mcnemar_pairwise(correct_a: pd.Series, correct_b: pd.Series) -> Dict[str, float]:
    """
    McNemar's test for two models' paired binary correctness indicators
    (e.g. EM-correct or answerability-correct) evaluated on the SAME
    questions.
    """
    if not _STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for McNemar's test. Install with `pip install statsmodels`.")

    correct_a = correct_a.astype(bool)
    correct_b = correct_b.astype(bool)
    both_correct = int((correct_a & correct_b).sum())
    both_wrong = int((~correct_a & ~correct_b).sum())
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())

    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = _mcnemar(table, exact=(a_only + b_only) < 25, correction=True)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
