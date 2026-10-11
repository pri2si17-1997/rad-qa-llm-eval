"""
Few-shot in-context demonstration sampling and rendering for RadQA.

Design:
  k=1: one answerable Findings example.
  k=2: 1 answerable Findings + 1 answerable Impression (no unanswerable).
  k=3: 1 answerable Findings + 1 answerable Impression + 1 unanswerable.

Demonstrations are drawn from the train split via a fixed seed and reused
verbatim for every test query and every model (Med-PaLM-style fixed prompt).
"""
import json
from typing import Any, Dict, List, Optional

import pandas as pd

FINDINGS_SUFFIX = "_O"
IMPRESSION_SUFFIX = "_I"

# One hand-authored reasoning string per demo slot. Kept model-agnostic so
# results do not depend on any single model's zero-shot output.
DEMO_REASONING = {
    "findings_answerable": (
        "The Findings section explicitly states the requested detail; "
        "the smallest contiguous span answering the question is returned verbatim."
    ),
    "impression_answerable": (
        "The Impression section directly names the requested finding; "
        "the verbatim clause is returned as the answer."
    ),
    "unanswerable": (
        "The context does not mention the information requested by the question, "
        "so no span can be extracted and the question is marked unanswerable."
    ),
}


def _first_gold_answer(row: pd.Series) -> Optional[str]:
    """Return the first non-empty gold answer text for a row, or None."""
    text = row.get("answer_text")
    if isinstance(text, list):
        for t in text:
            if isinstance(t, str) and t.strip():
                return t.strip()
        return None
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _pick_first(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    return df.iloc[0]


def sample_demonstrations(
    train_df: pd.DataFrame,
    k: int,
    seed: int,
    max_context_chars: int = 4000,
) -> List[Dict[str, Any]]:
    """Sample a fixed, stratified set of demonstrations from the train split.

    k=0 -> [] (zero-shot).
    k=1 -> [answerable Findings].
    k=2 -> [answerable Findings, answerable Impression].
    k=3 -> [answerable Findings, answerable Impression, unanswerable].

    Demonstrations are drawn deterministically after shuffling ``train_df`` with
    ``seed``. Context length is capped so demos do not dominate the target report.
    """
    if k == 0:
        return []
    if k not in (1, 2, 3):
        raise ValueError(f"Only k in {{0, 1, 2, 3}} is supported; got {k}")

    df = train_df.copy()
    df = df[df["context"].str.len() <= max_context_chars]
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    findings_answerable = df[
        (df["question_id"].str.endswith(FINDINGS_SUFFIX, na=False))
        & (~df["is_impossible"].astype(bool))
    ]
    impression_answerable = df[
        (df["question_id"].str.endswith(IMPRESSION_SUFFIX, na=False))
        & (~df["is_impossible"].astype(bool))
    ]
    unanswerable = df[df["is_impossible"].astype(bool)]

    demos: List[Dict[str, Any]] = []

    f_row = _pick_first(findings_answerable)
    if f_row is None:
        raise RuntimeError("No answerable Findings example found in train split.")
    f_gold = _first_gold_answer(f_row)
    if not f_gold:
        raise RuntimeError("Selected Findings demo has no gold answer text.")
    demos.append(
        {
            "context": f_row["context"],
            "question": f_row["question"],
            "answer_extracted": [f_gold],
            "answerable": "true",
            "reasoning": DEMO_REASONING["findings_answerable"],
        }
    )

    if k == 1:
        return demos

    i_row = _pick_first(impression_answerable)
    if i_row is None:
        raise RuntimeError("No answerable Impression example found in train split.")
    i_gold = _first_gold_answer(i_row)
    if not i_gold:
        raise RuntimeError("Selected Impression demo has no gold answer text.")
    demos.append(
        {
            "context": i_row["context"],
            "question": i_row["question"],
            "answer_extracted": [i_gold],
            "answerable": "true",
            "reasoning": DEMO_REASONING["impression_answerable"],
        }
    )

    if k == 2:
        return demos

    u_row = _pick_first(unanswerable)
    if u_row is None:
        raise RuntimeError("No unanswerable example found in train split.")
    demos.append(
        {
            "context": u_row["context"],
            "question": u_row["question"],
            "answer_extracted": [],
            "answerable": "false",
            "reasoning": DEMO_REASONING["unanswerable"],
        }
    )

    return demos


def render_demonstrations(demos: List[Dict[str, Any]]) -> str:
    """Render demos as an <example>...</example> block prepended to the user turn.

    Returns "" when demos is empty so zero-shot callers are bit-for-bit unchanged.
    """
    if not demos:
        return ""

    blocks = []
    for demo in demos:
        answer_payload = {
            "answer_extracted": demo["answer_extracted"],
            "answerable": demo["answerable"],
            "reasoning": demo["reasoning"],
        }
        blocks.append(
            "<example>\n"
            f"<context>{demo['context']}</context>\n"
            f"<question>{demo['question']}</question>\n"
            f"<answer>{json.dumps(answer_payload)}</answer>\n"
            "</example>"
        )

    return (
        "Here are examples of how to answer:\n\n"
        + "\n\n".join(blocks)
        + "\n\nNow answer the following:\n"
    )
