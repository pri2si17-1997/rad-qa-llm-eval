import json
import logging
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


def _unwrap_typed_value(v):
    if isinstance(v, dict) and set(v.keys()) == {"type", "value"}:
        logger.warning(f"Unwrapping typed-value envelope: {v!r}")
        return v["value"]
    return v


class LLMResponse(BaseModel):
    """Structured output for LLM."""

    answer_extracted: List[str] = Field(
        description="Extracted answers from the context for a given question."
    )
    answerable: Literal["true", "false"] = Field(
        description="If a question can be answered from the given context or not. Return either 'true' or 'false'."
    )
    reasoning: str = Field(description="Reasoning behind the answer.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_typed_envelopes(cls, data):
        if isinstance(data, dict):
            return {k: _unwrap_typed_value(v) for k, v in data.items()}
        return data

    @field_validator("answer_extracted", mode="before")
    @classmethod
    def coerce_answer_extracted(cls, v):
        if isinstance(v, str):
            logger.warning(f"answer_extracted returned as string, coercing: {v!r}")
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return [v] if v else []
        return v

    @field_validator("answerable", mode="before")
    @classmethod
    def coerce_answerable(cls, v):
        if isinstance(v, bool):
            logger.warning(f"answerable returned as bool, coercing: {v!r}")
            return "true" if v else "false"
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("reasoning", mode="before")
    @classmethod
    def coerce_reasoning(cls, v):
        if isinstance(v, list):
            logger.warning(f"reasoning returned as list, coercing: {v!r}")
            return " ".join(str(item) for item in v)
        if isinstance(v, dict):
            logger.warning(f"reasoning returned as dict, coercing: {v!r}")
            # last resort: no clean 'value' key to pull from (unwrap already tried that)
            return json.dumps(v)
        return v
