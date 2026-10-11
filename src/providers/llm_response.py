import json
import re
from logging import Logger
from typing import Dict, List, Union

from botocore.exceptions import ClientError
from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.response_model import LLMResponse


class ExceptionLogger(BaseCallbackHandler):
    def __init__(self, logger: Logger, model_id: str):
        super().__init__()
        self.logger = logger
        self.model_id = model_id

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.logger.error(f"Model Id: {self.model_id} | Response Error: {error}")

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        self.logger.error(f"Model Id: {self.model_id} | Response Error: {error}")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        self.logger.error(f"Model Id: {self.model_id} | Response Error: {error}")


class NoStructuredOutputError(Exception):
    """Raised when the model invocation returns None instead of a structured response —
    i.e. the model failed to produce a tool call at all."""

    pass


def _is_throttling_error(exception: BaseException) -> bool:
    if isinstance(exception, ClientError):
        return exception.response.get("Error", {}).get("Code") == "ThrottlingException"
    return "ThrottlingException" in str(exception)


def _is_retryable_error(exception: BaseException) -> bool:
    return _is_throttling_error(exception) or isinstance(
        exception, NoStructuredOutputError
    )


def _parse_json_response(raw_content: str, model_id: str, logger: Logger) -> dict:
    cleaned = re.sub(
        r"^(?:\s*\n\s*)?(?:```(?:json)?\n?|json:\s*\n?)", "", raw_content.strip()
    )
    cleaned = re.sub(r"(?:\n?```\s*$|\n?\s*$)", "", cleaned.strip())
    try:
        # raw_decode parses just the first valid JSON value and tells us where
        # it ended, ignoring any trailing junk (e.g. Llama appending a stray
        # "</answer>" after the closing brace) instead of erroring on it.
        parsed_dict, _ = json.JSONDecoder().raw_decode(cleaned)
        return LLMResponse(**parsed_dict).model_dump()  # always return a plain dict
    except Exception as ex:
        logger.error(
            f"Model Id: {model_id} | JSON parse failure: {ex} | raw_content={raw_content!r}"
        )
        return {
            "answer_extracted": [],
            "answerable": "false",
            "reasoning": "FAILED_TOOL_CALL",
        }


@retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _invoke_with_retry(llm_with_structured_op, messages, config, model_id, logger):
    # Rate limiting now happens inside llm_with_structured_op itself
    # (InMemoryRateLimiter wired in via ChatBedrockConverse(rate_limiter=...)),
    # so every attempt here — first try and retries alike — is already paced
    # without needing a separate wait() call in this file.
    response = llm_with_structured_op.invoke(messages, config=config)

    if "llama" in model_id:
        if response is None or not getattr(response, "content", None):
            raise NoStructuredOutputError("Model returned empty content")
        return _parse_json_response(
            raw_content=response.content, model_id=model_id, logger=logger
        )

    if response is None:
        raise NoStructuredOutputError(
            "Model returned no structured output (response is None)"
        )
    return response


def get_response_from_llm(
    model_id: str,
    messages: List[Dict[str, str]],
    llm_with_structured_op: BaseChatModel,
    logger: Logger,
) -> Dict[str, Union[int, str, List[str], bool, Dict[str, int]]]:
    callback = UsageMetadataCallbackHandler()
    try:
        error_callback = ExceptionLogger(logger=logger, model_id=model_id)

        response = _invoke_with_retry(
            llm_with_structured_op,
            messages,
            config={"callbacks": [callback, error_callback]},
            model_id=model_id,
            logger=logger,
        )

        token_usage = callback.usage_metadata.get(model_id, {})
        # Both the "gemini" path and the "llama" JSON-prompting path already
        # return a plain dict; only the tool-calling paths (Mistral/DeepSeek/
        # GPT-OSS via with_structured_output) return a pydantic object here.
        if isinstance(response, dict):
            final_response = {**response, **token_usage}
        else:
            final_response = {**response.model_dump(), **token_usage}
    except Exception as ex:
        logger.error(f"Model Id: {model_id} | Response Error: {ex}")
        # Even on failure, the underlying call(s) may have already consumed
        # tokens (e.g. a tool call that returned None still cost real usage,
        # or an earlier retry attempt succeeded at the API level before a
        # later one triggered the final failure) — grab whatever usage was
        # captured rather than losing it.
        token_usage = callback.usage_metadata.get(model_id, {})
        final_response = {
            "answer_extracted": [],
            "answerable": "false",
            "reasoning": "FAILED_TOOL_CALL",
            **token_usage,
        }

    return final_response
