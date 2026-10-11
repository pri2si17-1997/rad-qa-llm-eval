from typing import Dict, Optional

from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from src.response_model import LLMResponse

# Disable boto3's own internal retries — tenacity (in llm_helper.py) is now the
# single source of retry/backoff truth. Without this, boto3 silently retries
# under the hood, bypassing rate limiting and duplicating tenacity's backoff.
_NO_RETRY_CONFIG = Config(retries={"max_attempts": 1, "mode": "standard"})

# One rate limiter per model_id, sized to that model's actual RPM quota.
# Falls back to a conservative default if a model isn't in the map.
_MODEL_RPM_QUOTAS = {
    "us.meta.llama4-maverick-17b-instruct-v1:0": 7,
    "us.meta.llama4-scout-17b-instruct-v1:0": 7,
    "mistral.mistral-large-3-675b-instruct": 99,
    "mistral.ministral-3-14b-instruct": 99,
    "deepseek.v3.2": 99,
    "openai.gpt-oss-120b-1:0": 99,
    "openai.gpt-oss-20b-1:0": 99,
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": 99,
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": 99,
    "us.anthropic.claude-opus-5": 99,
    "us.anthropic.claude-sonnet-5": 99,
}
_DEFAULT_RPM = 5

_rate_limiters: Dict[str, InMemoryRateLimiter] = {}


def _get_rate_limiter(model_id: str) -> InMemoryRateLimiter:
    if model_id not in _rate_limiters:
        rpm = _MODEL_RPM_QUOTAS.get(model_id, _DEFAULT_RPM)
        _rate_limiters[model_id] = InMemoryRateLimiter(
            requests_per_second=rpm / 60.0,
            check_every_n_seconds=0.1,
            max_bucket_size=1,  # no burst allowance — stay strictly under RPM
        )
    return _rate_limiters[model_id]


def get_bedrock_llm(
    model_id: str,
    region_name: str,
    credentials_profile_name: str,
    model_config: Dict[str, int],
    endpoint_url: Optional[str] = None,
) -> BaseChatModel:
    bedrock_llm = ChatBedrockConverse(
        model_id=model_id,
        region_name=region_name,
        credentials_profile_name=credentials_profile_name,
        config=_NO_RETRY_CONFIG,
        rate_limiter=_get_rate_limiter(model_id),
        **({"endpoint_url": endpoint_url} if endpoint_url else {}),
        **model_config,
    )
    return bedrock_llm


def get_bedrock_llm_with_structured_op(
    model_id: str,
    region_name: str,
    credentials_profile_name: str,
    model_config: Dict[str, int],
    endpoint_url: Optional[str] = None,
) -> BaseChatModel:
    bedrock_llm = get_bedrock_llm(
        model_id=model_id,
        region_name=region_name,
        credentials_profile_name=credentials_profile_name,
        model_config=model_config,
        endpoint_url=endpoint_url,
    )
    if "llama" in model_id:
        return bedrock_llm
    # No .with_retry() here — retry logic now lives solely in _invoke_with_retry
    # (tenacity) in llm_helper.py, so it isn't duplicated at this layer too.
    bedrock_with_structured_op = bedrock_llm.with_structured_output(LLMResponse)
    return bedrock_with_structured_op


def get_gemini_llm(model_id: str, model_config: Dict[str, int]) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    gemini_llm = ChatGoogleGenerativeAI(
        model=model_id,
        vertexai=True,
        location="global",
        **model_config,
    )
    return gemini_llm


def get_gemini_llm_with_structured_op(
    model_id: str, model_config: Dict[str, int]
) -> BaseChatModel:
    from google.api_core.exceptions import ResourceExhausted

    gemini_llm = get_gemini_llm(model_id=model_id, model_config=model_config)
    gemini_llm_with_structured_op = gemini_llm.with_structured_output(
        schema=LLMResponse.model_json_schema(), method="json_schema"
    ).with_retry(stop_after_attempt=2, retry_if_exception_type=(ResourceExhausted,))
    return gemini_llm_with_structured_op
