"""Provider-agnostic model builder using only public packages.

This file shows how to configure LLM authentication for open-source users.
It shows how to configure LLM authentication using standard provider API keys.

## Setup

Set API keys as environment variables before running:

    # OpenAI (gpt-4o-mini, gpt-4o, etc.)
    export OPENAI_API_KEY="sk-..."

    # Anthropic (claude-opus-4-7, claude-sonnet-4-6, etc.)
    export ANTHROPIC_API_KEY="sk-ant-..."

    # AWS Bedrock (if using bedrock/ model prefix)
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_REGION_NAME="us-east-1"

    # Google Vertex AI (if using vertex_ai/ model prefix)
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"
    # or: gcloud auth application-default login

    # Azure OpenAI (if using azure/ model prefix)
    export AZURE_API_KEY="..."
    export AZURE_API_BASE="https://<your-deployment>.openai.azure.com/"
    export AZURE_API_VERSION="2024-05-01-preview"

## Recommended model strings for paper reproduction

Inner models (scored agent — needs strong tool-calling):
    openai/gpt-4o-mini              — cheapest, good baseline
    openai/gpt-4o                   — higher quality
    anthropic/claude-haiku-4-5      — fast and cheap
    vertex_ai/gemini-1.5-flash      — Google option

Outer models (proposer agent — needs strong reasoning + code):
    anthropic/claude-opus-4-7       — best performance (paper primary)
    anthropic/claude-sonnet-4-6     — good cost/quality tradeoff
    openai/gpt-4o                   — OpenAI option

All model strings use litellm format:
    <provider>/<model-name>
See https://docs.litellm.ai/docs/providers for full list.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def build_model(model_name: str) -> BaseChatModel:
    """Build a LangChain chat model from a litellm model name string.

    Supported prefixes: openai/, anthropic/, bedrock/, azure/, vertex_ai/,
                        cohere/, groq/, together_ai/, huggingface/, ...

    For bedrock/ and azure/ models, ensure the corresponding environment
    variables are set (see module docstring above).
    """
    _LITELLM_PREFIXES = (
        "bedrock/",
        "azure/",
        "vertex_ai/",
        "openai/",
        "anthropic/",
        "cohere/",
        "groq/",
        "together_ai/",
        "huggingface/",
    )
    from langchain.chat_models import init_chat_model

    if any(model_name.startswith(p) for p in _LITELLM_PREFIXES):
        kwargs: dict = {}
        if model_name.startswith("vertex_ai/"):
            import litellm as _litellm

            if getattr(_litellm, "api_base", None):
                kwargs["api_base"] = _litellm.api_base
        return init_chat_model(model_name, model_provider="litellm", **kwargs)

    # Bare model names (e.g. "gpt-4o-mini") — LangChain will auto-detect provider
    return init_chat_model(model_name)
