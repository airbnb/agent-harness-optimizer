"""LLM model builder — provider-agnostic.

Supports any litellm model string:
  openai/gpt-4.1, anthropic/claude-opus-4-7,
  bedrock/global.anthropic.claude-sonnet-4-6, azure/gpt-5.4-mini, etc.

Auth:
  - Standard env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_API_KEY, etc.
  - AWS Bedrock: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or IAM role
  - Custom auth plugin: set AHO_LITELLM_AUTH_PLUGIN to a module exposing
    initialize(model_name) — useful for gateways that mint credentials per call
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def _ensure_auth(model_name: str) -> None:
    """Initialize auth for a model without building a LangChain object.

    If AHO_LITELLM_AUTH_PLUGIN names an importable module with an
    initialize(model_name) function, it is called for bedrock/azure/vertex
    model strings. Unset (the default), this is a no-op and litellm's
    standard env-var auth applies. Also useful when a third-party library
    (e.g. tau2) calls litellm directly and needs credentials initialized
    before it runs.
    """
    import os

    plugin = os.environ.get("AHO_LITELLM_AUTH_PLUGIN")
    if not plugin:
        return
    try:
        from importlib import import_module

        mod = import_module(plugin)
        if any(model_name.startswith(p) for p in ("bedrock/", "azure/", "vertex_ai/", "global.")):
            mod.initialize(model_name)
            _disable_responses_bridge()
    except ImportError:
        pass


def _disable_responses_bridge() -> None:
    """Keep azure gpt-5.4+ chat calls on Chat Completions (litellm >= 1.98).

    litellm 1.98 bridges gpt-5.4 function-tool chat calls to the /responses
    endpoint, but auth plugins that wrap only the Chat Completions seams
    receive bridged requests without credentials. Function tools are served
    on Chat Completions (the request shape of all prior runs), so strip the
    bridge's mode override. Models explicitly addressed as
    ``responses/…`` keep their routing; litellm builds without the bridge
    (< 1.98) are a no-op.
    """
    try:
        import litellm.main as _lm

        orig = _lm.responses_api_bridge_check
    except (ImportError, AttributeError):
        return
    if getattr(orig, "_aho_no_bridge", False):
        return

    def _check(model, custom_llm_provider, **kwargs):  # type: ignore[no-untyped-def]
        info, m = orig(model, custom_llm_provider, **kwargs)
        if info.get("mode") == "responses" and "responses/" not in model:
            info["mode"] = None
        return info, m

    _check._aho_no_bridge = True  # type: ignore[attr-defined]
    _lm.responses_api_bridge_check = _check


def build_model(model_name: str) -> BaseChatModel:
    """Build a LangChain chat model from a litellm model name string.

    Call this on the main thread before spawning threads — for providers that
    require per-thread auth initialization (bedrock, azure, vertex_ai), this
    function handles it automatically.
    """
    _ensure_auth(model_name)

    _LITELLM_PREFIXES = (
        "bedrock/",
        "azure/",
        "vertex_ai/",
        "openai/",
        "anthropic/",
        "cohere/",
        "groq/",
    )
    from langchain.chat_models import init_chat_model

    if any(model_name.startswith(p) for p in _LITELLM_PREFIXES):
        return init_chat_model(model_name, model_provider="litellm")
    return init_chat_model(model_name)
