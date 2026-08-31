"""Shared tool-boundary middleware surface for non-author optimizers (-MW variants).

Gives GEPA-MW and MIPROv2-MW the same middleware surface PRISM's own mutation
slot receives: the candidate carries the full text of custom_middleware.py,
constrained to the three edit patterns (silent correction, error blocking,
prerequisite blocking).
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_MW_STUB = """from __future__ import annotations
from typing import Any, Callable, Awaitable
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents.middleware import AgentMiddleware


class FixMiddleware(AgentMiddleware):

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        call = request.tool_call
        name = call["name"]
        args = call.get("args", {})
        return await handler(request)
"""

MW_PATTERN_GUIDE = (
    "You are editing tool-call interception middleware (a LangChain AgentMiddleware "
    "subclass implementing awrap_tool_call). You may ONLY use these three edit "
    "patterns:\n"
    "1. Silent correction — mutate args before execution via "
    "request.override(tool_call=new_call)\n"
    '2. Block with error (model retries): return ToolMessage(content="Error: ...", '
    'status="error")\n'
    "3. Prerequisite block — verify message history before allowing a tool call\n"
    'ALWAYS use request.override() to mutate args. NEVER mutate call["args"] '
    "directly. Return the COMPLETE new middleware file content as valid Python."
)


def middleware_dir_from_text(text: str | None, dst: Path) -> Path | None:
    """Materialize a middleware component's text into a loadable middleware dir.

    Writes custom_middleware.py plus an auto-generated agent_setup.py wiring
    every AgentMiddleware subclass found in the text. Returns None (no
    middleware) when the text is empty, does not compile, or defines no
    AgentMiddleware subclass — a candidate with broken middleware simply runs
    middleware-free rather than crashing the eval.
    """
    if not text or not text.strip():
        return None
    try:
        compile(text, "custom_middleware.py", "exec")
    except SyntaxError:
        return None
    classes = re.findall(r"^class\s+(\w+)\s*\(\s*AgentMiddleware\s*\)", text, re.M)
    if not classes:
        return None
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "custom_middleware.py").write_text(text)
    imports = f"from custom_middleware import {', '.join(classes)}"
    instances = ", ".join(f"{c}()" for c in classes)
    (dst / "agent_setup.py").write_text(f"{imports}\n\nMIDDLEWARE = [{instances}]\n")
    return dst
