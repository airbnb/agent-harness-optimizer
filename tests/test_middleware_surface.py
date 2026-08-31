"""Tests for the shared -MW middleware surface (GEPA-MW / MIPROv2-MW)."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from agent_harness_optimizer.utils.middleware_surface import (
    DEFAULT_MW_STUB,
    MW_PATTERN_GUIDE,
    middleware_dir_from_text,
)

_VALID_MW = """from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


class GuardMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        return await handler(request)


class SecondMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        return await handler(request)
"""


def test_default_stub_materializes(tmp_path):
    d = middleware_dir_from_text(DEFAULT_MW_STUB, tmp_path / "mw")
    assert d is not None
    assert (d / "custom_middleware.py").read_text() == DEFAULT_MW_STUB
    setup = (d / "agent_setup.py").read_text()
    assert "MIDDLEWARE = [FixMiddleware()]" in setup


def test_multiple_classes_all_wired(tmp_path):
    d = middleware_dir_from_text(_VALID_MW, tmp_path / "mw")
    setup = (d / "agent_setup.py").read_text()
    assert "GuardMiddleware()" in setup and "SecondMiddleware()" in setup
    assert "from custom_middleware import GuardMiddleware, SecondMiddleware" in setup


def test_empty_and_none_return_none(tmp_path):
    assert middleware_dir_from_text(None, tmp_path / "a") is None
    assert middleware_dir_from_text("   \n", tmp_path / "b") is None


def test_syntax_error_degrades_to_none(tmp_path):
    assert middleware_dir_from_text("class Broken(:\n  pass", tmp_path / "mw") is None
    assert not (tmp_path / "mw").exists()


def test_no_agentmiddleware_subclass_returns_none(tmp_path):
    assert middleware_dir_from_text("x = 1\n", tmp_path / "mw") is None


def test_guide_as_comments_stays_compilable(tmp_path):
    # MIPROv2-MW seeds instructions with the guide as comments + the stub;
    # that text must still materialize as valid middleware.
    doc = "".join(f"# {line}\n" for line in MW_PATTERN_GUIDE.splitlines()) + DEFAULT_MW_STUB
    d = middleware_dir_from_text(doc, tmp_path / "mw")
    assert d is not None


def test_gepa_config_flag_exists():
    from agent_harness_optimizer.optimizers.gepa.optimizer import GEPAConfig

    assert GEPAConfig().middleware is False
    assert GEPAConfig(middleware=True).middleware is True


def test_miprov2_config_flag_exists():
    from agent_harness_optimizer.optimizers.miprov2.optimizer import MIPROv2Config

    assert MIPROv2Config().middleware is False
    assert MIPROv2Config(middleware=True).middleware is True
