"""Standalone BFCL v3 scorer, modeled on the terminal bench harness pattern.

Replicates the DeepAgentsWrapper/Harbor approach from deepagents_harbor/:
- async ainvoke (matches terminal bench's async run loop)
- ATIF v1.2 trajectory format (Step/ToolCall/Observation) saved per case
- Token tracking from AIMessage.usage_metadata
- Stuck-state detection (repeated tool loop, no-progress, exception)
- All harness metrics from spec sections 7.1–7.4

Usage:
    score = asyncio.run(score_bfcl(system_prompt="...", model_name="..."))
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import inspect
import json
import random
import re
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resolve paths for BFCL data files (relative to this file's location).
# ---------------------------------------------------------------------------

_HERE_RUNNER = Path(__file__).resolve().parent
_BFCL_DATA_DIR = _HERE_RUNNER.parent.parent / "data" / "bfcl"
_BFCL_DATA = _BFCL_DATA_DIR / "bfcl_v3_final.json"  # legacy 12-case file (kept for reference)
_BFCL_V4_FILES = [
    ("BFCL_v4_multi_turn_base.json", "BFCL_v4_multi_turn_base.json"),
    ("BFCL_v4_multi_turn_long_context.json", "BFCL_v4_multi_turn_long_context.json"),
    ("BFCL_v4_multi_turn_miss_func.json", "BFCL_v4_multi_turn_miss_func.json"),
    ("BFCL_v4_multi_turn_miss_param.json", "BFCL_v4_multi_turn_miss_param.json"),
]

from deepagents import create_deep_agent  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agent_harness_optimizer.benchmarks.bfcl_apis.gorilla_file_system import (  # noqa: E402
    GorillaFileSystem,
)
from agent_harness_optimizer.benchmarks.bfcl_apis.math_api import MathAPI  # noqa: E402
from agent_harness_optimizer.benchmarks.bfcl_apis.message_api import MessageAPI  # noqa: E402
from agent_harness_optimizer.benchmarks.bfcl_apis.posting_api import TwitterAPI  # noqa: E402
from agent_harness_optimizer.benchmarks.bfcl_apis.ticket_api import TicketAPI  # noqa: E402
from agent_harness_optimizer.benchmarks.bfcl_apis.trading_bot import TradingBot  # noqa: E402
from agent_harness_optimizer.benchmarks.bfcl_apis.travel_booking import TravelAPI  # noqa: E402
from agent_harness_optimizer.benchmarks.bfcl_apis.vehicle_control import (  # noqa: E402
    VehicleControlAPI,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BFCL_V3_IDS = {
    "multi_turn_composite_97",
    "multi_turn_composite_116",
    "multi_turn_composite_199",
    "multi_turn_miss_func_55",
    "multi_turn_miss_param_55",
}

_CLASS_REGISTRY: dict[str, type] = {
    "VehicleControlAPI": VehicleControlAPI,
    "MessageAPI": MessageAPI,
    "TradingBot": TradingBot,
    "TravelAPI": TravelAPI,
    "TicketAPI": TicketAPI,
    "TwitterAPI": TwitterAPI,
    "GorillaFileSystem": GorillaFileSystem,
    "MathAPI": MathAPI,
}

DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant with access to domain-specific API tools. "
    "Use these tools to accomplish the user's requests. "
    "Do not use the task tool or any subagent delegation. "
    "Do not use file tools (ls, read_file, write_file, etc.)."
)

_DEFAULT_MODEL = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"

# Stuck-state detection thresholds (mirrors spec section 7.6)
_STUCK_REPEAT_WINDOW = 5  # check last N steps
_STUCK_REPEAT_MIN = 3  # same tool+args N times = stuck

# Resource budgets — three independent axes
_TIME_BUDGET_S: float = 300.0  # wall-clock seconds per case (hard kill)
_STEP_BUDGET: int = 100  # max total tool calls per case
_TOKEN_BUDGET: int = 500_000  # max total (prompt + completion) tokens per case

# stuck_type vocabulary:
#   ""            — clean termination, no budget exceeded
#   "loop"        — _STUCK_REPEAT_MIN identical (tool, args) in last _STUCK_REPEAT_WINDOW steps
#   "timeout"     — wall-clock budget exceeded
#   "step_limit"  — tool call count exceeded _STEP_BUDGET
#   "token_limit" — token count exceeded _TOKEN_BUDGET
#   "infra_error" — infrastructure failure (DeploymentNotFound, RateLimitError after retries,
#                    ServiceUnavailable)
#   "crash"       — unhandled exception before terminal state (real agent/code failure)

# Substrings in an exception traceback that indicate infrastructure / provider failures
# rather than real agent crashes. Covers LiteLLM exception class names, Bedrock native
# error codes (ThrottlingException, ModelNotReadyException, etc.) and HTTP status signals.
_INFRA_ERROR_SIGNALS: tuple[str, ...] = (
    # LiteLLM exception class names (all providers routed through litellm)
    "RateLimitError",
    "NotFoundError",
    "ServiceUnavailableError",
    "APIConnectionError",
    "InternalServerError",
    "BadGatewayError",
    "AuthenticationError",
    # Bedrock-native error codes (surfaced in exception message before litellm wraps them)
    "ThrottlingException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ModelStreamErrorException",
    "ProvisionedThroughputExceededException",
    "ServiceQuotaExceededException",
    "InternalServerException",  # Bedrock InternalServerException (503-equivalent)
    # Azure-specific
    "DeploymentNotFound",
    "deployment does not exist",
    # Generic HTTP / network signals
    "429",
    "Too Many Requests",
    "rate_limit_exceeded",
    "503",
    "Service Unavailable",
    "Connection reset",
    "Connection refused",
    "RemoteDisconnected",
    "ReadTimeout",
    "ConnectTimeout",
)


# ---------------------------------------------------------------------------
# ATIF-style trajectory data structures (mirrors deepagents_harbor/wrapper.py)
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryToolCall:
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "function_name": self.function_name,
            "arguments": self.arguments,
        }


@dataclass
class TrajectoryObservation:
    source_call_id: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"source_call_id": self.source_call_id, "content": self.content}


@dataclass
class TrajectoryStep:
    step_id: int
    timestamp: str
    source: str  # "user" | "agent"
    message: str
    tool_calls: list[TrajectoryToolCall] = field(default_factory=list)
    observations: list[TrajectoryObservation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "message": self.message,
        }
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.observations:
            d["observations"] = [o.to_dict() for o in self.observations]
        return d


@dataclass
class ATIFTrajectory:
    """ATIF v1.2 trajectory — same schema as deepagents_harbor."""

    schema_version: str
    session_id: str
    case_id: str
    model_name: str
    steps: list[TrajectoryStep]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_steps: int
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "case_id": self.case_id,
            "agent": {
                "name": "bfcl-deepagent",
                "model_name": self.model_name,
                "framework": "deepagents",
            },
            "steps": [s.to_dict() for s in self.steps],
            "final_metrics": {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_steps": self.total_steps,
                "wall_time_s": self.wall_time_s,
            },
        }


# ---------------------------------------------------------------------------
# Harness result data classes
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    state_diff: str
    error: str
    tool_calls: list[dict[str, Any]]  # {"name": "fn", "args": {...}} dicts for analysis
    turns: int
    wall_time_s: float
    prompt_tokens: int
    completion_tokens: int
    stuck_type: str  # "" | "loop" | "timeout" | "step_limit" | "token_limit" | "crash"
    trajectory: ATIFTrajectory | None

    # Harness budget flags
    boundary_violation: bool = False
    human_intervention: bool = False
    within_time_budget: bool = True
    within_step_budget: bool = True
    within_token_budget: bool = True

    @property
    def within_cost_budget(self) -> bool:
        """Alias: cost budget = token budget (tokens proxy compute cost)."""
        return self.within_token_budget

    @property
    def vasr_eligible(self) -> bool:
        return (
            self.passed
            and not self.boundary_violation
            and not self.human_intervention
            and self.within_time_budget
            and self.within_step_budget
            and self.within_token_budget
        )


@dataclass
class BFCLScore:
    """Aggregated BFCL run results with harness effectiveness metrics (spec 7.1–7.4)."""

    results: list[CaseResult]
    total_wall_time_s: float

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def total(self) -> int:
        return len(self.results)

    # 7.1 Effectiveness
    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.results else 0.0

    @property
    def vasr(self) -> float:
        return (
            sum(r.vasr_eligible for r in self.results) / len(self.results) if self.results else 0.0
        )

    @property
    def autonomy_gap(self) -> float:
        return self.success_rate - self.vasr

    @property
    def cost_per_valid_success(self) -> float:
        # proxy: total tokens as cost unit
        valid = sum(r.vasr_eligible for r in self.results)
        total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in self.results)
        return float("inf") if valid == 0 else total_tokens / valid

    @property
    def runtime_per_valid_success(self) -> float:
        valid = sum(r.vasr_eligible for r in self.results)
        return float("inf") if valid == 0 else self.total_wall_time_s / valid

    # 7.2 Reliability
    @property
    def stuck_rate(self) -> float:
        """Fraction of cases that did not reach a clean terminal state."""
        return (
            sum(bool(r.stuck_type) for r in self.results) / len(self.results)
            if self.results
            else 0.0
        )

    def stuck_breakdown(self) -> dict[str, int]:
        """Count of each stuck subtype: loop, timeout, step_limit, token_limit, infra_error, crash."""  # noqa: E501
        counts: dict[str, int] = {
            "loop": 0,
            "timeout": 0,
            "step_limit": 0,
            "token_limit": 0,
            "infra_error": 0,
            "crash": 0,
        }
        for r in self.results:
            if r.stuck_type in counts:
                counts[r.stuck_type] += 1
        return counts

    @property
    def infra_error_rate(self) -> float:
        """Fraction of cases that failed due to infrastructure errors (deployment/rate limit)."""
        return (
            sum(r.stuck_type == "infra_error" for r in self.results) / len(self.results)
            if self.results
            else 0.0
        )

    def is_valid(self, max_infra_error_rate: float = 0.05) -> bool:
        """Returns False if infra_error_rate exceeds threshold — run should be discarded."""
        return self.infra_error_rate <= max_infra_error_rate

    # 7.3 Compliance
    @property
    def no_violation_rate(self) -> float:
        return (
            sum(not r.boundary_violation for r in self.results) / len(self.results)
            if self.results
            else 1.0
        )

    @property
    def zero_intervention_success_rate(self) -> float:
        return (
            sum(r.passed and not r.human_intervention for r in self.results) / len(self.results)
            if self.results
            else 0.0
        )

    # 7.4 Trace diagnosability
    @property
    def trace_completeness(self) -> float:
        complete = sum(r.trajectory is not None for r in self.results)
        return complete / len(self.results) if self.results else 0.0

    def by_category(self) -> dict[str, dict[str, Any]]:
        """Per-category breakdown of pass rate and VASR."""
        cats: dict[str, list[CaseResult]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r)
        return {
            cat: {
                "passed": sum(r.passed for r in rs),
                "total": len(rs),
                "success_rate": round(sum(r.passed for r in rs) / len(rs), 4),
                "vasr": round(sum(r.vasr_eligible for r in rs) / len(rs), 4),
            }
            for cat, rs in sorted(cats.items())
        }

    def to_dict(self) -> dict[str, Any]:
        breakdown = self.stuck_breakdown()
        n = len(self.results) or 1
        return {
            "success_rate": round(self.success_rate, 4),
            "vasr": round(self.vasr, 4),
            "autonomy_gap": round(self.autonomy_gap, 4),
            "runtime_per_valid_success": round(self.runtime_per_valid_success, 2),
            "cost_per_valid_success": round(self.cost_per_valid_success, 2),
            # 7.2 Reliability — three-axis budget breakdown
            "stuck_rate": round(self.stuck_rate, 4),
            "reliability": round(1.0 - self.stuck_rate, 4),
            "infra_error_rate": round(self.infra_error_rate, 4),
            "is_valid": self.is_valid(),
            "stuck_breakdown": breakdown,
            "stuck_breakdown_rate": {k: round(v / n, 4) for k, v in breakdown.items()},
            "budgets": {
                "time_budget_s": _TIME_BUDGET_S,
                "step_budget": _STEP_BUDGET,
                "token_budget": _TOKEN_BUDGET,
            },
            "no_violation_rate": round(self.no_violation_rate, 4),
            "zero_intervention_success_rate": round(self.zero_intervention_success_rate, 4),
            "trace_completeness": round(self.trace_completeness, 4),
            "passed": self.passed,
            "total": self.total,
            "total_wall_time_s": round(self.total_wall_time_s, 2),
            "by_category": self.by_category(),
            "per_case": [
                {
                    "case_id": r.case_id,
                    "passed": r.passed,
                    "vasr_eligible": r.vasr_eligible,
                    "state_diff": r.state_diff,
                    "error": r.error,
                    "tool_calls": r.tool_calls,
                    "turns": r.turns,
                    "wall_time_s": round(r.wall_time_s, 2),
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "stuck_type": r.stuck_type,
                    "within_time_budget": r.within_time_budget,
                    "within_step_budget": r.within_step_budget,
                    "within_token_budget": r.within_token_budget,
                }
                for r in self.results
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BFCLScore:
        """Reconstruct a BFCLScore from a saved to_dict() snapshot (no trajectories)."""
        results = [
            CaseResult(
                case_id=c["case_id"],
                category=c.get("category", ""),
                passed=c["passed"],
                state_diff=c.get("state_diff", ""),
                error=c.get("error", ""),
                tool_calls=c.get("tool_calls", []),
                turns=c.get("turns", 0),
                wall_time_s=c.get("wall_time_s", 0.0),
                prompt_tokens=c.get("prompt_tokens", 0),
                completion_tokens=c.get("completion_tokens", 0),
                stuck_type=c.get("stuck_type", ""),
                trajectory=None,
                boundary_violation=not c.get("vasr_eligible", c["passed"]) and c["passed"],
                human_intervention=False,
                within_time_budget=c.get("within_time_budget", True),
                within_step_budget=c.get("within_step_budget", True),
                within_token_budget=c.get("within_token_budget", True),
            )
            for c in d.get("per_case", [])
        ]
        return cls(results=results, total_wall_time_s=d.get("total_wall_time_s", 0.0))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


_NO_TEMP_MODELS = ("o1", "o3", "o4", "gpt-5")


# _build_model: delegates auth to agent_harness_optimizer.utils.llm.build_model, then applies
# temperature/seed overrides needed by the BFCL scorer.
def _build_model(model_name: str, temperature: float | None = None, seed: int | None = None):
    """Build a LangChain chat model with optional temperature/seed overrides."""
    from agent_harness_optimizer.utils.llm import build_model

    if temperature is None and seed is None:
        return build_model(model_name)
    # build_model triggers auth; then construct a fresh model with extra kwargs.
    build_model(model_name)  # side-effect: initializes LFH auth if needed
    base = model_name.split("/")[-1]
    supports_temp = not any(base.startswith(m) for m in _NO_TEMP_MODELS)
    kwargs: dict = {}
    if supports_temp and temperature is not None:
        kwargs["temperature"] = temperature
    if supports_temp and seed is not None:
        kwargs["seed"] = seed
    _LITELLM_PREFIXES = (
        "bedrock/",
        "azure/",
        "vertex_ai/",
        "openai/",
        "anthropic/",
        "cohere/",
        "groq/",
    )
    if any(model_name.startswith(p) for p in _LITELLM_PREFIXES):
        return init_chat_model(model_name, model_provider="litellm", **kwargs)
    return init_chat_model(model_name, **kwargs)


# ---------------------------------------------------------------------------
# BFCL data helpers
# ---------------------------------------------------------------------------


def _load_cases(use_v4: bool = True) -> list[dict[str, Any]]:
    """Load BFCL cases. Uses v4 (800 cases) by default; falls back to v3 (12 cases)."""
    if use_v4:
        return _load_v4_cases()
    # Legacy v3 path
    data = json.loads(_BFCL_DATA.read_text())
    cases = data if isinstance(data, list) else data.get("tasks", data.get("cases", []))
    return [c for c in cases if c["id"] in _BFCL_V3_IDS]


def _load_v4_cases() -> list[dict[str, Any]]:
    """Load and merge all 800 BFCL v4 multi-turn cases from JSONL + possible_answer files.

    Normalises v4 schema to match the v3 shape the rest of the harness expects:
      - question  -> conversation
      - ground_truth from possible_answer/  -> ground_truth
    """
    answer_dir = _BFCL_DATA_DIR / "possible_answer"
    cases: list[dict[str, Any]] = []

    for data_file, answer_file in _BFCL_V4_FILES:
        data_path = _BFCL_DATA_DIR / data_file
        answer_path = answer_dir / answer_file
        if not data_path.exists() or not answer_path.exists():
            continue

        # Build answer lookup by id
        answers: dict[str, list] = {}
        for line in answer_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            answers[obj["id"]] = obj["ground_truth"]

        # Load cases and normalise
        for line in data_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            case_id = obj["id"]
            cases.append(
                {
                    "id": case_id,
                    "category": data_file.replace("BFCL_v4_", "").replace(".json", ""),
                    "involved_classes": obj.get("involved_classes", []),
                    "tools": [],
                    "conversation": obj.get("question", []),
                    "initial_config": obj.get("initial_config", {}),
                    "ground_truth": answers.get(case_id, []),
                    "num_turns": len(obj.get("question", [])),
                    "num_expected_calls": len(answers.get(case_id, [])),
                    "difficulty": "hard",
                    "axes": [],
                    "rationale": "",
                }
            )

    return cases


STATELESS_CLASSES = {"MathAPI"}


def _instantiate_apis(case: dict[str, Any]) -> dict[str, Any]:
    instances: dict[str, Any] = {}
    for cls_name in case["involved_classes"]:
        instance = _CLASS_REGISTRY[cls_name]()
        if cls_name not in STATELESS_CLASSES:
            instance._load_scenario(
                copy.deepcopy(case["initial_config"].get(cls_name, {})), long_context=False
            )
        instances[cls_name] = instance
    return instances


def _load_tool_descriptions(tool_descriptions_dir: Path | None) -> dict[str, str]:
    """Load custom tool descriptions from a directory of per-class .txt files.

    Each file is named <ClassName>.txt and contains one line per method in the format:
        method_name(sig) — description

    Returns a dict mapping method_name -> description override.
    """
    if tool_descriptions_dir is None or not tool_descriptions_dir.is_dir():
        return {}
    overrides: dict[str, str] = {}
    for txt_file in tool_descriptions_dir.glob("*.txt"):
        for line in txt_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Accept "method_name(...) — desc" or "method_name — desc"
            sep = " — "
            if sep in line:
                lhs, desc = line.split(sep, 1)
                method_name = lhs.split("(")[0].strip()
                if method_name:
                    overrides[method_name] = desc.strip()
    return overrides


def _wrap_as_tools(
    instances: dict[str, Any],
    tool_descriptions_dir: Path | None = None,
) -> list[StructuredTool]:
    """Wrap API instance methods as LangChain StructuredTools.

    If tool_descriptions_dir is provided, any per-method description overrides
    found there (from files named <ClassName>.txt) will replace the default
    docstring-based description.
    """
    description_overrides = _load_tool_descriptions(tool_descriptions_dir)
    tools: list[StructuredTool] = []
    for instance in instances.values():
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if name.startswith("_"):
                continue
            description = description_overrides.get(name) or (method.__doc__ or "").strip()
            tools.append(
                StructuredTool.from_function(
                    func=_return_errors(method),
                    name=name,
                    description=description,
                )
            )
    return tools


def _return_errors(method: Any) -> Any:
    """Return sim-API exceptions to the model as error strings (BFCL semantics).

    langgraph >= 1.2 re-raises non-ToolInvocationError tool exceptions, which
    would turn a model's invalid request (e.g. TravelAPI ValueError) into a
    case-level crash instead of feedback the model can react to.
    """

    @functools.wraps(method)
    def _safe(*args: Any, **kwargs: Any) -> Any:
        try:
            return method(*args, **kwargs)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    return _safe


def _fix_gt_call(s: str) -> str:
    s = re.sub(r",\s*sender_id=['\"][^'\"]*['\"](?=\s*\))", "", s)
    return re.sub(r"sender_id=['\"][^'\"]*['\"],\s*", "", s)


def _replay_gt(case: dict[str, Any]) -> dict[str, Any]:
    gt_instances = _instantiate_apis(case)
    methods: dict[str, Any] = {
        name: method
        for inst in gt_instances.values()
        for name, method in inspect.getmembers(inst, predicate=inspect.ismethod)
        if not name.startswith("_")
    }
    for turn_gt in case["ground_truth"]:
        for call_str in turn_gt:
            with contextlib.suppress(Exception):
                eval(_fix_gt_call(call_str), {"__builtins__": {}}, methods)
    return gt_instances


def _state_diff(
    model_instances: dict[str, Any], gt_instances: dict[str, Any], case: dict[str, Any]
) -> str:
    diffs: list[str] = []
    for cls_name in case["involved_classes"]:
        m_inst = model_instances[cls_name]
        g_inst = gt_instances[cls_name]
        for attr in vars(g_inst):
            if attr.startswith("_"):
                continue
            m_val = getattr(m_inst, attr)
            g_val = getattr(g_inst, attr)
            if m_val != g_val:
                diffs.append(f"  {cls_name}.{attr}: got={m_val!r}, expected={g_val!r}")
    return "\n".join(diffs)


# ---------------------------------------------------------------------------
# ATIF trajectory builder (mirrors deepagents_harbor/_save_trajectory)
# ---------------------------------------------------------------------------


def _build_trajectory(
    *,
    messages: list,
    case_id: str,
    session_id: str,
    model_name: str,
    wall_time_s: float,
    conversation: list[list[dict[str, Any]]],
) -> tuple[ATIFTrajectory, int, int, list[dict[str, Any]], str]:
    """Convert agent messages into ATIF trajectory + extract metrics.

    Returns:
        (trajectory, prompt_tokens, completion_tokens, tool_call_dicts, stuck_type)
    """
    total_prompt = 0
    total_completion = 0
    all_tool_call_dicts: list[dict[str, Any]] = []

    steps: list[TrajectoryStep] = []
    # First step: inject the first user turn
    if conversation and conversation[0]:
        first = conversation[0][0]
        first_content = (
            first.get("content", "") if isinstance(first, dict) else getattr(first, "content", "")
        )
        steps.append(
            TrajectoryStep(
                step_id=1,
                timestamp=datetime.now(UTC).isoformat(),
                source="user",
                message=first_content or "",
            )
        )

    pending_step: TrajectoryStep | None = None
    observations: list[TrajectoryObservation] = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                total_prompt += usage.get("input_tokens", 0)
                total_completion += usage.get("output_tokens", 0)

            if pending_step is not None:
                if observations:
                    pending_step.observations = observations
                    observations = []
                steps.append(pending_step)
                pending_step = None

            atif_tool_calls: list[TrajectoryToolCall] = []
            text = ""
            for cb in (
                msg.content_blocks if hasattr(msg, "content_blocks") and msg.content_blocks else []
            ):
                if cb.get("type") == "text":
                    text += cb["text"]
                elif cb.get("type") == "tool_call":
                    atif_tool_calls.append(
                        TrajectoryToolCall(
                            tool_call_id=cb.get("id", ""),
                            function_name=cb.get("name", ""),
                            arguments=cb.get("args", {}),
                        )
                    )
                    all_tool_call_dicts.append(
                        {"name": cb.get("name", ""), "args": cb.get("args", {})}
                    )
            # Fallback for plain tool_calls attribute
            if not atif_tool_calls:
                for tc in getattr(msg, "tool_calls", None) or []:
                    atif_tool_calls.append(
                        TrajectoryToolCall(
                            tool_call_id=tc.get("id", ""),
                            function_name=tc.get("name", ""),
                            arguments=tc.get("args", {}),
                        )
                    )
                    all_tool_call_dicts.append(
                        {"name": tc.get("name", ""), "args": tc.get("args", {})}
                    )
            if not text and hasattr(msg, "text"):
                text = msg.text or ""

            new_step = TrajectoryStep(
                step_id=len(steps) + 1,
                timestamp=datetime.now(UTC).isoformat(),
                source="agent",
                message=text,
                tool_calls=atif_tool_calls,
            )
            if atif_tool_calls:
                pending_step = new_step
            else:
                steps.append(new_step)

        elif isinstance(msg, ToolMessage):
            observations.append(
                TrajectoryObservation(
                    source_call_id=msg.tool_call_id,
                    content=str(msg.content),
                )
            )

    if pending_step is not None:
        if observations:
            pending_step.observations = observations
        steps.append(pending_step)

    stuck_type = _detect_stuck(steps)

    traj = ATIFTrajectory(
        schema_version="ATIF-v1.2",
        session_id=session_id,
        case_id=case_id,
        model_name=model_name,
        steps=steps,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_steps=len(steps),
        wall_time_s=wall_time_s,
    )
    return traj, total_prompt, total_completion, all_tool_call_dicts, stuck_type


# ---------------------------------------------------------------------------
# Stuck-state detection (spec section 7.6)
# ---------------------------------------------------------------------------


def _detect_stuck(steps: list[TrajectoryStep]) -> str:
    """Return 'loop' if the last _STUCK_REPEAT_WINDOW steps contain a repeated (tool, args)
    pair _STUCK_REPEAT_MIN or more times. Returns '' otherwise."""
    recent = steps[-_STUCK_REPEAT_WINDOW:]
    tool_call_keys: list[str] = []
    for step in recent:
        for tc in step.tool_calls:
            tool_call_keys.append(f"{tc.function_name}:{json.dumps(tc.arguments, sort_keys=True)}")
    if len(tool_call_keys) >= _STUCK_REPEAT_MIN:
        from collections import Counter

        counts = Counter(tool_call_keys)
        if counts.most_common(1)[0][1] >= _STUCK_REPEAT_MIN:
            return "loop"
    return ""


# ---------------------------------------------------------------------------
# Async case runner (mirrors terminal bench's async run())
# ---------------------------------------------------------------------------


async def _run_case_async(
    case: dict[str, Any],
    model: Any,
    system_prompt: str,
    model_name: str,
    time_budget_s: float,
    tool_descriptions_dir: Path | None = None,
    middleware: list[Any] | None = None,
) -> CaseResult:
    """Run one BFCL case asynchronously, building an ATIF trajectory."""
    session_id = str(uuid.uuid4())
    t0 = time.monotonic()

    model_instances = _instantiate_apis(case)
    tools = _wrap_as_tools(model_instances, tool_descriptions_dir=tool_descriptions_dir)
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=MemorySaver(),
        **({"middleware": middleware} if middleware else {}),
    )
    config = {"configurable": {"thread_id": session_id}}

    invoke_result: dict[str, Any] | None = None
    error_msg = ""
    timed_out = False
    try:

        async def _run_turns() -> None:
            nonlocal invoke_result
            for turn_messages in case["conversation"]:
                # Copy: langgraph's ensure_message_ids mutates the input list
                # in place (dicts -> BaseMessage), which would corrupt
                # case["conversation"] for _build_trajectory and case reuse.
                msgs = [dict(m) for m in turn_messages] or [
                    {"role": "user", "content": "Please continue and complete any remaining tasks."}
                ]
                invoke_result = await agent.ainvoke({"messages": msgs}, config)

        await asyncio.wait_for(_run_turns(), timeout=time_budget_s)
    except TimeoutError:
        timed_out = True
        error_msg = f"TimeoutError: case exceeded {time_budget_s}s budget"
    except Exception:
        error_msg = traceback.format_exc()

    wall_time = time.monotonic() - t0
    within_time = wall_time <= time_budget_s

    if error_msg or invoke_result is None:
        _is_infra = not timed_out and any(s in (error_msg or "") for s in _INFRA_ERROR_SIGNALS)
        stuck_type = "timeout" if timed_out else ("infra_error" if _is_infra else "crash")
        return CaseResult(
            case_id=case["id"],
            category=case["category"],
            passed=False,
            state_diff="",
            error=error_msg or "No result returned",
            tool_calls=[],
            turns=len(case["conversation"]),
            wall_time_s=wall_time,
            prompt_tokens=0,
            completion_tokens=0,
            stuck_type=stuck_type,
            trajectory=None,
            within_time_budget=within_time,
            within_step_budget=True,
            within_token_budget=True,
        )

    messages = invoke_result.get("messages", [])
    traj, prompt_tok, compl_tok, tool_call_dicts, stuck_type = _build_trajectory(
        messages=messages,
        case_id=case["id"],
        session_id=session_id,
        model_name=model_name,
        wall_time_s=wall_time,
        conversation=case["conversation"],
    )

    total_tokens = prompt_tok + compl_tok
    total_steps = len(tool_call_dicts)
    within_step = total_steps <= _STEP_BUDGET
    within_token = total_tokens <= _TOKEN_BUDGET

    # Promote resource-budget violations to stuck type if not already set by loop detection
    if not stuck_type:
        if not within_step:
            stuck_type = "step_limit"
        elif not within_token:
            stuck_type = "token_limit"

    gt_instances = _replay_gt(case)
    diff = _state_diff(model_instances, gt_instances, case)

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        passed=not diff,
        state_diff=diff,
        error="",
        tool_calls=tool_call_dicts,
        turns=len(case["conversation"]),
        wall_time_s=wall_time,
        prompt_tokens=prompt_tok,
        completion_tokens=compl_tok,
        stuck_type=stuck_type,
        trajectory=traj,
        within_time_budget=within_time,
        within_step_budget=within_step,
        within_token_budget=within_token,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Train/holdout/eval split — deterministic by case_id hash, 1/8 buckets each.
# Bucket 0 = train (~100 cases across 4 categories)
# Bucket 1 = holdout (~100 cases across 4 categories)
# Buckets 2-7 = eval-only (used for final 800-case paper number)
# ---------------------------------------------------------------------------

_ALL_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
]


def _build_split_map(split_seed: int | None = None) -> dict[str, str]:
    """Build a deterministic stratified split map.

    Default (split_seed=None): 25/25/150 per category (original behavior).
      rank  0-24  → 'train'     (25/category × 4 = 100 total)
      rank 25-49  → 'holdout'   (25/category × 4 = 100 total)
      rank 50-199 → 'scorecard' (150/category × 4 = 600 total)

    4-fold CV (split_seed=0..3): 25/25/150 per category, disjoint across seeds.
      Each category's 200 cases are divided into 8 groups of 25.
      seed → train_group, holdout_group (1 group each):
        0 → group 0 / group 1
        1 → group 2 / group 3
        2 → group 4 / group 5
        3 → group 6 / group 7
      The remaining 6 groups (150/category = 600 total) become 'scorecard'.
      All 4 train groups are pairwise disjoint; same for holdout groups.
    """
    import hashlib

    all_cases = [c for c in _load_v4_cases() if c["category"] in _ALL_CATEGORIES]
    split_map: dict[str, str] = {}
    for category in _ALL_CATEGORIES:
        cat_cases = [c for c in all_cases if c["category"] == category]
        cat_cases.sort(key=lambda c: hashlib.md5(c["id"].encode()).hexdigest())
        if split_seed is None:
            for rank, case in enumerate(cat_cases):
                if rank < 25:
                    split_map[case["id"]] = "train"
                elif rank < 50:
                    split_map[case["id"]] = "holdout"
                else:
                    split_map[case["id"]] = "scorecard"
        else:
            # 1 group per split: train=seed*2, holdout=seed*2+1, rest=scorecard
            train_g = split_seed * 2
            holdout_g = split_seed * 2 + 1
            for rank, case in enumerate(cat_cases):
                group = rank // 25  # 0-7
                if group == train_g:
                    split_map[case["id"]] = "train"
                elif group == holdout_g:
                    split_map[case["id"]] = "holdout"
                else:
                    split_map[case["id"]] = "scorecard"
    return split_map


# Default split map (no seed) — built lazily
_SPLIT_MAP: dict[str, str] = {}
# Per-seed split maps — built lazily
_SPLIT_MAP_SEEDED: dict[int, dict[str, str]] = {}


def _case_split(case_id: str, split_seed: int | None = None) -> str:
    """Return 'train', 'holdout', or 'scorecard' for a case_id."""
    global _SPLIT_MAP
    if split_seed is None:
        if not _SPLIT_MAP:
            _SPLIT_MAP = _build_split_map()
        return _SPLIT_MAP.get(case_id, "scorecard")
    if split_seed not in _SPLIT_MAP_SEEDED:
        _SPLIT_MAP_SEEDED[split_seed] = _build_split_map(split_seed)
    return _SPLIT_MAP_SEEDED[split_seed].get(case_id, "scorecard")


def _load_split_cases(
    split: str | None,
    max_cases: int | None = None,
    split_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Load all 4 v4 multi-turn categories filtered to a split.

    Args:
        split: 'train', 'holdout', 'scorecard', or None (all 800 cases).
        max_cases: Optional cap after split filtering.
        split_seed: 0-3 enables 4-fold CV splits (25/25/150 per category).
    """
    cases = [c for c in _load_v4_cases() if c["category"] in _ALL_CATEGORIES]
    if split is not None:
        cases = [c for c in cases if _case_split(c["id"], split_seed) == split]
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


# ---------------------------------------------------------------------------
# Middleware loader — dynamically imports middleware from agent_setup.py
# ---------------------------------------------------------------------------


def _load_middleware(middleware_dir: Path | None) -> list[Any]:
    """Load middleware list from middleware_dir/agent_setup.py if present.

    The file must define a module-level list named `MIDDLEWARE` containing
    LangChain middleware instances. Returns an empty list if not found.
    """
    if middleware_dir is None or not middleware_dir.is_dir():
        return []
    setup_path = middleware_dir / "agent_setup.py"
    if not setup_path.exists():
        return []
    import importlib.util

    middleware_dir_str = str(middleware_dir)
    if middleware_dir_str not in sys.path:
        sys.path.insert(0, middleware_dir_str)
    try:
        # Pre-load custom_middleware.py so that relative imports in agent_setup.py
        # (`from .custom_middleware import X`) and absolute imports both work.
        custom_mw = Path(middleware_dir) / "custom_middleware.py"
        if custom_mw.exists():
            cm_spec = importlib.util.spec_from_file_location(
                "_bfcl_mw_pkg.custom_middleware",
                str(custom_mw),
            )
            if cm_spec and cm_spec.loader:
                cm_mod = importlib.util.module_from_spec(cm_spec)
                sys.modules.setdefault("_bfcl_mw_pkg", type(sys)("_bfcl_mw_pkg"))
                sys.modules["_bfcl_mw_pkg.custom_middleware"] = cm_mod
                sys.modules["custom_middleware"] = cm_mod
                cm_spec.loader.exec_module(cm_mod)  # type: ignore[union-attr]

        pkg_name = "_bfcl_mw_pkg.agent_setup"
        spec = importlib.util.spec_from_file_location(
            pkg_name, setup_path, submodule_search_locations=[]
        )
        if spec is None or spec.loader is None:
            return []
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "_bfcl_mw_pkg"
        sys.modules[pkg_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        raw = list(getattr(module, "MIDDLEWARE", []))
        # LLM sometimes writes MIDDLEWARE = [BFCLFixMiddleware] (class) instead of
        # [BFCLFixMiddleware()] (instance) — instantiate any bare classes defensively.
        return [m() if isinstance(m, type) else m for m in raw]
    except Exception as exc:
        print(f"  [bfcl] warning: failed to load middleware from {setup_path}: {exc}")
        return []


async def score_bfcl_async(
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model_name: str = _DEFAULT_MODEL,
    time_budget_s: float = _TIME_BUDGET_S,
    output_dir: Path | None = None,
    tool_descriptions_dir: Path | None = None,
    middleware_dir: Path | None = None,
    split: str | None = None,
    max_cases: int | None = None,
    case_ids: list[str] | None = None,
    split_seed: int | None = None,
) -> BFCLScore:
    """Run BFCL v4 multi_turn_base cases asynchronously and return aggregated metrics.

    Args:
        system_prompt: System prompt for the inner agent.
        model_name: LangChain model identifier (bedrock/azure/vertex go through LFH).
        time_budget_s: Per-case wall-clock budget for within_time_budget flag.
        output_dir: If set, saves trajectory JSON per case here.
        tool_descriptions_dir: If set, per-class .txt files here override default
            method docstrings as tool descriptions for the inner agent.
        middleware_dir: If set, loads middleware from agent_setup.py in this directory.
        split: 'train', 'holdout', 'scorecard', or None (all 800 cases).
        max_cases: If set, cap the number of cases run (useful for smoke tests).
        case_ids: If set, run only these specific case IDs (overrides split/max_cases).
        split_seed: 0-3 selects a 4-fold CV partition (25/25/150 per category).

    Returns:
        BFCLScore with per-case results and all harness metrics.
    """
    model = _build_model(model_name, temperature=0.0, seed=42)
    cases = _load_split_cases(split, max_cases=max_cases, split_seed=split_seed)
    if case_ids is not None:
        id_set = set(case_ids)
        cases = [c for c in cases if c["id"] in id_set]
    middleware = _load_middleware(middleware_dir)
    loop_start = time.monotonic()

    # Run cases with limited concurrency to avoid 429 rate limits on azure/gpt-5.4-mini
    concurrency = 8
    sem = asyncio.Semaphore(concurrency)

    async def _run_one(case: dict[str, Any]) -> CaseResult:
        async with sem:
            print(
                f"  [bfcl] case {case['id']} "
                f"({len(case['conversation'])} turns, {case['involved_classes']})"
            )
            max_retries = 4
            for attempt in range(max_retries):
                result = await _run_case_async(
                    case,
                    model,
                    system_prompt,
                    model_name,
                    time_budget_s,
                    tool_descriptions_dir=tool_descriptions_dir,
                    middleware=middleware,
                )
                # Retry on infra errors with exponential backoff + jitter
                if result.stuck_type == "infra_error" or (
                    result.stuck_type in ("crash", "exception")
                    and any(s in (result.error or "") for s in _INFRA_ERROR_SIGNALS)
                ):
                    if attempt < max_retries - 1:
                        sleep_s = (2**attempt) * 5 + random.uniform(0, 3)
                        print(
                            f"  [bfcl] {case['id']}: 429 rate limit, retry "
                            f"{attempt + 1}/{max_retries - 1} in {sleep_s:.1f}s"
                        )
                        await asyncio.sleep(sleep_s)
                        continue
                break
            status = "PASS" if result.passed else f"FAIL({result.stuck_type or 'state_diff'})"
            print(
                f"  [bfcl] {result.case_id}: {status} in {result.wall_time_s:.1f}s, "
                f"tokens={result.prompt_tokens}+{result.completion_tokens}"
            )
            if result.stuck_type == "infra_error" and result.error:
                print(f"  [bfcl] infra_error: {result.error[:200]}")
            elif result.stuck_type == "crash" and result.error:
                print(f"  [bfcl] crash error: {result.error}")
            if output_dir is not None and result.trajectory is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                traj_path = output_dir / f"{result.case_id}_trajectory.json"
                traj_path.write_text(json.dumps(result.trajectory.to_dict(), indent=2))
            return result

    results: list[CaseResult] = await asyncio.gather(*[_run_one(c) for c in cases])

    total_time = time.monotonic() - loop_start
    return BFCLScore(results=results, total_wall_time_s=total_time)


def score_bfcl(  # noqa: E302
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model_name: str = _DEFAULT_MODEL,
    time_budget_s: float = _TIME_BUDGET_S,
    output_dir: Path | None = None,
    tool_descriptions_dir: Path | None = None,
    middleware_dir: Path | None = None,
    split: str | None = None,
    max_cases: int | None = None,
) -> BFCLScore:
    """Synchronous wrapper for score_bfcl_async (for non-async callers)."""
    return asyncio.run(
        score_bfcl_async(
            system_prompt=system_prompt,
            model_name=model_name,
            time_budget_s=time_budget_s,
            output_dir=output_dir,
            tool_descriptions_dir=tool_descriptions_dir,
            middleware_dir=middleware_dir,
            split=split,
            max_cases=max_cases,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score BFCL v3 cases")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file:
        prompt = Path(args.system_prompt_file).read_text()

    out = Path(args.output_dir) if args.output_dir else None
    score = score_bfcl(model_name=args.model, system_prompt=prompt, output_dir=out)
    print(json.dumps(score.to_dict(), indent=2))
