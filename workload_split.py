"""Advisory workload-split signal for Default before Kanban creation or delegation.

Decision logic lives in code (``decide`` and ``decide_model``); Jev only answers
Noul judgments.  Three size judgments feed the split signal and two difficulty
judgments feed the model signal.  An explicit target count above the threshold
answers the split signal without Jev.
The result is a signal for Default.  It never blocks creation or delegation and
never selects a model: provider failure or an unsettled answer returns
``proceed`` and ``model_signal: not_judged`` or ``none``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from . import bridge

ROUTE = "jev.workload_split"
TOOL_NAME = "jev_workload_split"
LOG_NAME = "workload-split.jsonl"

# Rule thresholds (master-approved plan 2026-10-30: explicit count > 3 -> split).
COUNT_SPLIT_THRESHOLD = 3
NOUL_YES = 0.7
NOUL_NO = 0.3

EXECUTORS = ("kanban_research", "kanban_docs", "delegation", "other")
MAX_SUMMARY_CHARS = 1200
MAX_FIELD_CHARS = 200
MAX_LABEL_CHARS = 120

# All question wording in one place (typesafe-ai guidance).
QUESTIONS: dict[str, dict[str, Any]] = {
    "many_items": {
        "type": "noul",
        "instructions": (
            "Does the work in `request.summary` require collecting, checking, or editing "
            "more than ten separate items, such as organizations, sources, pages, or files?"
        ),
        "criteria": {
            "true": "The work covers a list or area whose items must each be found or checked one by one, and there are likely more than ten.",
            "false": "The work concerns one item or a handful of named items.",
        },
    },
    "exceeds_runtime": {
        "type": "noul",
        "instructions": (
            "Would a careful researcher using web search and file tools need longer than "
            "`request.runtime_limit_minutes` minutes to finish the work in `request.summary`?"
        ),
        "criteria": {
            "true": "Finishing the work carefully would take longer than the stated limit.",
            "false": "The work can be finished carefully within the stated limit.",
        },
    },
    "divisible": {
        "type": "noul",
        "instructions": (
            "Can the work in `request.summary` be divided into parts, for example by area, type, "
            "or file, where each part can be finished and accepted on its own?"
        ),
        "criteria": {
            "true": "Each part would be a usable result by itself, such as one area's list or one file's fix.",
            "false": "The parts only make sense together, such as one argument or one combined judgment.",
        },
    },
    "cross_source_synthesis": {
        "type": "noul",
        "instructions": (
            "Does finishing the work in `request.summary` require weighing what several separate "
            "sources say against each other to reach one conclusion?"
        ),
        "criteria": {
            "true": "The conclusion depends on comparing or reconciling different documents, such as a law, an official notice, statistics, and reports that must be read against each other.",
            "false": "The result can be taken from one source, or each source is recorded separately without being weighed against the others.",
        },
    },
    "quantitative_analysis": {
        "type": "noul",
        "instructions": (
            "Does finishing the work in `request.summary` require calculating or estimating "
            "numbers that no source states directly?"
        ),
        "criteria": {
            "true": "The worker must compute, estimate, or model figures, such as totals, rates, projections, or cost comparisons.",
            "false": "Any numbers needed can be copied from a source as stated.",
        },
    },
}

SIZE_KEYS = ("many_items", "exceeds_runtime", "divisible")
MODEL_KEYS = ("cross_source_synthesis", "quantitative_analysis")


def _clean(value: Any, name: str, limit: int, *, required: bool) -> str:
    text = bridge._safe_text(value, name, limit, allow_empty=not required)
    return text


def validate_args(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise bridge.BridgeValidationError("arguments must be an object")
    allowed = {"request_summary", "target_count", "deliverable", "runtime_limit_minutes", "executor", "label"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise bridge.BridgeValidationError(f"unknown fields: {', '.join(unknown)}")
    summary = _clean(raw.get("request_summary"), "request_summary", MAX_SUMMARY_CHARS, required=True)
    count = raw.get("target_count")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1000):
        raise bridge.BridgeValidationError("target_count must be an integer 1..1000 or omitted")
    minutes = raw.get("runtime_limit_minutes", 25)
    if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 600:
        raise bridge.BridgeValidationError("runtime_limit_minutes must be an integer 1..600")
    executor = raw.get("executor", "other")
    if executor not in EXECUTORS:
        raise bridge.BridgeValidationError(f"executor must be one of {list(EXECUTORS)}")
    return {
        "request_summary": summary,
        "target_count": count,
        "deliverable": _clean(raw.get("deliverable"), "deliverable", MAX_FIELD_CHARS, required=False),
        "runtime_limit_minutes": minutes,
        "executor": executor,
        "label": _clean(raw.get("label"), "label", MAX_LABEL_CHARS, required=False),
    }


def build_request(clean: Mapping[str, Any], *, model: str) -> dict[str, Any]:
    request: dict[str, Any] = {
        "summary": clean["request_summary"],
        "runtime_limit_minutes": clean["runtime_limit_minutes"],
    }
    if clean.get("deliverable"):
        request["deliverable"] = clean["deliverable"]
    return {"model": model, "state": {"request": request}, "questions": dict(QUESTIONS)}


def parse_response(value: Any, *, transport: str | None = None) -> dict[str, float]:
    del transport
    try:
        answers = value["answers"]
        out: dict[str, float] = {}
        for key in SIZE_KEYS:
            noul = answers[key]["noul"]
            if isinstance(noul, bool) or not isinstance(noul, (int, float)) or not 0 <= float(noul) <= 1:
                raise ValueError
            out[key] = float(noul)
    except (KeyError, TypeError, ValueError):
        raise bridge.JevRequestError("malformed_response")
    # The difficulty judgments are an added advisory field: a response without
    # them keeps the split signal and leaves the model signal unjudged.
    for key in MODEL_KEYS:
        try:
            noul = answers[key]["noul"]
        except (KeyError, TypeError):
            continue
        if not isinstance(noul, bool) and isinstance(noul, (int, float)) and 0 <= float(noul) <= 1:
            out[key] = float(noul)
    return out


def _band(value: float) -> str:
    if value >= NOUL_YES:
        return "yes"
    if value <= NOUL_NO:
        return "no"
    return "unsettled"


def decide(nouls: Mapping[str, float]) -> tuple[str, str]:
    """Truth table: (signal, reason_code). Only a clear yes recommends a split."""
    large = _band(nouls["many_items"]) == "yes" or _band(nouls["exceeds_runtime"]) == "yes"
    small = _band(nouls["many_items"]) == "no" and _band(nouls["exceeds_runtime"]) == "no"
    divisible = _band(nouls["divisible"])
    if large and divisible == "yes":
        return "recommend_split", "jev_large_and_divisible"
    if large and divisible == "no":
        return "consider_runtime", "jev_large_not_divisible"
    if small:
        return "proceed", "jev_small"
    return "proceed", "jev_unsettled"


def decide_model(nouls: Mapping[str, float]) -> tuple[str, str]:
    """Truth table: (model_signal, model_reason_code).

    A stronger model is worth considering only for work that is one combined
    judgment: divisible work is handled by splitting instead.
    """
    if any(key not in nouls for key in MODEL_KEYS):
        return "not_judged", "jev_model_judgment_missing"
    if _band(nouls["divisible"]) == "yes":
        return "none", "jev_divisible"
    synthesis = _band(nouls["cross_source_synthesis"]) == "yes"
    quantitative = _band(nouls["quantitative_analysis"]) == "yes"
    if synthesis and quantitative:
        return "consider_stronger_model", "jev_synthesis_and_quantitative"
    if synthesis:
        return "consider_stronger_model", "jev_cross_source_synthesis"
    if quantitative:
        return "consider_stronger_model", "jev_quantitative_analysis"
    return "none", "jev_no_hard_judgment"


_LOG_LOCK = threading.Lock()


def _append_log(root: Path | None, row: Mapping[str, Any]) -> None:
    if root is None:
        return
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with _LOG_LOCK, (root / LOG_NAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        bridge.log.warning("workload-split log append skipped", exc_info=True)


def workload_split_tool(
    args: Any,
    *,
    ctx: Any,
    post_fn: Callable[[bridge.ProviderSpec, Mapping[str, Any], float], Any] | None = None,
    log_root: str | Path | None = None,
) -> str:
    try:
        clean = validate_args(args)
    except (bridge.BridgeValidationError, TypeError, ValueError) as exc:
        return bridge._json({"signal": "invalid_input", "reason_code": "invalid_input", "error": str(exc)[:200]})
    root_value = log_root if log_root is not None else (bridge._config_value(ctx, "bridge_dir", "") or os.environ.get("HERMES_JEV_BRIDGE_DIR", ""))
    root = Path(root_value).expanduser() if root_value else None
    request_id = bridge._new_lifecycle_id()
    row: dict[str, Any] = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "request_id": request_id,
        "executor": clean["executor"],
        "label": clean["label"],
        "target_count": clean["target_count"],
        "runtime_limit_minutes": clean["runtime_limit_minutes"],
        "summary_sha256": hashlib.sha256(clean["request_summary"].encode("utf-8")).hexdigest()[:16],
    }
    count = clean["target_count"]
    if count is not None and count > COUNT_SPLIT_THRESHOLD:
        result = {"signal": "recommend_split", "reason_code": "count_over_threshold", "route": "count_rule",
                  "model_signal": "not_judged", "model_reason_code": "count_rule_split", "request_id": request_id}
        _append_log(root, {**row, **result})
        return bridge._json(result)

    nouls: dict[str, float] | None = None
    failure = "unknown"
    try:
        primary = bridge._configured_provider(ctx, "primary", bridge.PRIMARY_PROVIDER)
        fallback = bridge._configured_provider(ctx, "fallback", bridge.FALLBACK_PROVIDER)
        timeout = bridge._configured_timeout(ctx)
        nouls = bridge.request_decision_with_fallback(
            build_request(clean, model=primary.model),
            primary=primary, fallback=fallback, timeout=timeout,
            post_fn=post_fn, parser=parse_response, route=ROUTE, request_id=request_id,
        )
    except bridge.JevRequestError as exc:
        failure = exc.reason
    except Exception as exc:  # fail open to Default; never expose provider payloads
        failure = type(exc).__name__
    # A stated count within the threshold settles the split signal by rule;
    # Jev is still asked so the difficulty judgments are available.
    by_count = count is not None
    if nouls is None:
        result = {"signal": "proceed", "reason_code": "count_within_threshold" if by_count else "jev_unavailable",
                  "route": "count_rule" if by_count else "jev",
                  "model_signal": "not_judged", "model_reason_code": "jev_unavailable",
                  "failure": str(failure)[:64], "request_id": request_id}
        _append_log(root, {**row, **result})
        return bridge._json(result)
    signal, reason = ("proceed", "count_within_threshold") if by_count else decide(nouls)
    model_signal, model_reason = decide_model(nouls)
    result = {
        "signal": signal,
        "reason_code": reason,
        "route": "count_rule" if by_count else "jev",
        "model_signal": model_signal,
        "model_reason_code": model_reason,
        "nouls": {key: round(value, 3) for key, value in nouls.items()},
        "request_id": request_id,
    }
    _append_log(root, {**row, **result})
    bridge._lifecycle_log("delivered", ROUTE, request_id, destination="jev_workload_split_tool_result", status=signal)
    return bridge._json(result)


def tool_schema() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": (
            "Advisory split and model signals before creating a research/docs Kanban card or a delegate_task child. "
            "A target_count above 3 answers recommend_split by rule without Jev; a target_count of 3 or less answers proceed. "
            "Otherwise Jev judges size and divisibility. signal: recommend_split, consider_runtime, proceed. "
            "model_signal: consider_stronger_model when the work is one combined judgment that needs "
            "cross-source synthesis or quantitative analysis; none; not_judged when Jev was not asked or failed. "
            "Default decides both; this tool never blocks creation and never selects a model, and failure returns proceed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "request_summary": {"type": "string", "description": "What the worker will do, in one or two sentences. No secrets or personal data."},
                "target_count": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Number of explicit targets (files, regions, items) when the request states them. Omit when unknown."},
                "deliverable": {"type": "string", "description": "Short description of the output, e.g. 'list of organizations with URLs'."},
                "runtime_limit_minutes": {"type": "integer", "minimum": 1, "maximum": 600, "default": 25},
                "executor": {"type": "string", "enum": list(EXECUTORS)},
                "label": {"type": "string", "description": "Card title or short label, used later to match outcomes."},
            },
            "required": ["request_summary", "executor"],
            "additionalProperties": False,
        },
    }


def register_workload_split(ctx: Any) -> None:
    ctx.register_tool(
        name=TOOL_NAME,
        toolset=TOOL_NAME,
        schema=tool_schema(),
        handler=lambda args, **kwargs: workload_split_tool(args, ctx=ctx),
        description="Advisory workload-split and model signals before Kanban creation or delegation; never blocks.",
    )
