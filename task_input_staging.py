"""Optional default-only Kanban task-input staging tool.

This is a transport helper, not a Jev classifier or authority boundary. It
binds provenance to the generated task ID, persists a versioned input marker,
reads it back while the task is blocked, then asks Kanban to unblock the task.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import shlex
from typing import Any, Callable

TOOL_NAME = "jev_stage_task_input"
MARKER = "jev_decision_input_json:"
MAX_DECISION_INPUT_BYTES = 64_000
MAX_TASK_BODY_BYTES = 192_000
MAX_DECISION_TARGETS = 64
MAX_DECISION_SIGNALS = 20

_DECISION_INPUT_FIELDS = frozenset({
    "schema_version", "scope", "objective", "target_candidates", "requested_outcome",
    "action_mode", "allowed_actions", "forbidden_actions", "decision_points", "freshness",
    "deduplication", "baseline", "completion_conditions", "prior_context", "signals",
    "feedback", "output_requirements", "uncertainty", "provenance", "domain_extension",
})
_STRING_FIELDS = ("scope", "objective", "requested_outcome", "action_mode")
_ACTION_LIST_FIELDS = ("allowed_actions", "forbidden_actions", "completion_conditions")
_GENERAL_LIST_FIELDS = ("decision_points", "signals", "feedback", "uncertainty")
_OBJECT_FIELDS = ("freshness", "deduplication", "prior_context", "provenance")
_TASK_FIELDS = frozenset({
    "title", "assignee", "body", "parents", "tenant", "priority", "triage",
    "workspace_kind", "workspace_path", "project", "skills", "max_runtime_seconds",
    "goal_mode", "goal_max_turns", "completion_contract", "model", "provider",
})
_BLOCKING_VALUES = {"unknown", "missing", "malformed", "truncated", "stale", "unresolved", "incomplete", "unverified"}
_ALLOWED_NULL_KEYS = {"domain_extension", "valid_until", "truncation_cause"}


def _array(items: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "array", "items": items or {}}


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": True,
    }


_STRING = {"type": "string"}
_INPUT_PROPERTIES: dict[str, Any] = {
    "schema_version": {"type": "integer", "enum": [1]},
    "scope": _STRING,
    "objective": _STRING,
    "target_candidates": _array({"anyOf": [{"type": "string"}, {"type": "object"}]}),
    "requested_outcome": _STRING,
    "action_mode": _STRING,
    "allowed_actions": _array(_STRING),
    "forbidden_actions": _array(_STRING),
    "decision_points": _array(),
    "freshness": _object(
        {"status": _STRING, "observed_at": _STRING, "valid_until": {"type": ["string", "null"]}, "rule": _STRING},
        ["status", "observed_at", "rule"],
    ),
    "deduplication": _object({"status": _STRING, "basis": _STRING, "checked_at": _STRING}, ["status", "basis"]),
    "baseline": {"anyOf": [{"type": "object"}, {"type": "array"}]},
    "completion_conditions": _array(_STRING),
    "prior_context": {"type": "object"},
    "signals": _array(),
    "feedback": _array(),
    "output_requirements": {"anyOf": [{"type": "object"}, {"type": "array"}]},
    "uncertainty": _array(),
    "provenance": _object(
        {"task_id": {"type": ["string", "null"]}, "source": _STRING, "captured_at": _STRING, "source_references": _array(_STRING)},
        ["source", "captured_at", "source_references"],
    ),
    "domain_extension": {"anyOf": [{"type": "object"}, {"type": "null"}]},
}

TASK_INPUT_STAGING_SCHEMA: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Optional default-profile helper. Create a Kanban task blocked, bind a complete "
        "schema_version=1 jev_decision_input_json object to its generated task ID, persist "
        "and validate the marker by readback, then unblock. This is transport only: it does "
        "not evaluate Jev output, grant authority, or change default behavior. Requires the "
        "standard Kanban tools and terminal tool to be available."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "object",
                "description": "Task creation fields. The helper always creates it blocked; status and idempotency are controlled internally.",
                "properties": {
                    "title": _STRING,
                    "assignee": _STRING,
                    "body": _STRING,
                    "parents": _array(_STRING),
                    "tenant": _STRING,
                    "priority": {"type": "integer"},
                    "triage": {"type": "boolean"},
                    "workspace_kind": {"type": "string", "enum": ["scratch", "dir", "worktree"]},
                    "workspace_path": _STRING,
                    "project": _STRING,
                    "skills": _array(_STRING),
                    "max_runtime_seconds": {"type": "integer"},
                    "goal_mode": {"type": "boolean"},
                    "goal_max_turns": {"type": "integer"},
                    "completion_contract": _STRING,
                    "model": _STRING,
                    "provider": _STRING,
                },
                "required": ["title", "assignee"],
                "additionalProperties": False,
            },
            "decision_input": {
                "type": "object",
                "description": "Complete schema_version=1 input. The helper changes only provenance.task_id.",
                "properties": _INPUT_PROPERTIES,
                "required": sorted(_DECISION_INPUT_FIELDS),
                "additionalProperties": False,
            },
            "board": {"type": "string", "description": "Optional Kanban board slug."},
            "request_key": {
                "type": "string",
                "description": "Optional retry-stable key. Use a different value only for an intentionally new otherwise-identical task.",
            },
        },
        "required": ["task", "decision_input"],
        "additionalProperties": False,
    },
}


def _strict_json_clone(value: Any, path: str = "$", *, depth: int = 0) -> Any:
    """Copy JSON-only data and reject non-finite numbers or Python-only values."""
    if depth > 64:
        raise ValueError(f"{path} exceeds the supported JSON nesting depth")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_strict_json_clone(item, f"{path}[{index}]", depth=depth + 1) for index, item in enumerate(value)]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} object keys must be strings")
        return {key: _strict_json_clone(item, f"{path}.{key}", depth=depth + 1) for key, item in value.items()}
    raise ValueError(f"{path} contains a value that is not strict JSON")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _loads_strict_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_object_without_duplicate_keys, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON number")))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _timestamp(value: Any, name: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"decision_input.{name} must be a timezone-aware timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"decision_input.{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"decision_input.{name} must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _reject_blocking_unknowns(value: Any, path: str, *, allow_null_task_id: bool) -> None:
    if isinstance(value, dict):
        for key in ("status", "state", "resolution_status"):
            marker = value.get(key)
            if isinstance(marker, str) and marker.strip().lower() in _BLOCKING_VALUES:
                raise ValueError(f"{path}.{key} is unresolved")
        if any(value.get(key) is False for key in ("resolved", "loaded")):
            raise ValueError(f"{path} is explicitly unresolved")
        if value.get("truncated") is True or value.get("malformed") is True or value.get("stale") is True:
            raise ValueError(f"{path} is explicitly incomplete")
        for key, child in value.items():
            if child is None:
                nullable = key in _ALLOWED_NULL_KEYS or (key == "task_id" and allow_null_task_id)
                if not nullable:
                    raise ValueError(f"{path}.{key} is missing")
            _reject_blocking_unknowns(child, f"{path}.{key}", allow_null_task_id=allow_null_task_id)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if child is None:
                raise ValueError(f"{path}[{index}] is missing")
            _reject_blocking_unknowns(child, f"{path}[{index}]", allow_null_task_id=allow_null_task_id)
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _BLOCKING_VALUES or normalized.startswith(tuple(f"{item}:" for item in _BLOCKING_VALUES)):
            raise ValueError(f"{path} contains an unresolved value")


def _validate_decision_input(value: Any, *, require_bound: bool = False) -> dict[str, Any]:
    cloned = _strict_json_clone(value)
    if not isinstance(cloned, dict):
        raise ValueError("decision_input must be a complete JSON object")
    missing = sorted(_DECISION_INPUT_FIELDS - set(cloned))
    extra = sorted(set(cloned) - _DECISION_INPUT_FIELDS)
    if missing:
        raise ValueError("decision_input is missing required fields")
    if extra:
        raise ValueError("decision_input contains unsupported fields")
    packed = _canonical_json(cloned)
    if len(packed.encode("utf-8")) > MAX_DECISION_INPUT_BYTES:
        raise ValueError("decision_input exceeds the inline byte limit")
    if type(cloned.get("schema_version")) is not int or cloned["schema_version"] != 1:
        raise ValueError("decision_input.schema_version must be 1")
    for field in _STRING_FIELDS:
        if not isinstance(cloned.get(field), str) or not cloned[field].strip():
            raise ValueError(f"decision_input.{field} must be a non-empty string")
    targets = cloned.get("target_candidates")
    if not isinstance(targets, list) or len(targets) > MAX_DECISION_TARGETS or any(not isinstance(item, (str, dict)) for item in targets):
        raise ValueError("decision_input.target_candidates must be a bounded list of strings or objects")
    for field in _ACTION_LIST_FIELDS:
        rows = cloned.get(field)
        if not isinstance(rows, list) or not rows or any(not isinstance(row, str) or not row.strip() for row in rows):
            raise ValueError(f"decision_input.{field} must contain non-empty strings")
    for field in _GENERAL_LIST_FIELDS:
        if not isinstance(cloned.get(field), list):
            raise ValueError(f"decision_input.{field} must be a list")
    for field in _OBJECT_FIELDS:
        if not isinstance(cloned.get(field), dict):
            raise ValueError(f"decision_input.{field} must be an object")
    if not isinstance(cloned.get("baseline"), (dict, list)) or not isinstance(cloned.get("output_requirements"), (dict, list)):
        raise ValueError("decision_input.baseline and output_requirements must be objects or lists")
    if cloned.get("domain_extension") is not None and not isinstance(cloned["domain_extension"], dict):
        raise ValueError("decision_input.domain_extension must be an object or null")

    now = dt.datetime.now(dt.timezone.utc)
    freshness = cloned["freshness"]
    freshness_status = freshness.get("status")
    if freshness_status not in {"fresh", "not_time_sensitive"}:
        raise ValueError("decision_input.freshness.status is unresolved")
    observed_at = _timestamp(freshness.get("observed_at"), "freshness.observed_at")
    valid_until = freshness.get("valid_until")
    if freshness_status == "fresh":
        expiry = _timestamp(valid_until, "freshness.valid_until")
        if expiry < observed_at or now > expiry:
            raise ValueError("decision_input.freshness is stale")
    elif valid_until is not None:
        expiry = _timestamp(valid_until, "freshness.valid_until")
        if expiry < observed_at or now > expiry:
            raise ValueError("decision_input.freshness is stale")
    if observed_at > now or not isinstance(freshness.get("rule"), str) or not freshness["rule"].strip():
        raise ValueError("decision_input.freshness must include a valid observation and rule")

    deduplication = cloned["deduplication"]
    dedup_status = deduplication.get("status")
    if dedup_status not in {"checked", "not_applicable"} or not isinstance(deduplication.get("basis"), str) or not deduplication["basis"].strip():
        raise ValueError("decision_input.deduplication must be checked or not_applicable with a basis")
    checked_at = deduplication.get("checked_at")
    if dedup_status == "checked":
        if _timestamp(checked_at, "deduplication.checked_at") > now:
            raise ValueError("decision_input.deduplication.checked_at is in the future")

    if not cloned["completion_conditions"]:
        raise ValueError("decision_input.completion_conditions cannot be empty")
    if len(cloned["signals"]) + len(cloned["feedback"]) > MAX_DECISION_SIGNALS:
        raise ValueError("decision_input signals and feedback exceed the combined limit")
    provenance = cloned["provenance"]
    if not isinstance(provenance.get("source"), str) or not provenance["source"].strip():
        raise ValueError("decision_input.provenance.source is required")
    captured_at = _timestamp(provenance.get("captured_at"), "provenance.captured_at")
    if captured_at > now:
        raise ValueError("decision_input.provenance.captured_at is in the future")
    references = provenance.get("source_references")
    if not isinstance(references, list) or not references or any(not isinstance(ref, str) or not ref.strip() for ref in references):
        raise ValueError("decision_input.provenance.source_references must contain non-empty strings")
    task_id = provenance.get("task_id")
    if require_bound and (not isinstance(task_id, str) or not task_id.strip()):
        raise ValueError("decision_input.provenance.task_id must be bound")
    if not require_bound and task_id is not None and not isinstance(task_id, str):
        raise ValueError("decision_input.provenance.task_id must be a string or null")

    _reject_blocking_unknowns(cloned, "decision_input", allow_null_task_id=not require_bound)
    return cloned


def _failure(error: str, *, task_id: str | None = None, status: str | None = None) -> str:
    result: dict[str, Any] = {"ok": False, "error": error}
    if task_id:
        result["task_id"] = task_id
    if status:
        result["status"] = status
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _decode_result(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        result = _loads_strict_json(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return result if isinstance(result, dict) else None


def _dispatch(dispatch_tool: Callable[[str, dict[str, Any]], Any], name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return _decode_result(dispatch_tool(name, args))
    except Exception:
        return None


def _marker_lines(body: str) -> list[str]:
    return [line for line in body.splitlines() if line.startswith(MARKER)]


def _extract_persisted(body: str) -> dict[str, Any] | None:
    if body.count(MARKER) != 1:
        return None
    lines = _marker_lines(body)
    if len(lines) != 1:
        return None
    raw = lines[0][len(MARKER):].strip()
    try:
        return _validate_decision_input(_loads_strict_json(raw), require_bound=True)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _body_with_marker(body: str, decision_input: dict[str, Any]) -> str:
    if MARKER in body:
        raise ValueError("task body contains the reserved decision-input marker")
    separator = "" if not body else ("" if body.endswith("\n\n") else ("\n" if body.endswith("\n") else "\n\n"))
    updated = body + separator + MARKER + " " + _canonical_json(decision_input) + "\n"
    if len(updated.encode("utf-8")) > MAX_TASK_BODY_BYTES:
        raise ValueError("task body with decision input exceeds the non-truncating size limit")
    return updated


def _terminal_edit_command(task_id: str, body: str, board: str | None) -> str:
    argv = ["hermes", "kanban"]
    if board:
        argv.extend(["--board", board])
    argv.extend(["edit", task_id, "--body", body])
    return " ".join(shlex.quote(part) for part in argv)


def _read_task(dispatch_tool: Callable[[str, dict[str, Any]], Any], task_id: str, board: str | None) -> dict[str, Any] | None:
    args: dict[str, Any] = {"task_id": task_id}
    if board:
        args["board"] = board
    payload = _dispatch(dispatch_tool, "kanban_show", args)
    task = payload.get("task") if isinstance(payload, dict) else None
    return task if isinstance(task, dict) else None


def _bound_input(raw_input: dict[str, Any], task_id: str) -> dict[str, Any]:
    bound = _strict_json_clone(raw_input)
    provenance = bound.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("decision_input.provenance must be an object")
    provenance["task_id"] = task_id
    return _validate_decision_input(bound, require_bound=True)


def _validate_task(task: Any) -> dict[str, Any]:
    if not isinstance(task, dict) or set(task) - _TASK_FIELDS:
        raise ValueError("task must be an object with supported Kanban task fields")
    cloned = _strict_json_clone(task)
    for field in ("title", "assignee"):
        if not isinstance(cloned.get(field), str) or not cloned[field].strip():
            raise ValueError(f"task.{field} is required")
    if cloned.get("body") is not None and not isinstance(cloned.get("body"), str):
        raise ValueError("task.body must be a string")
    cloned["body"] = cloned.get("body") or ""
    for field in ("tenant", "workspace_path", "project", "completion_contract", "model", "provider"):
        if field in cloned and not isinstance(cloned[field], str):
            raise ValueError(f"task.{field} must be a string")
    for field in ("parents", "skills"):
        if field in cloned and (not isinstance(cloned[field], list) or any(not isinstance(item, str) for item in cloned[field])):
            raise ValueError(f"task.{field} must be an array of strings")
    for field in ("priority", "max_runtime_seconds", "goal_max_turns"):
        if field in cloned and (type(cloned[field]) is not int or cloned[field] < 1):
            raise ValueError(f"task.{field} must be a positive integer")
    for field in ("triage", "goal_mode"):
        if field in cloned and type(cloned[field]) is not bool:
            raise ValueError(f"task.{field} must be a boolean")
    if cloned.get("workspace_kind") not in (None, "scratch", "dir", "worktree"):
        raise ValueError("task.workspace_kind is unsupported")
    return cloned


def stage_task_input(params: Any, dispatch_tool: Callable[[str, dict[str, Any]], Any]) -> str:
    """Create blocked, bind and verify one input marker, then unblock it once."""
    try:
        request = _strict_json_clone(params)
        if not isinstance(request, dict) or set(request) - {"task", "decision_input", "board", "request_key"}:
            return _failure("request contains unsupported fields")
        task = _validate_task(request.get("task"))
        if MARKER in task["body"]:
            return _failure("task.body contains the reserved decision-input marker")
        decision_input = _validate_decision_input(request.get("decision_input"))
        board = request.get("board")
        if board is not None and (not isinstance(board, str) or not board.strip()):
            return _failure("board must be a non-empty string when supplied")
        request_key = request.get("request_key", "")
        if not isinstance(request_key, str):
            return _failure("request_key must be a string")
        if not callable(dispatch_tool):
            return _failure("Kanban tool dispatch is unavailable")
    except (TypeError, ValueError):
        return _failure("request or decision_input is not a valid complete input")

    fingerprint = {
        "board": board,
        "request_key": request_key,
        "task": task,
        "decision_input": decision_input,
    }
    idempotency_key = "jev-input-stage-" + hashlib.sha256(_canonical_json(fingerprint).encode("utf-8")).hexdigest()
    create_args = dict(task)
    create_args.update({"initial_status": "blocked", "idempotency_key": idempotency_key})
    if board:
        create_args["board"] = board

    created = _dispatch(dispatch_tool, "kanban_create", create_args)
    if not isinstance(created, dict) or created.get("ok") is not True:
        return _failure("Kanban task creation failed; no task was unblocked")
    task_id = created.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        return _failure("Kanban task creation returned no task id; retry with identical input to avoid duplicates")
    try:
        bound = _bound_input(decision_input, task_id)
    except (TypeError, ValueError):
        return _failure("decision_input could not be bound to the created task id", task_id=task_id, status="blocked")

    current = _read_task(dispatch_tool, task_id, board)
    if current is None:
        return _failure("created task could not be read back; it was not unblocked", task_id=task_id, status="blocked")
    status = current.get("status") if isinstance(current.get("status"), str) else None
    stored_body = current.get("body") or ""
    if not isinstance(stored_body, str):
        return _failure("created task body could not be read as text", task_id=task_id, status=status)
    if current.get("title") != task["title"] or current.get("assignee") != task["assignee"]:
        return _failure("idempotent task readback does not match the requested title and assignee", task_id=task_id, status=status)

    marker_present = MARKER in stored_body
    if marker_present:
        persisted = _extract_persisted(stored_body)
        if persisted != bound:
            return _failure("persisted decision input is invalid or differs from the task-bound request", task_id=task_id, status=status)
        if status != "blocked":
            if status in {"todo", "ready", "running", "review", "done"}:
                return json.dumps({"ok": True, "task_id": task_id, "status": status, "already_staged": True}, sort_keys=True)
            return _failure("staged task is not in a resumable state", task_id=task_id, status=status)
    else:
        if status != "blocked":
            return _failure("created task was not blocked before staging; it was not edited or unblocked", task_id=task_id, status=status)
        if stored_body != task["body"]:
            return _failure("blocked task body changed before staging; it was not overwritten", task_id=task_id, status=status)
        try:
            updated_body = _body_with_marker(stored_body, bound)
        except ValueError:
            return _failure("task body cannot safely hold the decision-input marker", task_id=task_id, status=status)
        terminal = _dispatch(dispatch_tool, "terminal", {"command": _terminal_edit_command(task_id, updated_body, board)})
        if not isinstance(terminal, dict) or terminal.get("exit_code") != 0:
            return _failure("Kanban body edit failed; task remains blocked", task_id=task_id, status="blocked")
        verified = _read_task(dispatch_tool, task_id, board)
        if verified is None or verified.get("status") != "blocked":
            verified_status = verified.get("status") if isinstance(verified, dict) and isinstance(verified.get("status"), str) else status
            return _failure("task did not remain blocked for persisted-input readback", task_id=task_id, status=verified_status)
        verified_body = verified.get("body") or ""
        persisted = _extract_persisted(verified_body) if isinstance(verified_body, str) else None
        if persisted != bound or len(_marker_lines(verified_body)) != 1:
            return _failure("persisted decision input failed exact readback validation; task remains blocked", task_id=task_id, status="blocked")

    unblock_args: dict[str, Any] = {"task_id": task_id}
    if board:
        unblock_args["board"] = board
    _dispatch(dispatch_tool, "kanban_unblock", unblock_args)
    final = _read_task(dispatch_tool, task_id, board)
    if final is None:
        return _failure("unblock was requested but final task state could not be read back", task_id=task_id, status="unknown")
    final_status = final.get("status") if isinstance(final.get("status"), str) else None
    final_body = final.get("body") or ""
    persisted_final = _extract_persisted(final_body) if isinstance(final_body, str) else None
    if persisted_final != bound or len(_marker_lines(final_body)) != 1:
        return _failure("final task readback no longer matches the validated decision input", task_id=task_id, status=final_status)
    if final_status not in {"todo", "ready", "running", "review", "done"}:
        return _failure("task did not reach a resumable state after unblock", task_id=task_id, status=final_status)
    return json.dumps({"ok": True, "task_id": task_id, "status": final_status, "already_staged": marker_present}, sort_keys=True)


def register_task_input_staging(ctx: Any) -> None:
    """Register the distinct tool; its caller enforces default-only opt-in."""
    def handler(args: dict[str, Any], **_: Any) -> str:
        return stage_task_input(args, ctx.dispatch_tool)

    ctx.register_tool(
        name=TOOL_NAME,
        toolset="jev_route_screening",
        schema=TASK_INPUT_STAGING_SCHEMA,
        handler=handler,
        description=TASK_INPUT_STAGING_SCHEMA["description"],
    )
