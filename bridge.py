from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .point_state import (
    POINT_STATE_CONTRACT_VERSION,
    ControlEffect,
    PointDecision,
    PointIdentity,
    PointStatePolicy,
    build_point_state_request,
    parse_point_state_response,
)

log = logging.getLogger(__name__)

# Schema v1 remains readable for the already-installed synthetic pilot. New
# worker-hook checkpoints use the common v2 worker_checkpoint envelope.
LEGACY_SCHEMA_VERSION = 1
SCHEMA_VERSION = 2
TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-latest"
TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_MODEL = "typesafe/jev-1.13"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
# Compatibility names for the original OpenRouter-only bridge.
ENDPOINT = OPENROUTER_ENDPOINT
MODEL = OPENROUTER_MODEL
FALLBACK_HTTP_STATUSES = frozenset({401, 403, 408, 429})
LABELS = ("progressing", "stalled", "scope_drift", "completion_candidate", "insufficient_information")
CRITERIA = {
    "progressing": "新しい証拠が増え、必要な確認が前進している。",
    "stalled": "同じ確認を繰り返し、必要な証拠が増えていない。",
    "scope_drift": "次の行動が依頼や完了条件の範囲を外れている。",
    "completion_candidate": "完了条件に必要な証拠がそろっている候補である。",
    "insufficient_information": "識別子、観測、または次の判別材料が不足している。",
}
MAX_EVENT_BYTES = 16_000
MAX_LIVE_CALLS_PER_RUN = 3
MAX_CALLS_PER_RUN = 3
RESERVATION_USD = Decimal("0.005")
REQUEST_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    endpoint: str
    api_key_env: str
    model: str
    transport: str


PRIMARY_PROVIDER = ProviderSpec(
    name="typesafe",
    endpoint=TYPESAFE_ENDPOINT,
    api_key_env=TYPESAFE_API_KEY_ENV,
    model=TYPESAFE_MODEL,
    transport="typesafe-system-one",
)
FALLBACK_PROVIDER = ProviderSpec(
    name="openrouter",
    endpoint=OPENROUTER_ENDPOINT,
    api_key_env=OPENROUTER_API_KEY_ENV,
    model=OPENROUTER_MODEL,
    transport="openrouter-decisions",
)
WORKER_CATEGORY = "worker_checkpoint"
LEGACY_CATEGORY = "ops_checkpoint"
# Kept as a compatibility name for callers/tests of the original pilot.
ALLOWED_CATEGORY = WORKER_CATEGORY
JST = ZoneInfo("Asia/Tokyo")
MIN_COMPLETED_TOOLS = 3
CHECKPOINT_CADENCE_SECONDS = 120.0
MAX_RECENT_STEPS = 6
MAX_CRITERIA = 8
MAX_REFS = 8
MAX_TEXT = 600
TRIGGER_SCHEMA_VERSION = 1
TRIGGER_STATES = {"pending_review", "intervention_requested", "no_intervention"}
INTERVENTION_CATEGORIES = {"under_scoped", "scope_drift", "evidence_mismatch", "repeated_stall", "completion_candidate"}
UNMEASURED_WORKER_CONCLUSION = {"status": "unmeasured", "reason": "checkpoint projection does not expose a final worker conclusion"}
CONTROL_SCHEMA_VERSION = 1
CONTROL_KINDS = {"provisional_stop", "release"}
PROVISIONAL_STOP_BLOCK_MESSAGE = "BLOCKED: Jev provisional_stop is active; default readback is required before the next tool call."
DEFAULT_SCOPE_SCHEMA_VERSION = 1
DEFAULT_SCOPE_KIND = "default_scope_screen"
DEFAULT_SCOPE_MAX_CALLS_PER_TRIGGER = MAX_CALLS_PER_RUN
DEFAULT_SCOPE_BLOCK_MESSAGE = "BLOCKED: Jev default scope-drift stop is active; default readback is required before the next tool call."
DEFAULT_SCOPE_IDENTITY_BLOCK_MESSAGE = "BLOCKED: Jev default scope-drift stop identity mismatch; matching default readback is required."
DEFAULT_SCOPE_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_SCOPE_UNKNOWN_CONFIDENCE = "不明"
WORKER_BINDING_SCHEMA_VERSION = 1
WORKER_BINDING_KIND = "worker_binding"
POINT_STATE_ENABLED_CONFIG = "point_state_enabled"


class BridgeValidationError(ValueError):
    pass


class JevRequestError(RuntimeError):
    def __init__(self, reason: str, status: int | None = None):
        super().__init__(reason)
        self.reason, self.status = reason, status


class TaskContractError(RuntimeError):
    """The trusted worker/task seam did not expose enough contract material."""


# These envelope flags remain only for compatibility with older consumers.
# They are metadata, not screening results or content gates.


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _iso_from_epoch(value: float | int) -> str:
    return dt.datetime.fromtimestamp(float(value), dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_time(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def _jst_day(timestamp: str | None = None) -> str:
    if timestamp:
        value = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    else:
        value = dt.datetime.now(dt.timezone.utc)
    return value.astimezone(JST).date().isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bounded_string(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise BridgeValidationError(f"{name} must be a non-empty string <= {limit} chars")
    return value.strip()


def _safe_text(value: Any, name: str, limit: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        if allow_empty and value is None:
            return ""
        raise BridgeValidationError(f"{name} must be a string")
    text = " ".join(value.replace("\x00", " ").split())
    if not text and allow_empty:
        return ""
    if not text:
        raise BridgeValidationError(f"{name} must be non-empty")
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    line = (_json(dict(record)) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "ab", closefd=True) as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        fd = -1
    finally:
        if fd >= 0:
            os.close(fd)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if not path.is_file() or path.is_symlink():
        raise BridgeValidationError("sidechannel path must be a regular file")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BridgeValidationError("sidechannel record must be an object")
            rows.append(value)
    return rows


class BridgeStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        self.events_path = self.root / "events.jsonl"
        self.ledger_path = self.root / "jev-ledger.jsonl"
        self.reviews_path = self.root / "reviews.jsonl"
        self.delivery_path = self.root / "deliveries.jsonl"
        self.triggers_path = self.root / "triggers.jsonl"
        self.controls_path = self.root / "controls.jsonl"
        self.diagnostics_path = self.root / "diagnostics.jsonl"
        self.worker_bindings_path = self.root / "worker-bindings.jsonl"
        self.default_scope_path = self.root / "default-scope.jsonl"
        self.lock_path = self.root / ".bridge.lock"
        for path in (self.events_path, self.ledger_path, self.reviews_path, self.delivery_path, self.triggers_path, self.controls_path, self.diagnostics_path, self.worker_bindings_path, self.default_scope_path):
            if path.exists() and not path.is_symlink() and path.is_file():
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def interprocess_lock(self):
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def append_event(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            _append_jsonl(self.events_path, event)

    def append_ledger(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            _append_jsonl(self.ledger_path, record)

    def append_review(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            _append_jsonl(self.reviews_path, record)

    def append_delivery(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            _append_jsonl(self.delivery_path, record)

    def append_trigger(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            _append_jsonl(self.triggers_path, record)

    def append_control(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            _append_jsonl(self.controls_path, record)

    def append_diagnostic(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            _append_jsonl(self.diagnostics_path, record)

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_jsonl(self.events_path)

    def ledger(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_jsonl(self.ledger_path)

    def reviews(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_jsonl(self.reviews_path)

    def deliveries(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_jsonl(self.delivery_path)

    def triggers(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_jsonl(self.triggers_path)

    def controls(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_jsonl(self.controls_path)

    def diagnostics(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_jsonl(self.diagnostics_path)

    def append_worker_binding(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            _append_jsonl(self.worker_bindings_path, record)

    def worker_bindings(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_jsonl(self.worker_bindings_path)

    def append_default_scope(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            _append_jsonl(self.default_scope_path, record)

    def default_scopes(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_jsonl(self.default_scope_path)


def _validate_worker_binding(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("schema_version") != WORKER_BINDING_SCHEMA_VERSION or raw.get("kind") != WORKER_BINDING_KIND:
        raise BridgeValidationError("worker binding envelope is invalid")
    pid = raw.get("worker_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise BridgeValidationError("worker binding pid is invalid")
    task_id = _safe_text(raw.get("task_id"), "worker binding task_id", 128)
    raw_run_id = raw.get("run_id")
    if isinstance(raw_run_id, bool) or not isinstance(raw_run_id, (str, int)):
        raise BridgeValidationError("worker binding run_id is invalid")
    run_id = _safe_text(str(raw_run_id), "worker binding run_id", 128)
    if not run_id.isdigit() or int(run_id) <= 0:
        raise BridgeValidationError("worker binding run_id is invalid")
    worker_profile = _safe_text(raw.get("worker_profile"), "worker binding worker_profile", 64)
    workspace_path = _safe_text(raw.get("workspace_path"), "worker binding workspace_path", 1024)
    if not Path(workspace_path).expanduser().is_absolute():
        raise BridgeValidationError("worker binding workspace_path must be absolute")
    source = _safe_text(raw.get("source"), "worker binding source", 64)
    if source != "on_kanban_worker_spawned":
        raise BridgeValidationError("worker binding source is not the dispatcher observer")
    return {
        "schema_version": WORKER_BINDING_SCHEMA_VERSION,
        "kind": WORKER_BINDING_KIND,
        "source": source,
        "binding_key": _safe_text(raw.get("binding_key"), "worker binding binding_key", 256),
        "task_id": task_id,
        "run_id": run_id,
        "worker_pid": pid,
        "worker_profile": worker_profile,
        "workspace_path": str(Path(workspace_path).expanduser()),
        "board": _safe_text(raw.get("board"), "worker binding board", 64, allow_empty=True),
        "dispatcher_profile": _safe_text(raw.get("dispatcher_profile"), "worker binding dispatcher_profile", 64, allow_empty=True),
        "assignee": _safe_text(raw.get("assignee"), "worker binding assignee", 128, allow_empty=True),
        "recorded_at": _safe_text(raw.get("recorded_at"), "worker binding recorded_at", 64),
    }


def record_worker_binding(
    store: BridgeStore,
    *,
    task_id: str,
    run_id: str | int,
    worker_pid: int,
    worker_profile: str,
    workspace_path: str,
    board: str | None = None,
    dispatcher_profile: str | None = None,
    assignee: str | None = None,
) -> dict[str, Any]:
    """Persist one dispatcher observer binding for a spawned worker PID."""
    key = f"{task_id}:{run_id}:{worker_pid}"
    record = {
        "schema_version": WORKER_BINDING_SCHEMA_VERSION,
        "kind": WORKER_BINDING_KIND,
        "source": "on_kanban_worker_spawned",
        "binding_key": key,
        "task_id": task_id,
        "run_id": str(run_id),
        "worker_pid": worker_pid,
        "worker_profile": worker_profile,
        "workspace_path": workspace_path,
        "board": board or "",
        "dispatcher_profile": dispatcher_profile or "",
        "assignee": assignee or "",
        "recorded_at": _now(),
    }
    validated = _validate_worker_binding(record)
    with store.interprocess_lock():
        for existing in reversed(store.worker_bindings()):
            try:
                current = _validate_worker_binding(existing)
            except BridgeValidationError:
                continue
            if current["binding_key"] == key:
                return current
        store.append_worker_binding(validated)
    return validated


def worker_binding_for_pid(
    store: BridgeStore,
    worker_pid: int,
    worker_profile: str,
    workspace_path: str | None = None,
) -> dict[str, Any] | None:
    """Return the newest valid observer binding matching this worker process."""
    try:
        pid = int(worker_pid)
    except (TypeError, ValueError):
        return None
    expected_workspace = Path(workspace_path).expanduser().resolve() if workspace_path else None
    for raw in reversed(store.worker_bindings()):
        try:
            binding = _validate_worker_binding(raw)
        except (BridgeValidationError, OSError, RuntimeError):
            continue
        if binding["worker_pid"] != pid or binding["worker_profile"] != worker_profile:
            continue
        if expected_workspace is not None:
            try:
                if Path(binding["workspace_path"]).expanduser().resolve() != expected_workspace:
                    continue
            except (OSError, RuntimeError):
                continue
        return binding
    return None


def _load_current_worker_binding(bridge_root: str | Path | None, profile: str) -> dict[str, Any] | None:
    if not bridge_root:
        return None
    root = Path(bridge_root).expanduser()
    if not root.is_dir():
        return None
    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "").strip() or None
    try:
        return worker_binding_for_pid(BridgeStore(root), os.getpid(), profile, workspace)
    except (OSError, ValueError, BridgeValidationError):
        return None


def _validate_common(raw: Mapping[str, Any], *, task_id: str | None = None) -> tuple[str, str, str, list[str], str, list[str]]:
    event_task = _safe_text(raw.get("task_id") or task_id, "task_id", 128)
    run_id = _safe_text(raw.get("run_id"), "run_id", 128)
    checkpoint_id = _safe_text(raw.get("checkpoint_id"), "checkpoint_id", 128)
    criteria = raw.get("criteria", raw.get("completion_conditions"))
    if not isinstance(criteria, list) or not criteria or len(criteria) > MAX_CRITERIA:
        raise BridgeValidationError("criteria must be 1..8 short strings")
    clean_criteria = [_safe_text(value, "criteria item", 300) for value in criteria]
    observation = _safe_text(raw.get("observation", "bounded worker checkpoint"), "observation", MAX_TEXT)
    refs = raw.get("evidence_refs")
    if not isinstance(refs, list) or len(refs) > MAX_REFS:
        raise BridgeValidationError("evidence_refs must be a list of short references")
    clean_refs = [_safe_text(value, "evidence reference", 300) for value in refs]
    return event_task, run_id, checkpoint_id, clean_criteria, observation, clean_refs


def _legacy_checkpoint(raw: Mapping[str, Any], *, task_id: str | None = None) -> dict[str, Any]:
    if raw.get("schema_version", LEGACY_SCHEMA_VERSION) != LEGACY_SCHEMA_VERSION or raw.get("synthetic") is not True:
        raise BridgeValidationError("legacy checkpoints require schema_version=1 synthetic=true")
    if raw.get("category") != LEGACY_CATEGORY:
        raise BridgeValidationError("legacy checkpoint category is not eligible")
    sensitive = raw.get("sensitive", False)
    projection_safe = raw.get("projection_safe", True)
    if not isinstance(sensitive, bool) or not isinstance(projection_safe, bool):
        raise BridgeValidationError("legacy checkpoint compatibility markers must be boolean")
    event_task, run_id, checkpoint_id, criteria, observation, refs = _validate_common(raw, task_id=task_id)
    tools = raw.get("completed_tools")
    age = raw.get("changed_evidence_age_seconds")
    if isinstance(tools, bool) or not isinstance(tools, int) or tools < MIN_COMPLETED_TOOLS:
        raise BridgeValidationError("completed_tools must be >= 3")
    if isinstance(age, bool) or not isinstance(age, (int, float)) or age < CHECKPOINT_CADENCE_SECONDS:
        raise BridgeValidationError("changed_evidence_age_seconds must be >= 120")
    original = _safe_text(raw.get("original_request", "synthetic bridge checkpoint"), "original_request", 300)
    unknown = _safe_text(raw.get("current_unknown", "bounded synthetic observation"), "current_unknown", 300)
    next_action = _safe_text(raw.get("next_action", "default reviewer decides next step"), "next_action", 300)
    event: dict[str, Any] = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "synthetic": True,
        "sensitive": sensitive,
        "category": LEGACY_CATEGORY,
        "projection_safe": projection_safe,
        "projection_version": 1,
        "task_id": event_task,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "criteria": criteria,
        "observation": observation,
        "evidence_refs": refs,
        "completed_tools": tools,
        "changed_evidence_age_seconds": float(age),
        "original_request": original,
        "current_unknown": unknown,
        "next_action": next_action,
    }
    target = raw.get("target_session_key")
    if target is not None:
        event["target_session_key"] = _safe_text(target, "target_session_key", 300)
    named_milestone = raw.get("next_named_milestone")
    if named_milestone is not None:
        event["next_named_milestone"] = _safe_text(named_milestone, "next_named_milestone", 160)
    event["event_key"] = f"{event_task}:{run_id}:{checkpoint_id}"
    if len(_json(event).encode("utf-8")) > MAX_EVENT_BYTES:
        raise BridgeValidationError("checkpoint projection exceeds 16000 UTF-8 bytes")
    return event


def _real_checkpoint(raw: Mapping[str, Any], *, task_id: str | None = None) -> dict[str, Any]:
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("category") != WORKER_CATEGORY:
        raise BridgeValidationError("worker checkpoints require schema_version=2 category=worker_checkpoint")
    sensitive = raw.get("sensitive", False)
    projection_safe = raw.get("projection_safe")
    if not isinstance(sensitive, bool) or not isinstance(projection_safe, bool) or raw.get("projection_version") != 1:
        raise BridgeValidationError("worker checkpoint compatibility markers are invalid")
    event_task, run_id, checkpoint_id, criteria, observation, refs = _validate_common(raw, task_id=task_id)
    tools = raw.get("completed_tools")
    since = raw.get("tools_since_previous")
    age = raw.get("changed_evidence_age_seconds")
    if isinstance(tools, bool) or not isinstance(tools, int) or tools < MIN_COMPLETED_TOOLS:
        raise BridgeValidationError("completed_tools must be >= 3")
    if isinstance(since, bool) or not isinstance(since, int) or since < MIN_COMPLETED_TOOLS:
        raise BridgeValidationError("tools_since_previous must be >= 3")
    if isinstance(age, bool) or not isinstance(age, (int, float)) or age < CHECKPOINT_CADENCE_SECONDS:
        raise BridgeValidationError("changed_evidence_age_seconds must be >= 120")
    steps = raw.get("recent_steps")
    if not isinstance(steps, list) or not steps or len(steps) > MAX_RECENT_STEPS:
        raise BridgeValidationError("recent_steps must contain 1..6 bounded steps")
    clean_steps: list[dict[str, Any]] = []
    for item in steps:
        if not isinstance(item, Mapping):
            raise BridgeValidationError("recent_steps entries must be objects")
        action = _safe_text(item.get("action"), "step action", 96)
        result = _safe_text(item.get("result"), "step result", 240)
        changed = item.get("evidence_changed")
        if not isinstance(changed, bool):
            raise BridgeValidationError("step evidence_changed must be boolean")
        refs_item = item.get("evidence_refs", [])
        if not isinstance(refs_item, list) or len(refs_item) > MAX_REFS:
            raise BridgeValidationError("step evidence_refs must be a short list")
        clean_steps.append({"action": action, "result": result, "evidence_changed": changed, "evidence_refs": [_safe_text(v, "step reference", 300) for v in refs_item]})
    original = _safe_text(raw.get("original_request"), "original_request", 300)
    unknown = _safe_text(raw.get("current_unknown"), "current_unknown", 300)
    next_action = _safe_text(raw.get("next_action"), "next_action", 300)
    fingerprint = _safe_text(raw.get("evidence_fingerprint"), "evidence_fingerprint", 128)
    event: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "synthetic": bool(raw.get("synthetic", False)),
        "sensitive": sensitive,
        "projection_safe": projection_safe,
        "projection_version": 1,
        "source": _safe_text(raw.get("source", "worker_hook"), "source", 64),
        "category": WORKER_CATEGORY,
        "profile": _safe_text(raw.get("profile", "unknown"), "profile", 64),
        "task_id": event_task,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "criteria": criteria,
        "observation": observation,
        "recent_steps": clean_steps,
        "evidence_refs": refs,
        "completed_tools": tools,
        "tools_since_previous": since,
        "changed_evidence": raw.get("changed_evidence") is True,
        "changed_evidence_age_seconds": float(age),
        "original_request": original,
        "current_unknown": unknown,
        "next_action": next_action,
        "evidence_fingerprint": fingerprint,
    }
    target = raw.get("target_session_key")
    if target is not None:
        event["target_session_key"] = _safe_text(target, "target_session_key", 300)
    named_milestone = raw.get("next_named_milestone")
    if named_milestone is not None:
        event["next_named_milestone"] = _safe_text(named_milestone, "next_named_milestone", 160)
    event["event_key"] = f"{event_task}:{run_id}:{checkpoint_id}"
    if len(_json(event).encode("utf-8")) > MAX_EVENT_BYTES:
        raise BridgeValidationError("checkpoint projection exceeds 16000 UTF-8 bytes")
    return event


def validate_checkpoint(raw: Mapping[str, Any], *, task_id: str | None = None) -> dict[str, Any]:
    """Validate both the legacy manual envelope and the common worker envelope."""
    if not isinstance(raw, Mapping):
        raise BridgeValidationError("checkpoint must be an object")
    allowed = {
        "schema_version", "synthetic", "sensitive", "projection_safe", "projection_version", "source",
        "category", "profile", "task_id", "run_id", "checkpoint_id", "criteria", "completion_conditions",
        "observation", "recent_steps", "evidence_refs", "completed_tools", "tools_since_previous",
        "changed_evidence", "changed_evidence_age_seconds", "target_session_key", "next_named_milestone", "original_request",
        "current_unknown", "next_action", "evidence_fingerprint",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise BridgeValidationError("unsupported fields: " + ",".join(unknown))
    version = raw.get("schema_version", LEGACY_SCHEMA_VERSION)
    if version == LEGACY_SCHEMA_VERSION:
        return _legacy_checkpoint(raw, task_id=task_id)
    if version == SCHEMA_VERSION:
        return _real_checkpoint(raw, task_id=task_id)
    raise BridgeValidationError("unsupported checkpoint schema version")


def _event_projection_is_eligible(event: Mapping[str, Any]) -> bool:
    try:
        if event.get("schema_version") == LEGACY_SCHEMA_VERSION and event.get("category") == LEGACY_CATEGORY and event.get("synthetic") is True:
            _legacy_checkpoint(event, task_id=event.get("task_id"))
            return True
        if event.get("schema_version") == SCHEMA_VERSION and event.get("category") == WORKER_CATEGORY:
            _real_checkpoint(event, task_id=event.get("task_id"))
            return True
    except BridgeValidationError:
        return False
    return False


def _reference_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)][:MAX_REFS]
    return []


def _args_evidence_refs(args: Any) -> list[str]:
    if not isinstance(args, Mapping):
        return []
    refs: list[str] = []
    allowed_keys = {"path", "file", "file_path", "source_path", "target_path", "output_path", "url", "uri", "evidence_ref", "evidence_refs"}
    for key, value in args.items():
        key_text = str(key).lower()
        if key_text not in allowed_keys or any(word in key_text for word in ("token", "secret", "password", "content")):
            continue
        for item in _reference_values(value):
            try:
                clean = _safe_text(item, "evidence reference", 300)
            except BridgeValidationError:
                continue
            if clean not in refs:
                refs.append(clean)
            if len(refs) >= MAX_REFS:
                return refs
    return refs


def _json_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and len(value) <= 8_000 and value.lstrip().startswith("{"):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, Mapping) else None
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return None


def _project_tool_result(tool_name: str, args: Any, result: Any, status: Any = None, error_type: Any = None, error_message: Any = None) -> dict[str, Any]:
    data = _json_mapping(result)
    refs = _args_evidence_refs(args)
    changed = False
    verified = False
    failed = str(status or "").lower() in {"error", "failed", "cancelled"} or str(error_type or "").strip() != ""
    parts: list[str] = []
    if data is not None:
        for key in ("ok", "success", "status", "exit_code", "changed", "written", "created", "updated", "verified"):
            if key in data and isinstance(data[key], (str, int, float, bool)):
                value = data[key]
                if key in {"changed", "written", "created", "updated"} and value is True:
                    changed = True
                if key == "verified" and value is True:
                    verified = True
                if key == "ok" and value is False:
                    failed = True
                parts.append(f"{key}={str(value).lower() if isinstance(value, bool) else value}")
        files = data.get("files_modified")
        if isinstance(files, list) and files:
            changed = True
            parts.append(f"files_modified={min(len(files), MAX_REFS)}")
        for key in ("path", "file_path", "output_path", "source_path"):
            if key in data and isinstance(data[key], str):
                try:
                    ref = _safe_text(data[key], "result reference", 300)
                except BridgeValidationError:
                    ref = ""
                if ref and ref not in refs:
                    refs.append(ref)
                if len(refs) >= MAX_REFS:
                    break
        if isinstance(data.get("error"), str):
            try:
                parts.append("error=" + _safe_text(data["error"], "result error", 120))
            except BridgeValidationError:
                failed = True
    else:
        text = str(result or "").strip()
        if text:
            failed = failed or text.lower().startswith(("error", "failed", "exception"))
            parts.append("text_present=true")
    if str(error_message or "").strip():
        parts.append("error_message_present=true")
        failed = True
    if tool_name in {"write_file", "patch", "kanban_comment", "kanban_complete"} and not failed:
        changed = True
    kind = "error" if failed else "changed" if changed else "verified" if verified else "ok"
    try:
        summary = _safe_text("; ".join(parts) or "result_present=false", "result summary", 240)
    except BridgeValidationError:
        summary = "result_summary_unavailable"
    return {"result": summary, "result_kind": kind, "evidence_changed": bool(changed and not failed), "evidence_refs": refs[:MAX_REFS]}


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    run_id: str
    profile: str
    started_at: float
    original_request: str
    criteria: tuple[str, ...]
    current_unknown: str
    next_action: str
    evidence_refs: tuple[str, ...]


def _extract_scope_admission(body: str) -> Mapping[str, Any]:
    limited = body[:32_000]
    for marker in ("scope_admission_json:", "scope_admission_json="):
        index = limited.find(marker)
        if index < 0:
            continue
        tail = limited[index + len(marker):].lstrip()
        try:
            value, _ = json.JSONDecoder().raw_decode(tail)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, Mapping):
            return value
    raise TaskContractError("scope_admission_json is not available in task body")


def _load_dispatcher_contract() -> TaskContract:
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    raw_run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    profile = os.environ.get("HERMES_PROFILE", "").strip()
    if not task_id or not raw_run_id or not profile:
        raise TaskContractError("trusted HERMES_KANBAN_TASK/HERMES_KANBAN_RUN_ID/HERMES_PROFILE is missing")
    if not raw_run_id.isdigit() or int(raw_run_id) <= 0:
        raise TaskContractError("trusted dispatcher run id is invalid")
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        with kbc.connect_closing() as conn:
            task = kb.get_task(conn, task_id)
            run = kb.get_run(conn, int(raw_run_id))
            latest = kb.latest_run(conn, task_id)
    except Exception as exc:
        raise TaskContractError("active Kanban board read is unavailable") from exc
    if task is None or run is None or latest is None:
        raise TaskContractError("active task/run was not found in the trusted board")
    if task.status != "running" or task.current_run_id != int(raw_run_id) or latest.id != int(raw_run_id):
        raise TaskContractError("active task/run is not the trusted running assignment")
    if task.assignee != profile or (run.profile and run.profile != profile):
        raise TaskContractError("task/run profile binding does not match trusted worker profile")
    if not isinstance(task.body, str) or not task.body.strip():
        raise TaskContractError("task body is missing")
    admission = _extract_scope_admission(task.body)
    values = admission.get("completion_conditions")
    if not isinstance(values, list) or not values:
        raise TaskContractError("scope_admission_json.completion_conditions is missing")
    criteria = tuple(_safe_text(value, "completion condition", 300) for value in values[:MAX_CRITERIA] if isinstance(value, str) and value.strip())
    if not criteria:
        raise TaskContractError("completion conditions contain no usable strings")
    original_value = admission.get("outcome_target") or task.title
    original = _safe_text(original_value, "original request", 300)
    decision_points = admission.get("decision_points")
    if isinstance(decision_points, list) and decision_points:
        unknown = "; ".join(str(value) for value in decision_points[:4] if isinstance(value, str) and value.strip())
        current_unknown = _safe_text(unknown or "unknown: decision points were not readable", "current unknown", 300)
    else:
        current_unknown = "unknown: task contract does not enumerate remaining decision points"
    next_value = admission.get("next_action") or "read back the remaining completion conditions before completion"
    next_action = _safe_text(next_value, "next action", 300)
    started_at = run.started_at or task.started_at
    if not isinstance(started_at, (int, float)) or started_at <= 0:
        raise TaskContractError("trusted task/run start time is unavailable")
    return TaskContract(
        task_id=task_id,
        run_id=raw_run_id,
        profile=profile,
        started_at=float(started_at),
        original_request=original,
        criteria=criteria,
        current_unknown=current_unknown,
        next_action=next_action,
        evidence_refs=(f"kanban://task/{task_id}",),
    )


def _contract_event_key(task_id: str, run_id: str) -> str:
    return f"{task_id}:{run_id}"


class Producer:
    """Create bounded real-worker checkpoints from the normal post-tool hook."""

    def __init__(self, store: BridgeStore, profile_name: str | None = None, *, contract_loader: Callable[[], TaskContract] | None = None, clock: Callable[[], float] | None = None, cadence_seconds: float = CHECKPOINT_CADENCE_SECONDS, bridge_root: str | Path | None = None):
        self.store = store
        self.profile_name = profile_name or os.environ.get("HERMES_PROFILE", "") or "unknown"
        self.bridge_root = bridge_root
        self.contract_loader = contract_loader or _load_dispatcher_contract
        self.clock = clock or time.time
        self.cadence_seconds = float(cadence_seconds)
        self._states: dict[str, dict[str, Any]] = {}
        self._loaded_contract: TaskContract | None = None
        self._diagnosed: set[tuple[str, str]] = set()
        self._lock = threading.RLock()

    def checkpoint_tool(self, args: dict, **kwargs: Any) -> str:
        return _json({"accepted": True, "synthetic_only": False, "manual_optional": True, "hook_persists": True})

    def pre_tool_call(self, tool_name: str = "", args: Any = None, **kwargs: Any) -> dict[str, str] | None:
        """Block the next worker tool while a bound Jev provisional stop is active."""
        binding = _worker_binding(self.profile_name)
        if binding is None:
            return None
        task_id, run_id = binding
        try:
            active = _active_provisional_stop(self.store, task_id, run_id)
        except Exception:
            # A configured control channel must fail closed: executing after an
            # unreadable stop log would violate the explicit stop request.
            log.warning("Jev provisional-stop control read failed; blocking tool=%s", tool_name, exc_info=True)
            return {"action": "block", "message": "BLOCKED: Jev control readback failed; no tool execution permitted."}
        if active is None:
            return None
        if tool_name == "kanban_block":
            self_binding = _self_block_binding(self.profile_name, self.bridge_root, binding)
            if self_binding is not None and _kanban_block_args_match(args, self_binding):
                # Returning None preserves the native tool's schema, task/run,
                # board, and permission checks; this hook grants no board access.
                return None
        return {"action": "block", "message": PROVISIONAL_STOP_BLOCK_MESSAGE}

    def _diagnostic(self, reason: str, contract: TaskContract | None = None, *, detail: str = "") -> None:
        task_id = contract.task_id if contract else os.environ.get("HERMES_KANBAN_TASK", "") or "unknown"
        run_id = contract.run_id if contract else os.environ.get("HERMES_KANBAN_RUN_ID", "") or "unknown"
        key = (f"{task_id}:{run_id}", reason)
        if key in self._diagnosed:
            return
        self._diagnosed.add(key)
        try:
            clean_detail = _safe_text(detail or reason, "diagnostic detail", 240)
        except BridgeValidationError:
            clean_detail = reason
        self.store.append_diagnostic({"kind": "producer_diagnostic", "reason": reason, "detail": clean_detail, "task_id": task_id, "run_id": run_id, "profile": self.profile_name, "timestamp": _now()})

    def _contract(self) -> TaskContract | None:
        # Re-read the trusted dispatcher contract for every hook. Cached semantic
        # task data must not outlive the active task/run/profile binding.
        try:
            contract = self.contract_loader()
            if contract.profile != self.profile_name:
                raise TaskContractError("trusted task profile differs from configured producer profile")
            self._loaded_contract = contract
            return contract
        except Exception as exc:
            self._diagnostic("task_contract_read_failed", detail=str(exc) or type(exc).__name__)
            return None

    def _state(self, contract: TaskContract) -> dict[str, Any]:
        key = _contract_event_key(contract.task_id, contract.run_id)
        state = self._states.get(key)
        if state is not None:
            return state
        events = [row for row in self.store.events() if row.get("task_id") == contract.task_id and str(row.get("run_id")) == contract.run_id]
        current = [row for row in events if row.get("category") == WORKER_CATEGORY]
        current.sort(key=lambda row: _parse_time(row.get("created_at")) or 0.0)
        latest = current[-1] if current else None
        last_at = _parse_time(latest.get("created_at")) if latest else contract.started_at
        state = {
            "total_tools": int(latest.get("completed_tools", 0)) if latest else 0,
            "tools_since": 0,
            "changed": False,
            "steps": [],
            "last_checkpoint_at": last_at or contract.started_at,
            "checkpoint_count": len(events),
        }
        self._states[key] = state
        return state

    def _manual_event(self, args: Mapping[str, Any], hook_task_id: str | None) -> None:
        try:
            contract = self._contract()
            trusted_task_id = contract.task_id if contract else None
            event = validate_checkpoint(args, task_id=trusted_task_id or hook_task_id)
            if contract and event.get("task_id") != contract.task_id:
                raise BridgeValidationError("manual checkpoint task_id does not match trusted task")
            event.update({"state": "pending", "created_at": _now(), "producer_profile": self.profile_name, "source": "manual"})
            with self.store.interprocess_lock():
                existing = self.store.events()
                if any(row.get("event_key") == event["event_key"] for row in existing):
                    return
                self.store.append_event(event)
        except Exception as exc:
            self._diagnostic("manual_checkpoint_rejected", detail=str(exc) or type(exc).__name__)

    def _build_event(self, contract: TaskContract, state: Mapping[str, Any], now: float) -> dict[str, Any]:
        refs: list[str] = list(contract.evidence_refs)
        for step in state["steps"]:
            for ref in step.get("evidence_refs", []):
                if ref not in refs and len(refs) < MAX_REFS:
                    refs.append(ref)
        fingerprint_source = {
            "task_id": contract.task_id,
            "run_id": contract.run_id,
            "criteria": contract.criteria,
            "steps": state["steps"],
            "refs": refs,
            "changed": state["changed"],
        }
        fingerprint = hashlib.sha256(_json(fingerprint_source).encode("utf-8")).hexdigest()[:24]
        checkpoint_id = f"cp-{int(state['total_tools'])}-{fingerprint[:12]}"
        kinds = ",".join(step.get("result_kind", "ok") for step in state["steps"])
        observation = _safe_text(f"bounded hook projection: tools_since_previous={state['tools_since']}; evidence_changed={str(bool(state['changed'])).lower()}; result_kinds={kinds}", "observation", MAX_TEXT)
        age = max(0.0, float(now) - float(state["last_checkpoint_at"]))
        raw = {
            "schema_version": SCHEMA_VERSION,
            "synthetic": False,
            "sensitive": False,
            "projection_safe": True,
            "projection_version": 1,
            "source": "worker_hook",
            "category": WORKER_CATEGORY,
            "profile": contract.profile,
            "task_id": contract.task_id,
            "run_id": contract.run_id,
            "checkpoint_id": checkpoint_id,
            "criteria": list(contract.criteria),
            "observation": observation,
            "recent_steps": list(state["steps"]),
            "evidence_refs": refs,
            "completed_tools": int(state["total_tools"]),
            "tools_since_previous": int(state["tools_since"]),
            "changed_evidence": bool(state["changed"]),
            "changed_evidence_age_seconds": age,
            "original_request": contract.original_request,
            "current_unknown": contract.current_unknown,
            "next_action": contract.next_action,
            "evidence_fingerprint": fingerprint,
        }
        return _real_checkpoint(raw, task_id=contract.task_id) | {"state": "pending", "created_at": _iso_from_epoch(now), "producer_profile": self.profile_name}

    def _persist_if_due(self, contract: TaskContract, state: dict[str, Any], now: float) -> None:
        if state["tools_since"] < MIN_COMPLETED_TOOLS or not state["changed"]:
            return
        if now - float(state["last_checkpoint_at"]) < self.cadence_seconds:
            return
        with self.store.interprocess_lock():
            events = self.store.events()
            run_events = [row for row in events if row.get("task_id") == contract.task_id and str(row.get("run_id")) == contract.run_id]
            if len(run_events) >= min(MAX_CALLS_PER_RUN, MAX_LIVE_CALLS_PER_RUN):
                self._diagnostic("checkpoint_cap_reached", contract)
                state["checkpoint_count"] = len(run_events)
                return
            event = self._build_event(contract, state, now)
            if any(row.get("evidence_fingerprint") == event["evidence_fingerprint"] for row in run_events):
                self._diagnostic("checkpoint_unchanged", contract)
                return
            latest = max((_parse_time(row.get("created_at")) or 0.0 for row in run_events), default=contract.started_at)
            if now - latest < self.cadence_seconds:
                return
            self.store.append_event(event)
            state["last_checkpoint_at"] = now
            state["tools_since"] = 0
            state["changed"] = False
            state["steps"] = []
            state["checkpoint_count"] = len(run_events) + 1

    def post_tool_call(self, tool_name: str = "", args: Any = None, task_id: str | None = None, result: Any = None, status: Any = None, error_type: Any = None, error_message: Any = None, tool_call_id: str | None = None, **kwargs: Any) -> None:
        if tool_name == "jev_bridge_checkpoint" and isinstance(args, Mapping):
            self._manual_event(args, task_id)
            return
        if not tool_name:
            return
        contract = self._contract()
        if contract is None:
            return
        # Callback task_id is runtime correlation metadata, never board identity.
        projection = _project_tool_result(tool_name, args, result, status, error_type, error_message)
        with self._lock:
            state = self._state(contract)
            state["total_tools"] += 1
            state["tools_since"] += 1
            state["changed"] = bool(state["changed"] or projection["evidence_changed"])
            step = {
                "action": _safe_text(tool_name, "step action", 96),
                "result": projection["result"],
                "result_kind": projection["result_kind"],
                "evidence_changed": projection["evidence_changed"],
                "evidence_refs": projection["evidence_refs"],
            }
            state["steps"].append(step)
            state["steps"] = state["steps"][-MAX_RECENT_STEPS:]
            try:
                self._persist_if_due(contract, state, float(self.clock()))
            except Exception as exc:
                self._diagnostic("checkpoint_persist_failed", contract, detail=str(exc) or type(exc).__name__)


def _point_state_milestone(event: Mapping[str, Any]) -> str:
    """Return the stable named contract milestone used by point-state budgets.

    Checkpoint ids identify delivery events, not policy scope.  If the host does
    not provide a named milestone yet, all current-task events intentionally
    share one bounded scope instead of resetting correction/refresh budgets.
    """
    for key in ("next_named_milestone", "named_milestone", "milestone"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return _safe_text(value, "point-state milestone", 160)
    return "current-task"


def _point_snapshot_from_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Project one bounded event into the opt-in point-state contract."""
    steps = event.get("recent_steps")
    latest = steps[-1] if isinstance(steps, list) and steps else None
    return {
        "goal": event.get("original_request", ""),
        "scope": event.get("criteria", []),
        "next_named_milestone": _point_state_milestone(event),
        "expected_next_action": event.get("next_action", ""),
        "latest_bounded_input": latest,
        "latest_bounded_result": event.get("observation", ""),
        "refs": event.get("evidence_refs", []),
        "next_named_verification": event.get("next_named_verification"),
        "correction_instruction": event.get("correction_instruction"),
    }


def _point_snapshot_from_request(request_body: Mapping[str, Any]) -> dict[str, Any]:
    """Read the existing default-scope projection without retaining tool args."""
    state_value = request_body.get("state")
    try:
        state = json.loads(state_value) if isinstance(state_value, str) else {}
    except (TypeError, ValueError):
        state = {}
    if not isinstance(state, Mapping):
        state = {}
    if "goal" in state or "scope" in state:
        return {
            "goal": state.get("goal", ""),
            "scope": state.get("scope", []),
            "next_named_milestone": state.get("next_named_milestone", ""),
            "expected_next_action": state.get("expected_next_action", ""),
            "latest_bounded_input": state.get("latest_bounded_input"),
            "latest_bounded_result": state.get("latest_bounded_result", ""),
            "refs": state.get("refs", []),
            "next_named_verification": state.get("next_named_verification"),
        }
    return {
        "goal": state.get("original_request", ""),
        "scope": state.get("completion_conditions", []),
        "next_named_milestone": state.get("checkpoint_id", ""),
        "expected_next_action": state.get("expected_next_action", ""),
        "latest_bounded_input": state.get("current_action"),
        "latest_bounded_result": state.get("remaining_unknowns", ""),
        "refs": state.get("evidence_refs", []),
        "next_named_verification": state.get("next_named_verification"),
    }


def build_jev_request(event: Mapping[str, Any], *, model: str = MODEL, point_state_enabled: bool = False) -> dict[str, Any]:
    """Build the bounded structural projection shared by both providers."""
    if not _event_projection_is_eligible(event):
        raise BridgeValidationError("event projection is not eligible")
    model_name = _safe_text(model, "model", 128)
    if point_state_enabled:
        request = build_point_state_request(_point_snapshot_from_event(event))
        request["model"] = model_name
        return request
    steps = event.get("recent_steps")
    if not isinstance(steps, list) or not steps:
        steps = [{"action": "synthetic_checkpoint", "result": event["observation"], "evidence_changed": False, "evidence_refs": event.get("evidence_refs", [])}]
    state = {
        "event_key": event["event_key"],
        "task_id": event["task_id"],
        "run_id": event["run_id"],
        "checkpoint_id": event["checkpoint_id"],
        "profile": event.get("profile", "unknown"),
        "original_request": event["original_request"],
        "completion_conditions": event["criteria"],
        "recent_steps": steps,
        "evidence_change": {
            "changed": bool(event.get("changed_evidence", False)),
            "tools_since_previous": event.get("tools_since_previous", event.get("completed_tools", 0)),
            "age_seconds": event.get("changed_evidence_age_seconds"),
            "fingerprint": event.get("evidence_fingerprint", "legacy"),
        },
        "remaining_unknowns": event["current_unknown"],
        "next_action": event["next_action"],
        "evidence_refs": event["evidence_refs"],
    }
    state_json = _json(state)
    if len(state_json.encode("utf-8")) > MAX_EVENT_BYTES:
        raise BridgeValidationError("request state exceeds 16000 UTF-8 bytes")
    return {
        "model": model_name,
        "state": state_json,
        "questions": {"route": {"type": "choice", "instructions": "Choose exactly one advisory route label; never change worker authority.", "criteria": CRITERIA}},
    }


def _safe_usage(value: Any) -> dict[str, int | float | str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError
    clean: dict[str, int | float | str] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        candidate = value.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and 0 <= candidate <= 10_000_000:
            clean[key] = candidate
    cost = value.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        parsed_cost = Decimal(str(cost))
        if parsed_cost.is_finite() and parsed_cost >= 0:
            clean["cost"] = str(parsed_cost)
    elif isinstance(cost, str) and len(cost) <= 64:
        parsed_cost = Decimal(cost)
        if parsed_cost.is_finite() and parsed_cost >= 0:
            clean["cost"] = cost
    return clean or None


def parse_jev_response(value: Any, *, transport: str | None = None) -> dict[str, Any]:
    try:
        route = value["answers"]["route"]
        label = route["choice"]
        probs = route["probabilities"]
        confidence = route["confidence"]
        if label not in LABELS or not isinstance(probs, Mapping) or set(probs) != set(LABELS):
            raise ValueError
        probabilities = {name: float(probs[name]) for name in LABELS}
        if any(v < 0 or v > 1 for v in probabilities.values()) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError
        response_model = value.get("model", MODEL)
        if not isinstance(response_model, str) or not response_model.strip():
            raise ValueError
        decision = {
            "label": label,
            "probabilities": probabilities,
            "confidence": float(confidence),
            "model": _safe_text(response_model, "response model", 128),
            "usage": _safe_usage(value.get("usage")),
        }
        if transport is not None:
            decision["transport"] = _safe_text(transport, "transport", 64)
        for key in ("route_label", "drift_category", "reason"):
            candidate = route.get(key)
            if isinstance(candidate, str) and candidate.strip():
                decision[key] = candidate.strip()
        for key in ("under_scoped", "evidence_mismatch"):
            if route.get(key) is True:
                decision[key] = True
        return decision
    except (KeyError, TypeError, ValueError, OverflowError, InvalidOperation):
        raise JevRequestError("malformed_response")


def _parse_point_state_for_bridge(value: Any, *, transport: str | None = None) -> dict[str, Any]:
    """Adapt the frozen point-state parser to the existing provider seam."""
    decision = parse_point_state_response(value)
    result: dict[str, Any] = {
        "label": decision.state or "unknown",
        "point_state": decision.state,
        "point_state_accepted": decision.accepted,
        "point_state_issue": decision.issue,
        "confidence": decision.confidence,
        "provider_confidence": decision.provider_confidence,
        "_point_decision": decision,
    }
    if transport is not None:
        result["transport"] = _safe_text(transport, "transport", 64)
    return result


def _runtime_api_key(env_name: str) -> str:
    """Read only the named runtime environment credential, never its value in errors."""
    if not isinstance(env_name, str) or not env_name.strip() or len(env_name) > 64:
        raise JevRequestError("configuration_invalid")
    candidate = os.environ.get(env_name)
    if not candidate:
        hermes_home = os.environ.get("HERMES_HOME", "").strip()
        if hermes_home:
            try:
                from hermes_cli.env_loader import load_hermes_dotenv
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    load_hermes_dotenv(hermes_home=hermes_home)
            except Exception:
                # The process may use an external secret source rather than .env.
                pass
            candidate = os.environ.get(env_name)
    if not isinstance(candidate, str) or not candidate.strip():
        raise JevRequestError("credential_unavailable")
    return candidate.strip()


def _post_provider(spec: ProviderSpec, request_body: Mapping[str, Any], timeout: float = REQUEST_TIMEOUT_SECONDS) -> Any:
    if not isinstance(spec, ProviderSpec) or not isinstance(request_body, Mapping):
        raise JevRequestError("configuration_invalid")
    try:
        endpoint = _safe_text(spec.endpoint, "provider endpoint", 512)
        if not endpoint.startswith("https://"):
            raise JevRequestError("configuration_invalid")
        key = _runtime_api_key(spec.api_key_env)
        request = Request(endpoint, data=_json(dict(request_body)).encode("utf-8"), method="POST", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"})
    except JevRequestError:
        raise
    except (TypeError, ValueError, UnicodeError):
        raise JevRequestError("configuration_invalid")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(2_000_000)
            status_value = getattr(response, "status", 200)
    except HTTPError as exc:
        raise JevRequestError("http_error", exc.code) from exc
    except TimeoutError as exc:
        raise JevRequestError("timeout") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        raise JevRequestError("timeout" if isinstance(reason, TimeoutError) else "network_error") from exc
    except OSError as exc:
        raise JevRequestError("network_error") from exc
    try:
        status = int(status_value)
    except (TypeError, ValueError):
        raise JevRequestError("malformed_response")
    if not 200 <= status < 300:
        raise JevRequestError("http_error", status)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JevRequestError("malformed_response") from exc


def _post_jev(request_body: Mapping[str, Any], timeout: float = REQUEST_TIMEOUT_SECONDS) -> Any:
    """Compatibility wrapper for callers of the former OpenRouter-only bridge."""
    return _post_provider(FALLBACK_PROVIDER, request_body, timeout)


def _fallback_allowed(error: JevRequestError) -> bool:
    return error.reason in {"credential_unavailable", "network_error", "timeout"} or error.status in FALLBACK_HTTP_STATUSES or (error.status is not None and 500 <= error.status <= 599)


def request_decision_with_fallback(
    request_body: Mapping[str, Any],
    *,
    primary: ProviderSpec = PRIMARY_PROVIDER,
    fallback: ProviderSpec = FALLBACK_PROVIDER,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    post_fn: Callable[[ProviderSpec, Mapping[str, Any], float], Any] | None = None,
    parser: Callable[..., dict[str, Any]] = parse_jev_response,
) -> dict[str, Any]:
    """Call TypeSafe first and use OpenRouter only for bounded availability failures."""
    if not isinstance(request_body, Mapping):
        raise BridgeValidationError("request body must be an object")
    sender = post_fn or _post_provider

    def call(spec: ProviderSpec) -> dict[str, Any]:
        body = dict(request_body)
        body["model"] = spec.model
        return parser(sender(spec, body, timeout), transport=spec.transport)

    try:
        return call(primary)
    except JevRequestError as exc:
        if not _fallback_allowed(exc):
            raise
    return call(fallback)


def _configured_provider(ctx: Any, prefix: str, default: ProviderSpec) -> ProviderSpec:
    """Read provider routing settings without ever reading a credential value."""
    try:
        endpoint_value = _config_value(ctx, f"{prefix}_endpoint", default.endpoint)
        env_value = _config_value(ctx, f"{prefix}_api_key_env", default.api_key_env)
        model_value = _config_value(ctx, f"{prefix}_model", default.model)
        endpoint = _safe_text(endpoint_value, f"{prefix}_endpoint", 512)
        api_key_env = _safe_text(env_value, f"{prefix}_api_key_env", 64)
        model = _safe_text(model_value, f"{prefix}_model", 128)
        if not endpoint.startswith("https://") or api_key_env.upper() != api_key_env or not api_key_env.replace("_", "").isalnum() or not api_key_env[0].isalpha():
            raise BridgeValidationError("provider settings are invalid")
        return ProviderSpec(default.name, endpoint, api_key_env, model, default.transport)
    except (BridgeValidationError, IndexError, TypeError):
        raise JevRequestError("configuration_invalid")


def _configured_timeout(ctx: Any) -> float:
    value = _config_value(ctx, "request_timeout_seconds", REQUEST_TIMEOUT_SECONDS)
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise JevRequestError("configuration_invalid")
    if not 0 < timeout <= 120:
        raise JevRequestError("configuration_invalid")
    return timeout


def _cost(usage: Any) -> Decimal:
    value = usage.get("cost") if isinstance(usage, Mapping) else None
    try:
        parsed = Decimal(str(value))
        return parsed if parsed.is_finite() and parsed >= 0 else Decimal("0")
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def advisory_text(event: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    refs = ", ".join(event.get("evidence_refs", [])) or "(none)"
    return (
        "Jev advisory only; it is not an instruction and does not change worker authority.\n"
        f"route={decision['label']} confidence={decision['confidence']:.3f}\n"
        f"task={event['task_id']} run={event['run_id']} checkpoint={event['checkpoint_id']}\n"
        f"observation={event['observation']}\nreferences={refs}"
    )


def _trigger_worker_conclusion(event: Mapping[str, Any]) -> Any:
    value = event.get("worker_conclusion")
    if isinstance(value, str) and value.strip():
        return _safe_text(value, "worker_conclusion", MAX_TEXT)
    if isinstance(value, Mapping):
        status = value.get("status")
        reason = value.get("reason")
        if isinstance(status, str) and status.strip() and isinstance(reason, str) and reason.strip():
            return {"status": _safe_text(status, "worker_conclusion.status", 64), "reason": _safe_text(reason, "worker_conclusion.reason", MAX_TEXT)}
    return dict(UNMEASURED_WORKER_CONCLUSION)


def _trigger_observed_actions(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    steps = event.get("recent_steps")
    if not isinstance(steps, list):
        return actions
    for item in steps[:MAX_RECENT_STEPS]:
        if not isinstance(item, Mapping):
            continue
        try:
            action = _safe_text(item.get("action"), "observed action", 96)
            result = _safe_text(item.get("result"), "observed result", 240)
        except BridgeValidationError:
            continue
        actions.append({
            "action": action,
            "result": result,
            "evidence_changed": item.get("evidence_changed") is True,
            "evidence_refs": [_safe_text(ref, "observed evidence reference", 300) for ref in item.get("evidence_refs", []) if isinstance(ref, str)][:MAX_REFS],
        })
    return actions


def _trigger_category(event: Mapping[str, Any], decision: Mapping[str, Any], prior_reviews: list[dict[str, Any]]) -> str | None:
    explicit = decision.get("drift_category") or decision.get("route_label")
    if isinstance(explicit, str) and explicit.strip() in INTERVENTION_CATEGORIES:
        return explicit.strip()
    if decision.get("under_scoped") is True:
        return "under_scoped"
    if decision.get("evidence_mismatch") is True:
        return "evidence_mismatch"
    label = decision.get("label")
    if label == "scope_drift":
        return "scope_drift"
    if label == "completion_candidate":
        return "completion_candidate"
    if label == "stalled":
        task_id, run_id, event_key = event.get("task_id"), str(event.get("run_id")), event.get("event_key")
        repeated = any(
            row.get("task_id") == task_id and str(row.get("run_id")) == run_id
            and row.get("label") == "stalled" and row.get("event_key") != event_key
            for row in prior_reviews
        )
        return "repeated_stall" if repeated else None
    return None


def _trigger_reason(category: str, decision: Mapping[str, Any]) -> str:
    supplied = decision.get("reason")
    if isinstance(supplied, str) and supplied.strip():
        return _safe_text(supplied, "trigger reason", MAX_TEXT)
    return {
        "under_scoped": "completion-condition coverage requires default readback",
        "scope_drift": "next action may exceed the admitted scope",
        "evidence_mismatch": "observed action and evidence references require comparison",
        "repeated_stall": "repeated stalled observations require default readback",
        "completion_candidate": "completion candidate requires default evidence readback",
    }[category]


def _trigger_id(dedupe_key: str) -> str:
    return "trg-" + hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()[:24]


def _control_id(dedupe_key: str) -> str:
    return "ctl-" + hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()[:24]


def _control_dedupe_key(task_id: str, run_id: str, checkpoint_id: str, trigger_id: str, control: str, scope: str = "worker") -> str:
    base = f"{task_id}:{run_id}:{checkpoint_id}:{trigger_id}:{control}"
    return base if scope == "worker" else f"{base}:{scope}"


def _control_record(trigger: Mapping[str, Any], control: str, *, source: str, reason: str, scope: str = "worker") -> dict[str, Any]:
    if control not in CONTROL_KINDS:
        raise BridgeValidationError("unsupported worker control")
    task_id = _safe_text(trigger.get("task_id"), "control task_id", 128)
    run_id = _safe_text(str(trigger.get("run_id")), "control run_id", 128)
    checkpoint_id = _safe_text(trigger.get("checkpoint_id"), "control checkpoint_id", 128)
    trigger_id = _safe_text(trigger.get("trigger_id"), "control trigger_id", 128)
    source_name = _safe_text(source, "control source", 64)
    clean_reason = _safe_text(reason, "control reason", MAX_TEXT)
    scope_name = _safe_text(scope, "control scope", 64)
    dedupe_key = _control_dedupe_key(task_id, run_id, checkpoint_id, trigger_id, control, scope_name)
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "control_id": _control_id(dedupe_key),
        "dedupe_key": dedupe_key,
        "task_id": task_id,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "trigger_id": trigger_id,
        "control": control,
        "scope": scope_name,
        "state": "active" if control == "provisional_stop" else "released",
        "source": source_name,
        "reason": clean_reason,
        "created_at": _now(),
    }


def _latest_bound_controls(store: BridgeStore, task_id: str, run_id: str) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(store.controls()):
        if not isinstance(row, Mapping) or row.get("schema_version") != CONTROL_SCHEMA_VERSION:
            continue
        if row.get("task_id") != task_id or str(row.get("run_id")) != str(run_id):
            continue
        if row.get("control") not in CONTROL_KINDS:
            continue
        trigger_id = row.get("trigger_id")
        checkpoint_id = row.get("checkpoint_id")
        if not isinstance(trigger_id, str) or not isinstance(checkpoint_id, str):
            continue
        scope = row.get("scope", "worker")
        if not isinstance(scope, str) or not scope.strip():
            scope = "worker"
        latest[(trigger_id, checkpoint_id, scope)] = (index, dict(row))
    return [row for _, row in sorted(latest.values(), key=lambda item: item[0])]


def _active_provisional_stop(store: BridgeStore, task_id: str, run_id: str) -> dict[str, Any] | None:
    active = [row for row in _latest_bound_controls(store, task_id, run_id) if row.get("control") == "provisional_stop" and row.get("scope", "worker") == "worker"]
    return active[-1] if active else None


def _active_scope_control(store: BridgeStore, trigger: Mapping[str, Any], *, scope: str, control: str = "provisional_stop") -> dict[str, Any] | None:
    task_id = str(trigger.get("task_id", ""))
    run_id = str(trigger.get("run_id", ""))
    trigger_id = trigger.get("trigger_id")
    rows = [
        row for row in _latest_bound_controls(store, task_id, run_id)
        if row.get("trigger_id") == trigger_id and row.get("scope", "worker") == scope and row.get("control") == control
    ]
    return rows[-1] if rows else None


def _scope_identity(session_id: Any, turn_id: Any) -> tuple[str, str]:
    session = session_id.strip() if isinstance(session_id, str) else ""
    turn = turn_id.strip() if isinstance(turn_id, str) else ""
    if session:
        return "session:" + session, "turn:" + turn if turn else ""
    if turn:
        return "turn:" + turn, "turn:" + turn
    return "", ""


def _identity_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16]


def _default_scope_action(tool_name: Any, args: Any) -> dict[str, Any] | None:
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None
    action = {"tool_name": _safe_text(tool_name, "default tool_name", 96), "arg_keys": [], "evidence_refs": []}
    if isinstance(args, Mapping):
        keys: list[str] = []
        for key in args:
            key_text = str(key).strip()
            if not key_text or any(word in key_text.casefold() for word in ("token", "secret", "password", "content")):
                continue
            keys.append(key_text[:96])
        action["arg_keys"] = sorted(set(keys))[:MAX_REFS]
        action["evidence_refs"] = _args_evidence_refs(args)
    return action


def _default_scope_request_text(value: Any) -> str:
    """Return a bounded user request without retaining tool payloads."""
    if isinstance(value, str):
        text = " ".join(value.replace("\x00", " ").split())
        return text[:300].rstrip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                candidate = item.get("text", item.get("content"))
                if isinstance(candidate, str):
                    parts.append(candidate)
        text = " ".join(" ".join(parts).replace("\x00", " ").split())
        return text[:300].rstrip()
    return ""


def _default_scope_confidence(value: Any) -> float | None:
    """Normalize a Jev confidence while preserving an explicit unknown value."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().casefold() in {"unknown", "none", "n/a", "na", "不明"}:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevRequestError("malformed_response")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise JevRequestError("malformed_response")
    return confidence


def _default_scope_confidence_guidance(label: str, confidence: float | None) -> dict[str, Any]:
    """Describe the minimal default-side rule without implementing approval."""
    if confidence is None:
        return {
            "confidence_display": DEFAULT_SCOPE_UNKNOWN_CONFIDENCE,
            "confirmation_required": False,
            "next_action": "display_unknown_only",
            "rule": "confidence不明は表示のみ。確認待ちや自動停止を追加しない。",
        }
    if label == "scope_drift" and confidence >= DEFAULT_SCOPE_CONFIDENCE_THRESHOLD:
        return {
            "confidence_display": f"{confidence:.3f}",
            "confirmation_required": True,
            "next_action": "master_confirmation_before_contrary_action",
            "rule": "判定に反する行動の前に、反対行動を取りたい理由と予定行動をmasterへ示して確認する。",
        }
    if label == "scope_drift":
        return {
            "confidence_display": f"{confidence:.3f}",
            "confirmation_required": False,
            "next_action": "readback_original_request_and_default_decision",
            "rule": "confidenceが閾値未満のため、元の依頼をreadbackしてdefaultが判断する。",
        }
    return {
        "confidence_display": f"{confidence:.3f}",
        "confirmation_required": False,
        "next_action": "default_decision",
        "rule": "Jev判定に反する行動でなければ、defaultが通常どおり判断する。",
    }


def _default_scope_advisory_message(
    action: Mapping[str, Any],
    *,
    label: str,
    confidence: float | None,
    reason: str,
) -> str:
    """Expose Jev's bounded decision and the default-side next step."""
    guidance = _default_scope_confidence_guidance(label, confidence)
    tool_name = action.get("tool_name", "unknown")
    lines = [
        DEFAULT_SCOPE_BLOCK_MESSAGE,
        f"Jev判定: {label}",
        f"confidence: {guidance['confidence_display']}",
        f"Jevの理由: {reason}",
        f"予定行動: {tool_name}",
        f"対応: {guidance['rule']}",
    ]
    return "\n".join(str(line) for line in lines)


def build_default_scope_request(
    trigger: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    session_identity: str,
    turn_identity: str = "",
    model: str = MODEL,
) -> dict[str, Any]:
    """Build the small default-side scope projection from a durable trigger."""
    if not isinstance(trigger, Mapping) or not isinstance(action, Mapping):
        raise BridgeValidationError("default scope projection must be objects")
    task_id = _safe_text(trigger.get("task_id"), "default scope task_id", 128)
    run_id = _safe_text(str(trigger.get("run_id")), "default scope run_id", 128)
    trigger_id = _safe_text(trigger.get("trigger_id"), "default scope trigger_id", 128)
    checkpoint_id = _safe_text(trigger.get("checkpoint_id"), "default scope checkpoint_id", 128)
    original_request = _safe_text(trigger.get("original_request"), "default scope original_request", 300)
    conditions = trigger.get("completion_conditions")
    if not isinstance(conditions, list) or not conditions or len(conditions) > MAX_CRITERIA:
        raise BridgeValidationError("default scope completion_conditions are invalid")
    clean_conditions = [_safe_text(item, "default scope completion condition", 300) for item in conditions]
    current_unknown = _safe_text(
        trigger.get("current_unknown", "unknown: default scope contract did not expose remaining decision points"),
        "default scope current_unknown",
        MAX_TEXT,
    )
    expected_next = _safe_text(trigger.get("next_action"), "default scope next_action", 300)
    clean_action = {
        "tool_name": _safe_text(action.get("tool_name"), "default scope current action", 96),
        "arg_keys": [
            _safe_text(item, "default scope argument key", 96)
            for item in action.get("arg_keys", [])
            if isinstance(item, str)
        ][:MAX_REFS],
        "evidence_refs": [
            _safe_text(item, "default scope action reference", 300)
            for item in action.get("evidence_refs", [])
            if isinstance(item, str)
        ][:MAX_REFS],
    }
    state = {
        "schema_version": DEFAULT_SCOPE_SCHEMA_VERSION,
        "kind": DEFAULT_SCOPE_KIND,
        "task_id": task_id,
        "run_id": run_id,
        "trigger_id": trigger_id,
        "checkpoint_id": checkpoint_id,
        "session_hash": _identity_hash(session_identity),
        "turn_hash": _identity_hash(turn_identity) if turn_identity else "",
        "original_request": original_request,
        "completion_conditions": clean_conditions,
        "current_action": clean_action,
        "expected_next_action": expected_next,
        "remaining_unknowns": current_unknown,
        "evidence_refs": [
            _safe_text(item, "default scope evidence reference", 300)
            for item in trigger.get("evidence_refs", [])
            if isinstance(item, str)
        ][:MAX_REFS],
        "confidence_policy": {
            "threshold": DEFAULT_SCOPE_CONFIDENCE_THRESHOLD,
            "high_confidence": "scope_driftなら判定に反する行動の前にmaster確認",
            "low_confidence": "元の依頼をreadbackしてdefaultが判断",
            "unknown": "不明と表示するだけで確認待ちや自動停止を追加しない",
        },
        "advisory_only": True,
        "selection_scope": "Classify only whether the bounded default action drifts from the frozen request and completion conditions. Do not execute tools or change authority.",
    }
    state_json = _json(state)
    if len(state_json.encode("utf-8")) > MAX_EVENT_BYTES:
        raise BridgeValidationError("default scope projection exceeds 16000 UTF-8 bytes")
    model_name = _safe_text(model, "default scope model", 128)
    return {
        "model": model_name,
        "state": state_json,
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Choose exactly one advisory route label. A scope_drift candidate requests default readback; Jev never executes or changes authority.",
                "criteria": {label: CRITERIA[label] for label in LABELS},
            }
        },
    }


class DefaultScopeGuard:
    """Screen each bounded default action against the frozen user request."""

    def __init__(
        self,
        ctx: Any,
        store: BridgeStore,
        *,
        decision_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        process_pending: Callable[[], Any] | None = None,
    ) -> None:
        self.ctx = ctx
        self.store = store
        self.decision_fn = decision_fn or self._decision
        self.process_pending = process_pending
        self.consumer: Consumer | None = None
        configured = _config_value(ctx, "default_scope.enabled", True)
        self.enabled = not (isinstance(configured, str) and configured.strip().lower() in {"0", "false", "no", "off"}) and configured is not False
        # Point-state is a worker-consumer policy.  The established default
        # scope guard must retain its independent 0.8 legacy route even when a
        # shared context opts the worker consumer into point-state.
        self.point_state_enabled = False
        self.point_state_policy = None
        self._lock = threading.RLock()
        self._contracts: dict[str, dict[str, Any]] = {}

    def attach_consumer(self, consumer: "Consumer") -> None:
        self.consumer = consumer

    def pre_llm_call(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        user_message: Any = None,
        completion_conditions: Any = None,
        current_unknown: Any = None,
        evidence_refs: Any = None,
        **_: Any,
    ) -> None:
        """Freeze the current user scope before the default model acts.

        This is deliberately local state: it is an advisory input to the next
        tool-call screen, not a replacement for the trusted task/run identity
        used by the worker consumer.
        """
        if not self.enabled:
            return None
        session_identity, turn_identity = _scope_identity(session_id, turn_id)
        request = _default_scope_request_text(user_message)
        if not session_identity or not request:
            return None
        conditions = completion_conditions if isinstance(completion_conditions, list) else []
        clean_conditions = [
            _safe_text(item, "default scope completion condition", 300)
            for item in conditions
            if isinstance(item, str) and item.strip()
        ][:MAX_CRITERIA]
        if not clean_conditions:
            clean_conditions = ["stay within the frozen user request and its direct completion"]
        unknown = _default_scope_request_text(current_unknown) or "unknown: default scope contract did not expose remaining decision points"
        refs = evidence_refs if isinstance(evidence_refs, list) else []
        clean_refs = [
            _safe_text(item, "default scope evidence reference", 300)
            for item in refs
            if isinstance(item, str) and item.strip()
        ][:MAX_REFS]
        contract = {
            "session_identity": session_identity,
            "turn_identity": turn_identity,
            "session_hash": _identity_hash(session_identity),
            "turn_hash": _identity_hash(turn_identity) if turn_identity else "",
            "original_request": request,
            "completion_conditions": clean_conditions,
            "current_unknown": unknown,
            "evidence_refs": clean_refs,
        }
        with self._lock:
            if turn_identity:
                self._contracts[f"{session_identity}:{turn_identity}"] = contract
            self._contracts[session_identity] = contract
        return None

    def _contract_for(self, session_identity: str, turn_identity: str) -> dict[str, Any] | None:
        with self._lock:
            if turn_identity:
                contract = self._contracts.get(f"{session_identity}:{turn_identity}")
                if contract is not None:
                    return dict(contract)
            contract = self._contracts.get(session_identity)
            return dict(contract) if contract is not None else None

    def _decision(self, request_body: Mapping[str, Any]) -> dict[str, Any]:
        primary = _configured_provider(self.ctx, "primary", PRIMARY_PROVIDER)
        fallback = _configured_provider(self.ctx, "fallback", FALLBACK_PROVIDER)
        timeout = _configured_timeout(self.ctx)
        # Keep the default scope classifier and its 0.8 threshold unchanged.
        # point_state_enabled is consumed only by Consumer._decision.
        return request_decision_with_fallback(request_body, primary=primary, fallback=fallback, timeout=timeout)

    def _latest_triggers(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.store.triggers():
            trigger_id = row.get("trigger_id")
            if isinstance(trigger_id, str):
                latest[trigger_id] = row
        return latest

    def _latest_scope_rows(self, trigger_id: str) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.store.default_scopes():
            if row.get("trigger_id") != trigger_id:
                continue
            key = row.get("dedupe_key")
            if isinstance(key, str):
                latest[key] = row
        return latest

    def _binding(self, trigger_id: str) -> dict[str, Any] | None:
        rows = list(self._latest_scope_rows(trigger_id).values())
        for row in reversed(rows):
            if isinstance(row.get("session_hash"), str) and row.get("session_hash"):
                return row
        return None

    def _released(self, trigger_id: str) -> bool:
        return any(row.get("state") == "no_intervention" for row in self._latest_scope_rows(trigger_id).values())

    def _append_scope_locked(self, record: Mapping[str, Any]) -> None:
        self.store.append_default_scope(record)

    def _reserve(self, trigger: Mapping[str, Any], action: Mapping[str, Any], session_identity: str, turn_identity: str) -> tuple[dict[str, Any] | None, bool]:
        trigger_id = _safe_text(trigger.get("trigger_id"), "default scope trigger_id", 128)
        action_hash = _identity_hash(_json(action))
        session_hash = _identity_hash(session_identity)
        dedupe_key = f"{trigger_id}:{session_hash}:{action_hash}"
        with self.store.interprocess_lock():
            latest = self._latest_scope_rows(trigger_id)
            existing = latest.get(dedupe_key)
            if existing is not None:
                return existing, False
            attempts = {
                row.get("dedupe_key")
                for row in latest.values()
                if row.get("state") in {"reserved", "candidate", "advisory", "fail_open", "identity_mismatch"}
            }
            if len(attempts) >= DEFAULT_SCOPE_MAX_CALLS_PER_TRIGGER:
                return {"state": "cap_reached", "dedupe_key": dedupe_key}, False
            record = {
                "schema_version": DEFAULT_SCOPE_SCHEMA_VERSION,
                "kind": DEFAULT_SCOPE_KIND,
                "dedupe_key": dedupe_key,
                "trigger_id": trigger_id,
                "task_id": _safe_text(trigger.get("task_id"), "default scope task_id", 128),
                "run_id": _safe_text(str(trigger.get("run_id")), "default scope run_id", 128),
                "checkpoint_id": _safe_text(trigger.get("checkpoint_id"), "default scope checkpoint_id", 128),
                "session_hash": session_hash,
                "turn_hash": _identity_hash(turn_identity) if turn_identity else "",
                "action_hash": action_hash,
                "current_action": dict(action),
                "state": "reserved",
                "request_counted": True,
                "created_at": _now(),
            }
            self._append_scope_locked(record)
            return record, True

    def _append_outcome(self, reservation: Mapping[str, Any], *, state: str, label: str, reason: str = "", confidence: float | None = None, guidance: Mapping[str, Any] | None = None, error_reason: str | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        record = dict(reservation)
        confidence_guidance = dict(guidance or _default_scope_confidence_guidance(label, confidence))
        record.update({
            "state": state,
            "label": label,
            "reason": _safe_text(reason or label, "default scope reason", MAX_TEXT),
            "confidence": float(confidence) if confidence is not None else None,
            "confidence_display": confidence_guidance["confidence_display"],
            "confidence_gate": confidence_guidance,
            "error_reason": _safe_text(error_reason, "default scope error", 96, allow_empty=True) if error_reason else None,
            "created_at": _now(),
        })
        if metadata:
            record.update(dict(metadata))
        with self.store.interprocess_lock():
            self._append_scope_locked(record)
        return record

    @staticmethod
    def _candidate(decision: Mapping[str, Any]) -> bool:
        return decision.get("label") == "scope_drift" or decision.get("drift_category") == "scope_drift" or decision.get("route_label") == "scope_drift"

    def _pending_trigger(self) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for trigger in self._latest_triggers().values():
            if trigger.get("state") != "pending_review":
                continue
            if _active_scope_control(self.store, trigger, scope="worker") is None:
                continue
            candidates.append(trigger)
        return candidates[-1] if candidates else None

    def _pending_default_trigger(self, session_identity: str) -> dict[str, Any] | None:
        session_hash = _identity_hash(session_identity)
        candidates: list[dict[str, Any]] = []
        for trigger in self._latest_triggers().values():
            if trigger.get("scope") != "default_scope" or trigger.get("state") != "pending_review":
                continue
            if trigger.get("session_hash") != session_hash or self._released(str(trigger.get("trigger_id"))):
                continue
            if _active_scope_control(self.store, trigger, scope="default") is not None:
                candidates.append(trigger)
        return candidates[-1] if candidates else None

    @staticmethod
    def _default_trigger(contract: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
        action_hash = _identity_hash(_json(action))
        session_hash = _safe_text(contract.get("session_hash"), "default scope session_hash", 128)
        turn_hash = _safe_text(contract.get("turn_hash", ""), "default scope turn_hash", 128, allow_empty=True)
        trigger_id = "dtrg-" + hashlib.sha256(f"{session_hash}:{turn_hash}:{action_hash}".encode("utf-8")).hexdigest()[:24]
        checkpoint_id = "dcp-" + hashlib.sha256(f"{trigger_id}:{action_hash}".encode("utf-8")).hexdigest()[:24]
        return {
            "schema_version": TRIGGER_SCHEMA_VERSION,
            "scope": "default_scope",
            "trigger_id": trigger_id,
            "dedupe_key": f"{trigger_id}:scope_drift",
            "task_id": "default:" + session_hash,
            "run_id": "default:" + (turn_hash or session_hash),
            "checkpoint_id": checkpoint_id,
            "source_profile": "default",
            "original_request": _safe_text(contract.get("original_request"), "default scope original_request", 300),
            "completion_conditions": list(contract.get("completion_conditions", [])),
            "observed_actions": [dict(action)],
            "evidence_refs": list(contract.get("evidence_refs", [])),
            "worker_conclusion": {"status": "unmeasured", "reason": "default scope screen has no worker conclusion"},
            "current_unknown": _safe_text(contract.get("current_unknown"), "default scope current_unknown", MAX_TEXT),
            "next_action": f"execute default tool {action.get('tool_name', 'unknown')}",
            "drift_category": "scope_drift",
            "reason": "default action may exceed the frozen user scope",
            "route_label": "scope_drift",
            "state": "pending_review",
            "created_at": _now(),
            "default_decision": "pending_readback",
            "session_hash": session_hash,
            "turn_hash": turn_hash,
        }

    def _block_for_existing_stop(self, trigger: Mapping[str, Any], session_identity: str) -> dict[str, str]:
        binding = self._binding(str(trigger["trigger_id"]))
        if binding is None or not session_identity or binding.get("session_hash") != _identity_hash(session_identity):
            return {"action": "block", "message": DEFAULT_SCOPE_IDENTITY_BLOCK_MESSAGE}
        session_hash = _identity_hash(session_identity)
        candidates = [
            row
            for row in self._latest_scope_rows(str(trigger["trigger_id"])).values()
            if row.get("state") == "candidate" and row.get("session_hash") == session_hash
        ]
        if candidates:
            candidate = candidates[-1]
            action = candidate.get("current_action")
            label = candidate.get("label")
            confidence = candidate.get("confidence")
            reason = candidate.get("reason")
            if (
                isinstance(action, Mapping)
                and isinstance(label, str)
                and isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and isinstance(reason, str)
            ):
                return {
                    "action": "block",
                    "message": _default_scope_advisory_message(
                        action,
                        label=label,
                        confidence=float(confidence),
                        reason=reason,
                    ),
                }
        return {"action": "block", "message": DEFAULT_SCOPE_BLOCK_MESSAGE}

    def _process_point_state_default(
        self,
        reservation: Mapping[str, Any],
        trigger: Mapping[str, Any],
        action: Mapping[str, Any],
        request_body: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> dict[str, str] | None:
        if self.point_state_policy is None:
            raise JevRequestError("point_state_disabled")
        point_decision = Consumer._point_decision_from_mapping(decision)
        snapshot = _point_snapshot_from_request(request_body)
        identity = PointIdentity.from_snapshot(
            str(trigger.get("task_id", "")),
            str(trigger.get("run_id", "")),
            str(reservation.get("action_hash", "")),
            snapshot,
            contract_version=str(decision.get("contract_version", POINT_STATE_CONTRACT_VERSION)),
        )
        effect = self.point_state_policy.evaluate(
            point_decision,
            identity,
            issue="default_scope",
            milestone=str(trigger.get("checkpoint_id") or "current"),
            snapshot=snapshot,
        )
        state = point_decision.state or "unknown"
        metadata = {
            "point_state": state,
            "point_state_contract_version": identity.contract_version,
            "point_state_snapshot_hash": identity.snapshot_hash,
            "point_state_event_id": identity.event_id,
            "point_state_accepted": point_decision.accepted,
            "point_state_issue": point_decision.issue,
            "point_state_action": effect.action,
            "point_state_reason": effect.reason,
            "guidance_only": effect.guidance_only,
            "completion_allowed": effect.completion_allowed,
        }
        requires_readback = effect.action == "review" and state in {"scope_or_authorization_blocked", "acceptance_ready"}
        if requires_readback:
            self._append_outcome(
                reservation,
                state="candidate",
                label="scope_drift",
                reason=effect.reason,
                confidence=point_decision.confidence,
                guidance=_default_scope_confidence_guidance("scope_drift", point_decision.confidence),
                metadata=metadata,
            )
            if self.consumer is not None:
                with self.store.interprocess_lock():
                    if trigger.get("scope") == "default_scope" and not any(row.get("trigger_id") == trigger.get("trigger_id") for row in self.store.triggers()):
                        self.store.append_trigger(trigger)
                    self.consumer._append_control_locked(trigger, "provisional_stop", source="default_scope", reason=effect.reason, scope="default")
            return {"action": "block", "message": _default_scope_advisory_message(action, label="scope_drift", confidence=point_decision.confidence, reason=effect.reason)}
        self._append_outcome(
            reservation,
            state="advisory",
            label=state,
            reason=effect.reason,
            confidence=point_decision.confidence,
            guidance=_default_scope_confidence_guidance(state, point_decision.confidence),
            metadata=metadata,
        )
        return None

    def pre_tool_call(self, tool_name: str = "", args: Any = None, session_id: str = "", turn_id: str = "", **_: Any) -> dict[str, str] | None:
        if not self.enabled or tool_name == "jev_bridge_review":
            return None
        # The consumer normally processes events asynchronously.  Only wake it
        # synchronously when a pending worker event exists; ordinary default
        # calls with no worker event remain unchanged and fail open.
        if callable(self.process_pending):
            try:
                if any(row.get("state") == "pending" for row in self.store.events()):
                    self.process_pending()
            except Exception:
                log.warning("Jev default scope pending processing skipped", exc_info=True)
        session_identity, turn_identity = _scope_identity(session_id, turn_id)
        if not session_identity:
            return None
        pending_default = self._pending_default_trigger(session_identity)
        if pending_default is not None:
            return self._block_for_existing_stop(pending_default, session_identity)
        trigger = self._pending_trigger()
        contract: dict[str, Any] | None = None
        if trigger is None:
            contract = self._contract_for(session_identity, turn_identity)
            if contract is None:
                return None
        elif self._released(str(trigger["trigger_id"])):
            return None
        active = _active_scope_control(self.store, trigger, scope="default") if trigger is not None else None
        if trigger is not None and active is not None:
            return self._block_for_existing_stop(trigger, session_identity)
        binding = self._binding(str(trigger["trigger_id"])) if trigger is not None else None
        if trigger is not None and binding is not None and binding.get("session_hash") != _identity_hash(session_identity):
            return {"action": "block", "message": DEFAULT_SCOPE_IDENTITY_BLOCK_MESSAGE}
        action = _default_scope_action(tool_name, args)
        if action is None:
            return None
        if contract is not None:
            trigger = self._default_trigger(contract, action)
            if self._released(str(trigger["trigger_id"])):
                return None
        reservation, reserved = self._reserve(trigger, action, session_identity, turn_identity)
        if reservation is None or not reserved:
            if reservation and reservation.get("state") == "candidate":
                return self._block_for_existing_stop(trigger, session_identity)
            return None
        try:
            request_body = build_default_scope_request(
                trigger,
                action,
                session_identity=session_identity,
                turn_identity=turn_identity,
                model=_config_value(self.ctx, "default_scope_model", MODEL),
            )
            decision = dict(self.decision_fn(request_body))
            label = decision.get("label")
            if label not in LABELS:
                raise JevRequestError("malformed_response")
            confidence = _default_scope_confidence(decision.get("confidence"))
            reason = decision.get("reason") if isinstance(decision.get("reason"), str) else label
            guidance = _default_scope_confidence_guidance(str(label), confidence)
            if self._candidate(decision) and confidence is not None and confidence >= DEFAULT_SCOPE_CONFIDENCE_THRESHOLD:
                self._append_outcome(reservation, state="candidate", label="scope_drift", reason=reason, confidence=confidence, guidance=guidance)
                if self.consumer is not None:
                    with self.store.interprocess_lock():
                        if trigger.get("scope") == "default_scope" and not any(row.get("trigger_id") == trigger.get("trigger_id") for row in self.store.triggers()):
                            self.store.append_trigger(trigger)
                        self.consumer._append_control_locked(trigger, "provisional_stop", source="default_scope", reason="default action may exceed the admitted scope", scope="default")
                return {"action": "block", "message": _default_scope_advisory_message(action, label=str(label), confidence=confidence, reason=reason)}
            if self._candidate(decision):
                # Low-confidence and unknown-confidence scope drift are advisory
                # only: do not add a stop or confirmation wait.  The default
                # profile retains the execution and completion decision.
                self._append_outcome(reservation, state="advisory", label="scope_drift", reason=reason, confidence=confidence, guidance=guidance)
                return None
            # insufficient_information and all other labels remain advisory only.
            self._append_outcome(reservation, state="advisory", label=str(label), reason=reason, confidence=confidence, guidance=guidance)
            return None
        except Exception as exc:
            self._append_outcome(
                reservation,
                state="fail_open",
                label="unavailable",
                reason="default scope Jev failed; original flow preserved",
                guidance=_default_scope_confidence_guidance("unavailable", None),
                error_reason=getattr(exc, "reason", type(exc).__name__),
            )
            log.info("Jev default scope screen failed open", exc_info=True)
            return None

    def record_default_decision(self, trigger_id: str, decision: str, readback: Mapping[str, Any]) -> dict[str, Any] | None:
        if decision not in {"intervene", "no_intervention"}:
            raise BridgeValidationError("default scope decision must be intervene or no_intervention")
        if not isinstance(readback, Mapping):
            raise BridgeValidationError("default scope readback must be an object")
        trigger = self._latest_triggers().get(trigger_id)
        if trigger is None:
            return None
        active = _active_scope_control(self.store, trigger, scope="default")
        if active is None:
            return {"trigger_id": trigger_id, "scope": "default_scope", "state": "already_settled"}
        if readback.get("task_id") != trigger["task_id"] or str(readback.get("run_id")) != str(trigger["run_id"]):
            raise BridgeValidationError("default scope readback task/run does not match trigger")
        if readback.get("original_request") != trigger["original_request"]:
            raise BridgeValidationError("default scope readback original_request does not match trigger")
        if readback.get("completion_conditions") != trigger["completion_conditions"]:
            raise BridgeValidationError("default scope readback completion_conditions do not match trigger")
        if readback.get("evidence_refs") != trigger["evidence_refs"]:
            raise BridgeValidationError("default scope readback evidence_refs do not match trigger")
        next_action = _safe_text(readback.get("next_action"), "default scope next_action", 300)
        conclusion = Consumer._readback_worker_conclusion(readback.get("worker_conclusion"))
        binding = self._binding(trigger_id)
        session_identity, _ = _scope_identity(readback.get("session_id"), readback.get("turn_id"))
        if binding is not None and (not session_identity or binding.get("session_hash") != _identity_hash(session_identity)):
            raise BridgeValidationError("default scope readback session identity does not match the active stop")
        state = "intervention_requested" if decision == "intervene" else "no_intervention"
        record = {
            "schema_version": DEFAULT_SCOPE_SCHEMA_VERSION,
            "kind": DEFAULT_SCOPE_KIND,
            "dedupe_key": f"readback:{trigger_id}",
            "trigger_id": trigger_id,
            "task_id": trigger["task_id"],
            "run_id": str(trigger["run_id"]),
            "checkpoint_id": trigger["checkpoint_id"],
            "session_hash": binding.get("session_hash") if binding else _identity_hash(session_identity),
            "turn_hash": _identity_hash(_scope_identity(readback.get("session_id"), readback.get("turn_id"))[1]) if _scope_identity(readback.get("session_id"), readback.get("turn_id"))[1] else "",
            "state": state,
            "label": "scope_drift",
            "next_action": next_action,
            "worker_conclusion": conclusion,
            "created_at": _now(),
        }
        with self.store.interprocess_lock():
            self._append_scope_locked(record)
            if self.consumer is not None:
                if decision == "no_intervention":
                    self.consumer._append_control_locked(trigger, "release", source="default", reason="default scope readback selected no_intervention", scope="default")
                else:
                    self.consumer._append_control_locked(trigger, "provisional_stop", source="default", reason="default scope readback selected intervention", scope="default")
        result = dict(record)
        result.update({"scope": "default_scope", "default_decision": decision})
        return result

    def review_tool(self, args: Mapping[str, Any] | None = None, **_: Any) -> str:
        payload = dict(args or {})
        trigger_id = _safe_text(payload.get("trigger_id"), "trigger_id", 128)
        readback = payload.get("readback")
        if not isinstance(readback, Mapping):
            raise BridgeValidationError("default scope readback must be an object")
        return _json(self.record_default_decision(trigger_id, payload.get("decision"), readback) or {"trigger_id": trigger_id, "scope": "default_scope", "state": "already_settled"})


def _binding_matches_live_assignment(binding: Mapping[str, Any], profile_name: str) -> bool:
    """Recheck the observer binding against the trusted running board row."""
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        board = binding["board"]
        task_id = binding["task_id"]
        run_id = binding["run_id"]
        with kbc.connect_closing(board=board) as conn:
            task = kb.get_task(conn, task_id)
            run = kb.get_run(conn, int(run_id))
            latest = kb.latest_run(conn, task_id)
        if task is None or run is None or latest is None:
            return False
        if getattr(run, "task_id", None) != task_id or getattr(run, "status", None) != "running":
            return False
        if getattr(task, "status", None) != "running":
            return False
        if str(getattr(task, "current_run_id", "")) != run_id or str(getattr(latest, "id", "")) != run_id:
            return False
        if getattr(task, "assignee", None) != profile_name:
            return False
        run_profile = getattr(run, "profile", None)
        return not run_profile or run_profile == profile_name
    except Exception:
        return False


def _worker_binding(profile_name: str) -> tuple[str, str] | None:
    """Read the original process/profile binding used by the stop guard."""
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    profile = os.environ.get("HERMES_PROFILE", "").strip()
    if profile != profile_name or profile == "default" or not task_id or not run_id:
        return None
    try:
        return _safe_text(task_id, "worker task_id", 128), _safe_text(run_id, "worker run_id", 128)
    except BridgeValidationError:
        return None


def _trusted_current_board() -> str | None:
    try:
        from hermes_cli import kanban_db as kb
        get_current_board = getattr(kb, "get_current_board", None)
        if not callable(get_current_board):
            return None
        board = get_current_board()
        if not isinstance(board, str) or not board or board != board.strip() or board != board.lower():
            return None
        return board
    except Exception:
        return None


def _self_block_binding(profile_name: str, bridge_root: str | Path | None, expected: tuple[str, str]) -> dict[str, Any] | None:
    """Establish a trusted current task/run only for the native self-block path."""
    task_id, run_id = expected
    if os.environ.get("HERMES_PROFILE", "").strip() != profile_name or profile_name == "default":
        return None
    current_board = _trusted_current_board()
    if current_board is None:
        return None
    try:
        load_binding = globals().get("_load_current_worker_binding")
        if callable(load_binding) and bridge_root:
            binding = load_binding(bridge_root, profile_name)
            if binding is None:
                return None
            if binding.get("task_id") != task_id or binding.get("run_id") != run_id:
                return None
            if binding.get("worker_profile") != profile_name:
                return None
            if binding.get("dispatcher_profile") not in {"", "default"}:
                return None
            if binding.get("assignee") not in {"", profile_name}:
                return None
            board = binding.get("board") or current_board
            if board != current_board:
                return None
            candidate = dict(binding)
            candidate["board"] = board
            if not _binding_matches_live_assignment(candidate, profile_name):
                return None
            return candidate
        contract = _load_dispatcher_contract()
        if contract.task_id != task_id or contract.run_id != run_id or contract.profile != profile_name:
            return None
        candidate = {
            "task_id": task_id,
            "run_id": run_id,
            "worker_profile": profile_name,
            "dispatcher_profile": "default",
            "assignee": profile_name,
            "board": current_board,
        }
        return candidate if _binding_matches_live_assignment(candidate, profile_name) else None
    except Exception:
        return None


def _kanban_block_args_match(args: Any, binding: Mapping[str, Any]) -> bool:
    """Validate only the native self-block shape; native handler still runs."""
    if not isinstance(args, dict) or set(args) - {"task_id", "reason", "kind", "board"}:
        return False
    reason = args.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return False
    kind = args.get("kind")
    if kind is not None and (not isinstance(kind, str) or kind not in {"dependency", "needs_input", "capability", "transient"}):
        return False
    if "task_id" in args and (not isinstance(args["task_id"], str) or not args["task_id"] or args["task_id"] != binding.get("task_id")):
        return False
    trusted_board = binding.get("board")
    current_board = _trusted_current_board()
    if not isinstance(trusted_board, str) or trusted_board != current_board:
        return False
    if "board" in args and (not isinstance(args["board"], str) or args["board"] != trusted_board):
        return False
    return True


class Consumer:
    def __init__(self, ctx: Any, store: BridgeStore, decision_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None):
        self.ctx, self.store = ctx, store
        self.decision_fn = decision_fn or self._decision
        self._threads: set[threading.Thread] = set()
        self._review_requested = False
        self._lock = threading.RLock()
        self.scope_guard: DefaultScopeGuard | None = None
        self.point_state_enabled = _config_bool(ctx, POINT_STATE_ENABLED_CONFIG, False)
        self.point_state_policy = PointStatePolicy(store_path=store.root / "point-state-ledger.json") if self.point_state_enabled else None

    def _decision(self, event: Mapping[str, Any]) -> dict[str, Any]:
        primary = _configured_provider(self.ctx, "primary", PRIMARY_PROVIDER)
        fallback = _configured_provider(self.ctx, "fallback", FALLBACK_PROVIDER)
        timeout = _configured_timeout(self.ctx)
        request = build_jev_request(event, model=primary.model, point_state_enabled=self.point_state_enabled)
        parser = _parse_point_state_for_bridge if self.point_state_enabled else parse_jev_response
        return request_decision_with_fallback(request, primary=primary, fallback=fallback, timeout=timeout, parser=parser)

    def _config(self, key: str, default: Any = None) -> Any:
        try:
            return self.ctx.get_config(key, default)
        except Exception:
            return default

    _ATTEMPT_STATES = {"reserved", "completed", "failed", "ready", "triggered", "no_intervention_candidate"}

    @classmethod
    def _counted_rows(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in rows if row.get("state") in cls._ATTEMPT_STATES and row.get("request_counted", True) is not False]

    def _latest_ledger(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.store.ledger():
            event_key = row.get("event_key")
            if isinstance(event_key, str):
                latest[event_key] = row
        return latest

    def _latest_triggers(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.store.triggers():
            trigger_id = row.get("trigger_id")
            if isinstance(trigger_id, str):
                latest[trigger_id] = row
        return latest

    def _eligible(self, event: Mapping[str, Any]) -> bool:
        if not _event_projection_is_eligible(event):
            return False
        latest = self._latest_ledger()
        event_key = event.get("event_key")
        existing = latest.get(event_key)
        if existing is not None and existing.get("state") in self._ATTEMPT_STATES:
            return False
        run_key = f"{event['task_id']}:{event['run_id']}"
        run_rows = self._counted_rows([row for row in latest.values() if row.get("run_key") == run_key])
        return len(run_rows) < min(MAX_CALLS_PER_RUN, MAX_LIVE_CALLS_PER_RUN)

    def _reserve(self, event: Mapping[str, Any]) -> bool:
        with self.store.interprocess_lock():
            if not self._eligible(event):
                return False
            self.store.append_ledger({"event_key": event["event_key"], "run_key": f"{event['task_id']}:{event['run_id']}", "task_id": event["task_id"], "run_id": event["run_id"], "checkpoint_id": event["checkpoint_id"], "state": "reserved", "reserved_cost_usd": str(RESERVATION_USD), "request_counted": True, "jst_day": _jst_day(), "timestamp": _now()})
            return True

    def _append_control_locked(self, trigger: Mapping[str, Any], control: str, *, source: str, reason: str, scope: str = "worker") -> dict[str, Any]:
        record = _control_record(trigger, control, source=source, reason=reason, scope=scope)
        if any(row.get("dedupe_key") == record["dedupe_key"] for row in self.store.controls()):
            return next(row for row in reversed(self.store.controls()) if row.get("dedupe_key") == record["dedupe_key"])
        self.store.append_control(record)
        return record

    def _append_trigger(
        self,
        event: Mapping[str, Any],
        decision: Mapping[str, Any],
        category: str | None,
        *,
        point_effect: ControlEffect | None = None,
        point_identity: PointIdentity | None = None,
    ) -> dict[str, Any] | None:
        if category is None:
            return None
        dedupe_key = f"{event['task_id']}:{event['run_id']}:{event['checkpoint_id']}:{category}"
        trigger_id = _trigger_id(dedupe_key)
        with self.store.interprocess_lock():
            latest = self._latest_triggers()
            if any(row.get("dedupe_key") == dedupe_key for row in latest.values()):
                return None
            if any(
                row.get("task_id") == event.get("task_id")
                and str(row.get("run_id")) == str(event.get("run_id"))
                and row.get("state") == "pending_review"
                for row in latest.values()
            ):
                return None
            record = {
                "schema_version": TRIGGER_SCHEMA_VERSION,
                "trigger_id": trigger_id,
                "dedupe_key": dedupe_key,
                "task_id": event["task_id"],
                "run_id": event["run_id"],
                "checkpoint_id": event["checkpoint_id"],
                "source_profile": event.get("profile", "unknown"),
                "original_request": event["original_request"],
                "completion_conditions": list(event["criteria"]),
                "observed_actions": _trigger_observed_actions(event),
                "evidence_refs": list(event["evidence_refs"]),
                "worker_conclusion": _trigger_worker_conclusion(event),
                "current_unknown": _safe_text(event.get("current_unknown", "unknown: default scope contract did not expose remaining decision points"), "current_unknown", MAX_TEXT),
                "next_action": event["next_action"],
                "drift_category": category,
                "reason": _trigger_reason(category, decision),
                "route_label": _safe_text(decision.get("route_label") or decision.get("label"), "route_label", 64),
                "state": "pending_review",
                "created_at": _now(),
                "default_decision": "pending_readback",
            }
            if point_effect is not None:
                guidance = {
                    "action": point_effect.action,
                    "reason": point_effect.reason,
                    "instructions": list(point_effect.instructions),
                    "verification_name": point_effect.verification_name,
                    "guidance_only": point_effect.guidance_only,
                    "completion_allowed": point_effect.completion_allowed,
                }
                record.update({
                    "point_state": decision.get("point_state"),
                    "point_state_milestone": decision.get("point_state_milestone"),
                    "point_state_action": point_effect.action,
                    "point_state_reason": point_effect.reason,
                    "point_state_instructions": list(point_effect.instructions),
                    "point_state_verification_name": point_effect.verification_name,
                    "point_state_guidance": guidance,
                    "guidance_only": point_effect.guidance_only,
                    "completion_allowed": point_effect.completion_allowed,
                })
                if point_identity is not None:
                    record.update({
                        "point_state_contract_version": point_identity.contract_version,
                        "point_state_snapshot_hash": point_identity.snapshot_hash,
                        "point_state_event_id": point_identity.event_id,
                    })
            self.store.append_trigger(record)
            self._append_control_locked(record, "provisional_stop", source="jev", reason=record["reason"])
            return record

    @staticmethod
    def _point_decision_from_mapping(decision: Mapping[str, Any]) -> PointDecision:
        candidate = decision.get("_point_decision")
        if isinstance(candidate, PointDecision):
            return candidate
        response = {
            "state": decision.get("point_state", decision.get("state", decision.get("label"))),
            "confidence": decision.get("confidence"),
        }
        return parse_point_state_response(response)

    @staticmethod
    def _point_state_trigger_category(state: str, effect: ControlEffect) -> str | None:
        """Map every non-continue effect to a visible default review category."""
        if effect.action == "continue_scoped_action":
            return None
        if state == "scope_or_authorization_blocked" or effect.reason == "guard_precedence":
            return "scope_drift"
        if state == "acceptance_ready":
            return "completion_candidate"
        # Corrections, named verification, bounded refresh, abstentions and
        # exhausted budgets are guidance/hold cases, never silent completion.
        return "evidence_mismatch"

    def _process_point_state_event(self, event: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
        if self.point_state_policy is None:
            raise JevRequestError("point_state_disabled")
        point_decision = self._point_decision_from_mapping(decision)
        snapshot = _point_snapshot_from_event(event)
        identity = PointIdentity.from_snapshot(
            str(event.get("task_id", "")),
            str(event.get("run_id", "")),
            str(event.get("event_key") or event.get("checkpoint_id", "")),
            snapshot,
            contract_version=str(decision.get("contract_version", POINT_STATE_CONTRACT_VERSION)),
        )
        guards = event.get("guards", []) if isinstance(event.get("guards", []), list) else []
        issue = str(event.get("category") or "point_state")
        # The checkpoint is the event identity only.  Budgets are scoped to the
        # stable task/run/issue/named-milestone contract key.
        milestone = _point_state_milestone(event)
        effect = self.point_state_policy.evaluate(
            point_decision,
            identity,
            issue=issue,
            milestone=milestone,
            snapshot=snapshot,
            guards=guards,
        )
        state = point_decision.state or "unknown"
        trigger_category = self._point_state_trigger_category(state, effect)
        trigger = None
        if trigger_category is not None:
            trigger_decision = {
                "label": trigger_category,
                "route_label": state,
                "confidence": point_decision.confidence if point_decision.confidence is not None else 0.0,
                "reason": effect.reason,
                "point_state": state,
                "point_state_milestone": milestone,
            }
            trigger = self._append_trigger(
                event,
                trigger_decision,
                trigger_category,
                point_effect=effect,
                point_identity=identity,
            )
            # A prior point-state hold for this task/run is the visible review
            # destination for later events; do not create a second stop.
            if trigger is None:
                for row in reversed(self.store.triggers()):
                    if (
                        row.get("task_id") == event.get("task_id")
                        and str(row.get("run_id")) == str(event.get("run_id"))
                        and row.get("state") == "pending_review"
                    ):
                        trigger = row
                        break
        review_state = "triggered" if trigger is not None else "completed"
        guidance = {
            "action": effect.action,
            "reason": effect.reason,
            "instructions": list(effect.instructions),
            "verification_name": effect.verification_name,
            "guidance_only": effect.guidance_only,
            "completion_allowed": effect.completion_allowed,
        }
        self.store.append_review({
            "event_key": event["event_key"],
            "state": review_state,
            "label": state,
            "point_state": state,
            "point_state_contract_version": identity.contract_version,
            "point_state_snapshot_hash": identity.snapshot_hash,
            "point_state_event_id": identity.event_id,
            "point_state_milestone": milestone,
            "point_state_accepted": point_decision.accepted,
            "point_state_issue": point_decision.issue,
            "point_state_confidence": point_decision.confidence,
            "point_state_action": effect.action,
            "point_state_reason": effect.reason,
            "point_state_guidance": guidance,
            "guidance_only": effect.guidance_only,
            "completion_allowed": effect.completion_allowed,
            "trigger_id": trigger.get("trigger_id") if trigger else None,
            "timestamp": _now(),
        })
        self.store.append_ledger({
            "event_key": event["event_key"],
            "run_key": f"{event['task_id']}:{event['run_id']}",
            "state": review_state,
            "point_state_action": effect.action,
            "reserved_cost_usd": str(RESERVATION_USD),
            "request_counted": True,
            "jst_day": _jst_day(),
            "timestamp": _now(),
        })
        return {"decision": decision, "effect": effect, "trigger": trigger}

    def process_pending(self) -> bool:
        prior_reviews = self.store.reviews()
        for event in self.store.events():
            if event.get("state") != "pending" or not self._eligible(event):
                continue
            with self._lock:
                if not self._reserve(event):
                    return False
            try:
                decision = dict(self.decision_fn(event))
                if self.point_state_enabled:
                    self._process_point_state_event(event, decision)
                    return True
                if decision.get("label") not in LABELS or not isinstance(decision.get("confidence"), (int, float)):
                    raise JevRequestError("malformed_response")
                category = _trigger_category(event, decision, prior_reviews)
                trigger = self._append_trigger(event, decision, category)
                state = "triggered" if trigger is not None else "no_intervention_candidate" if category is None else "completed"
                self.store.append_review({
                    "event_key": event["event_key"],
                    "state": state,
                    "label": decision["label"],
                    "route_label": decision.get("route_label", decision["label"]),
                    "drift_category": category,
                    "confidence": float(decision["confidence"]),
                    "probabilities": decision.get("probabilities"),
                    "usage": decision.get("usage"),
                    "cost": str(_cost(decision.get("usage"))),
                    "trigger_id": trigger.get("trigger_id") if trigger else None,
                    "timestamp": _now(),
                })
                self.store.append_ledger({"event_key": event["event_key"], "run_key": f"{event['task_id']}:{event['run_id']}", "state": state, "reserved_cost_usd": str(RESERVATION_USD), "request_counted": True, "jst_day": _jst_day(), "timestamp": _now()})
            except Exception as exc:
                self.store.append_review({"event_key": event["event_key"], "state": "failed", "reason": getattr(exc, "reason", "request_error"), "timestamp": _now()})
                self.store.append_ledger({"event_key": event["event_key"], "run_key": f"{event['task_id']}:{event['run_id']}", "state": "failed", "reserved_cost_usd": str(RESERVATION_USD), "request_counted": True, "jst_day": _jst_day(), "timestamp": _now()})
            return True
        return False

    @staticmethod
    def _readback_worker_conclusion(value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            return _safe_text(value, "worker_conclusion", MAX_TEXT)
        if isinstance(value, Mapping):
            status = value.get("status")
            reason = value.get("reason")
            if isinstance(status, str) and status.strip() and isinstance(reason, str) and reason.strip():
                return {"status": _safe_text(status, "worker_conclusion.status", 64), "reason": _safe_text(reason, "worker_conclusion.reason", MAX_TEXT)}
        raise BridgeValidationError("default readback must include worker_conclusion or an explicit unmeasured marker")

    def record_default_decision(self, trigger_id: str, decision: str, readback: Mapping[str, Any]) -> dict[str, Any] | None:
        if decision not in {"intervene", "no_intervention"}:
            raise BridgeValidationError("default decision must be intervene or no_intervention")
        if not isinstance(readback, Mapping):
            raise BridgeValidationError("default readback must be an object")
        with self.store.interprocess_lock():
            trigger = self._latest_triggers().get(trigger_id)
            if trigger is None or trigger.get("state") != "pending_review":
                return None
            if readback.get("task_id") != trigger["task_id"] or str(readback.get("run_id")) != str(trigger["run_id"]):
                raise BridgeValidationError("default readback task/run does not match trigger")
            if readback.get("original_request") != trigger["original_request"]:
                raise BridgeValidationError("default readback original_request does not match trigger")
            conditions = readback.get("completion_conditions")
            if not isinstance(conditions, list) or conditions != trigger["completion_conditions"]:
                raise BridgeValidationError("default readback completion_conditions do not match trigger")
            refs = readback.get("evidence_refs")
            if not isinstance(refs, list) or refs != trigger["evidence_refs"]:
                raise BridgeValidationError("default readback evidence_refs do not match trigger")
            next_action = _safe_text(readback.get("next_action"), "default next_action", 300)
            conclusion = self._readback_worker_conclusion(readback.get("worker_conclusion"))
            updated = dict(trigger)
            updated.update({
                "worker_conclusion": conclusion,
                "next_action": next_action,
                "state": "intervention_requested" if decision == "intervene" else "no_intervention",
                "created_at": _now(),
                "default_decision": {"decision": decision, "basis": "default_readback", "readback_refs": refs[:MAX_REFS]},
            })
            self.store.append_trigger(updated)
            if decision == "no_intervention":
                self._append_control_locked(updated, "release", source="default", reason="default readback selected no_intervention")
            return updated

    def review_tool(self, args: Mapping[str, Any] | None = None, **kwargs: Any) -> str:
        """Expose only the durable trigger/readback seam to the default profile."""
        payload = dict(args or {})
        if not payload:
            pending = []
            session_id = kwargs.get("session_id", "")
            session_identity, _ = _scope_identity(session_id, "")
            for row in self._latest_triggers().values():
                if row.get("state") != "pending_review":
                    continue
                if row.get("scope") == "default_scope" and self.scope_guard is not None:
                    if _active_scope_control(self.store, row, scope="default") is None:
                        continue
                    binding = self.scope_guard._binding(row["trigger_id"])
                    if not session_identity or (binding and binding.get("session_hash") != _identity_hash(session_identity)):
                        continue
                    row = dict(row)
                    row["readback_template"] = {
                        key: row[key] for key in (
                            "task_id", "run_id", "original_request", "completion_conditions",
                            "evidence_refs", "worker_conclusion", "next_action",
                        )
                    }
                    row["readback_template"].update({"scope": "default_scope", "session_id": session_id})
                pending.append(row)
            return _json(pending)
        trigger_id = _safe_text(payload.get("trigger_id"), "trigger_id", 128)
        decision = payload.get("decision")
        readback = payload.get("readback")
        if not isinstance(readback, Mapping):
            raise BridgeValidationError("readback must be an object")
        if self.scope_guard is not None and (payload.get("scope") == "default_scope" or readback.get("scope") == "default_scope"):
            # Bind model-tool decisions to the host session, not a model-supplied identity.
            session_id = kwargs.get("session_id")
            if session_id:
                if readback.get("session_id") not in (None, "", session_id):
                    raise BridgeValidationError("default scope readback session identity does not match caller")
                payload["readback"] = dict(readback, session_id=session_id)
            return self.scope_guard.review_tool(payload)
        updated = self.record_default_decision(trigger_id, decision, readback)
        return _json(updated or {"trigger_id": trigger_id, "state": "already_settled"})

    def review_pending(self, readback_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not callable(readback_fn):
            raise BridgeValidationError("default readback function is required")
        results: list[dict[str, Any]] = []
        for trigger in self._latest_triggers().values():
            if trigger.get("state") != "pending_review":
                continue
            readback = readback_fn(dict(trigger))
            if not isinstance(readback, Mapping):
                raise BridgeValidationError("default readback function must return an object")
            decision = readback.get("decision")
            updated = self.record_default_decision(trigger["trigger_id"], decision, readback)
            if updated is not None:
                results.append(updated)
        return results

    def _review_worker(self) -> None:
        while True:
            with self._lock:
                self._review_requested = False
            try:
                while self.process_pending():
                    pass
            except Exception:
                log.warning("Jev bridge consumer skipped advisory; no retry or authority change", exc_info=True)
            with self._lock:
                if self._review_requested:
                    continue
                self._threads.discard(threading.current_thread())
                return

    def post_tool_call(self, **kwargs: Any) -> None:
        with self._lock:
            self._review_requested = True
            if self._threads:
                return
            thread = threading.Thread(target=self._review_worker, daemon=True, name="jev-bridge-review")
            self._threads.add(thread)
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._threads.discard(thread)
            raise


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer"}, "synthetic": {"type": "boolean"}, "sensitive": {"type": "boolean"},
            "projection_safe": {"type": "boolean"}, "category": {"type": "string"}, "task_id": {"type": "string"},
            "run_id": {"type": "string"}, "checkpoint_id": {"type": "string"}, "criteria": {"type": "array", "items": {"type": "string"}},
            "observation": {"type": "string"}, "recent_steps": {"type": "array", "items": {"type": "object"}},
            "evidence_refs": {"type": "array", "items": {"type": "string"}}, "completed_tools": {"type": "integer"},
            "tools_since_previous": {"type": "integer"}, "changed_evidence": {"type": "boolean"},
            "changed_evidence_age_seconds": {"type": "number"}, "original_request": {"type": "string"},
            "current_unknown": {"type": "string"}, "next_action": {"type": "string"},
        },
        "required": ["schema_version", "category", "task_id", "run_id", "checkpoint_id", "criteria", "observation", "evidence_refs", "completed_tools", "tools_since_previous", "changed_evidence"],
    }


def _config_value(ctx: Any, key: str, default: Any = None) -> Any:
    try:
        return ctx.get_config(key, default)
    except Exception:
        return default


def _config_bool(ctx: Any, key: str, default: bool = False) -> bool:
    value = _config_value(ctx, key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return value is not False and value is not None


def _profile_list(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def register(ctx: Any) -> None:
    root = _config_value(ctx, "bridge_dir", "") or os.environ.get("HERMES_JEV_BRIDGE_DIR", "")
    if not root:
        log.warning("jev-route-screening disabled: bridge_dir is not configured")
        return
    profile = str(getattr(ctx, "profile_name", "") or os.environ.get("HERMES_PROFILE", "")).strip()
    role = _config_value(ctx, "role", "")
    producer_profiles = _profile_list(_config_value(ctx, "producer_profiles", []))
    consumer_profiles = _profile_list(_config_value(ctx, "consumer_profiles", []))
    # Read old pilot fixtures without widening the live profile set. Real config
    # now declares role/profile lists explicitly; unknown or unconfigured names
    # remain disabled.
    if not role and profile == "ops":
        role, producer_profiles = "producer", {"ops"}
    elif not role and profile == "default":
        role, consumer_profiles = "consumer", {"default"}
    if role == "producer" and profile and profile != "default" and profile in producer_profiles:
        store = BridgeStore(root)
        producer = Producer(store, profile_name=profile, bridge_root=root)
        ctx.register_tool(name="jev_bridge_checkpoint", toolset="jev_route_screening", schema=_schema(), handler=producer.checkpoint_tool, description="Submit an optional bounded worker checkpoint; normal tool hooks also create checkpoints when cadence and evidence thresholds are met.")
        ctx.register_hook("pre_tool_call", producer.pre_tool_call)
        ctx.register_hook("post_tool_call", producer.post_tool_call)
    elif role == "consumer" and profile == "default" and profile in consumer_profiles:
        store = BridgeStore(root)
        consumer = Consumer(ctx, store)
        scope_guard = DefaultScopeGuard(ctx, store, process_pending=consumer.process_pending)
        scope_guard.attach_consumer(consumer)
        consumer.scope_guard = scope_guard
        ctx.register_tool(
            name="jev_bridge_review", toolset="jev_route_screening",
            schema={
                "name": "jev_bridge_review",
                "description": "Inspect pending Jev stops with empty arguments, including same-session default readback templates. After reviewing the evidence, submit an explicit decision with scope, trigger_id and readback; no_intervention releases only the matched stop. This tool remains available during a provisional stop.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "enum": ["worker", "default_scope"]},
                        "trigger_id": {"type": "string"},
                        "decision": {"type": "string", "enum": ["intervene", "no_intervention"]},
                        "readback": {"type": "object", "description": "Reviewed evidence from readback_template; retain task/run, original request, completion conditions and evidence references; provide next_action and worker_conclusion. The host session binds default-scope decisions."},
                    },
                    "additionalProperties": False,
                },
            },
            handler=consumer.review_tool,
            description="Inspect Jev stops and record an explicit default readback decision.",
        )
        ctx.register_hook("pre_llm_call", scope_guard.pre_llm_call)
        ctx.register_hook("pre_tool_call", scope_guard.pre_tool_call)
        ctx.register_hook("post_tool_call", consumer.post_tool_call)
    else:
        log.info("jev-route-screening disabled for unconfigured profile=%s role=%s", profile or "unknown", role or "unknown")
