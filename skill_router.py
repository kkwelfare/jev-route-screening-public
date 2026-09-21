"""Advisory targeted-reading router for the ``hermes-agent`` skill.

The router is deliberately a preflight hint, not a loader or policy gate.  It
asks Jev for the smallest reference bundle, preserves local hard invariants,
and records which ``skill_view`` calls actually followed the hint.  The normal
Hermes skill loader and worker authority remain unchanged.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from . import bridge

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SKILL_NAME = "hermes-agent"
MAX_REQUEST_CHARS = 4_000
MAX_CONTEXT_CHARS = 6_000
MAX_STATE_BYTES = 16_000
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_TIMEOUT_SECONDS = 15.0
MAX_PROVIDER_ATTEMPTS = 2

# The manifest is intentionally explicit.  Jev chooses a route label; the
# local mapping, not Jev, decides which repository-relative files that label
# can name.  This keeps an untrusted response from becoming a file path.
ROUTE_BUNDLES: dict[str, tuple[str, ...]] = {
    "main_only": (),
    "cli_reference": ("references/cli-reference.md",),
    "slash_commands_reference": ("references/slash-commands.md",),
    "providers_and_models_reference": ("references/providers-and-models.md",),
    "configuration_reference": ("references/configuration.md",),
    "security_privacy_reference": ("references/security-privacy.md",),
    "background_systems_reference": ("references/background-systems.md",),
    "themes_bundle": ("references/themes.md", "templates/skin.yaml"),
    "desktop_plugins_bundle": ("references/desktop-plugins.md", "templates/plugin.js"),
    "troubleshooting_reference": ("references/troubleshooting.md",),
    "native_mcp_reference": ("references/native-mcp.md",),
    "project_context_files_reference": ("references/project-context-files.md",),
    "webhooks_reference": ("references/webhooks.md",),
    "tui_widgets_reference": ("references/tui-widgets.md",),
    "windows_quirks_reference": ("references/windows-quirks.md",),
    "contributor_guide_reference": ("references/contributor-guide.md",),
    "delegate_concurrency_reference": ("references/delegate-task-concurrency-diagnosis.md",),
    "petdex_reference": ("references/petdex.md",),
    "portal_auth_reference": ("references/portal-auth-for-third-party-apps.md",),
}

ROUTE_CRITERIA: dict[str, str] = {
    "main_only": "Read only the hub SKILL.md; no deeper reference is needed.",
    "cli_reference": "Hermes CLI commands, subcommands, flags, or one-shot commands.",
    "slash_commands_reference": "In-session slash commands and their behavior.",
    "providers_and_models_reference": "Provider, model, API endpoint, API key, or OAuth setup.",
    "configuration_reference": "config.yaml, toolsets, runtime settings, or display settings.",
    "security_privacy_reference": "Secrets, redaction, privacy, PII, approvals, or authorization boundaries.",
    "background_systems_reference": "Delegation, cron, curator, Kanban, or background orchestration.",
    "themes_bundle": "Custom themes, active skins, colors, or skin templates.",
    "desktop_plugins_bundle": "Hermes desktop panes, commands, or desktop plugin code.",
    "troubleshooting_reference": "Troubleshooting, diagnostics, or common failure recovery.",
    "native_mcp_reference": "Native MCP servers, MCP tools, or MCP integration.",
    "project_context_files_reference": "Project context files, AGENTS.md, or workspace context.",
    "webhooks_reference": "Webhooks, inbound hooks, or webhook configuration.",
    "tui_widgets_reference": "TUI widgets, dock widgets, or terminal UI extensions.",
    "windows_quirks_reference": "Windows-specific behavior, paths, or process quirks.",
    "contributor_guide_reference": "Contributing to Hermes or repository development guidance.",
    "delegate_concurrency_reference": "Delegate-task concurrency, child workers, or parallel delegation failures.",
    "petdex_reference": "Petdex mascot installation or selection.",
    "portal_auth_reference": "Third-party portal authentication flows.",
}

ALL_REFERENCE_FILES = frozenset(path for paths in ROUTE_BUNDLES.values() for path in paths)
HARD_INVARIANTS = (
    "Read the hermes-agent SKILL.md hub as the authoritative safety and routing baseline.",
    "Security, authentication, configuration, and authority guidance is never omitted because Jev is uncertain.",
    "Jev output is advisory only; it cannot execute tools, change permissions, or mutate skills/configuration.",
    "The existing worker/skill loader decides actual reads and may expand beyond the candidates.",
)

_MANDATORY_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        ("secret", "credential", "password", "token", "pii", "privacy", "redact", "approval", "authorize", "permission"),
        ("references/security-privacy.md",),
        "mandatory_security_or_authority",
    ),
    (
        ("auth", "oauth", "login", "authentication"),
        ("references/security-privacy.md", "references/portal-auth-for-third-party-apps.md"),
        "mandatory_authentication",
    ),
    (
        ("config", "config.yaml", "setting", "toolset", "runtime", "environment"),
        ("references/configuration.md",),
        "mandatory_configuration",
    ),
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _jst_day() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo("Asia/Tokyo")).date().isoformat()


def _hash_value(value: Any) -> str:
    text = str(value or "")
    return "sha256:" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _clean_text(value: Any, limit: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        return "" if allow_empty else "unknown"
    text = " ".join(value.replace("\x00", " ").split())
    if not text:
        return "" if allow_empty else "unknown"
    return text[:limit].rstrip() if len(text) > limit else text


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_float(value: Any, default: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if 0 < parsed <= maximum else default


def _as_int(value: Any, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if 1 <= parsed <= maximum else default


def _config_value(ctx: Any, key: str, default: Any = None) -> Any:
    try:
        return ctx.get_config(key, default)
    except Exception:
        return default


def _user_text(value: Any) -> str:
    if isinstance(value, str):
        return _clean_text(value, MAX_REQUEST_CHARS, allow_empty=True)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return _clean_text(" ".join(parts), MAX_REQUEST_CHARS, allow_empty=True)
    return ""


def _mandatory_references(request: str) -> tuple[list[str], str | None]:
    lowered = request.casefold()
    refs: list[str] = []
    reasons: list[str] = []
    for keywords, candidates, reason in _MANDATORY_RULES:
        if any(keyword in lowered for keyword in keywords):
            reasons.append(reason)
            for candidate in candidates:
                if candidate not in refs:
                    refs.append(candidate)
    return refs, "+".join(reasons) if reasons else None


def _files_for_route(route: str) -> list[str]:
    return list(ROUTE_BUNDLES.get(route, ()))


def _required_reads(route: str, request: str) -> tuple[list[str], str]:
    refs = _files_for_route(route)
    mandatory, reason = _mandatory_references(request)
    for candidate in mandatory:
        if candidate not in refs:
            refs.append(candidate)
    return ["SKILL.md", *dict.fromkeys(refs)], reason or "jev_route"


def _skill_root() -> Path | None:
    """Resolve the active skill root without hardcoding a Hermes home."""
    try:
        from hermes_constants import get_bundled_skills_dir, get_skills_dir

        bases = (get_skills_dir(), get_bundled_skills_dir())
    except Exception:
        return None
    for base in bases:
        candidate = Path(base) / "autonomous-ai-agents" / SKILL_NAME
        if candidate.is_dir():
            return candidate
    return None


def _normalise_skill_read(name: Any, file_path: Any) -> str | None:
    if not isinstance(name, str) or name.strip().casefold() not in {SKILL_NAME, "autonomous-ai-agents/hermes-agent"}:
        return None
    relative = _clean_text(file_path or "SKILL.md", 300)
    for prefix in ("skills/hermes-agent/", "autonomous-ai-agents/hermes-agent/"):
        if relative.startswith(prefix):
            relative = relative[len(prefix):]
    if relative in {"", ".", "SKILL.md"}:
        return "SKILL.md"
    if relative in ALL_REFERENCE_FILES or relative in {"templates/clock.mjs", "templates/plugin.js", "templates/skin.yaml"}:
        return relative
    return None


def _status_value(status: Any, error_type: Any) -> str:
    if isinstance(error_type, str) and error_type.strip():
        return "error"
    if isinstance(status, str) and status.strip():
        return status.strip().lower()[:32]
    return "ok"


def _parse_reference_response(value: Any) -> dict[str, Any]:
    """Validate the small typed response returned by TypeSafe/OpenRouter."""
    if not isinstance(value, Mapping):
        raise bridge.JevRequestError("malformed_response")
    answers = value.get("answers")
    if not isinstance(answers, Mapping):
        raise bridge.JevRequestError("malformed_response")
    answer = answers.get("reference_route") or answers.get("route")
    if not isinstance(answer, Mapping):
        raise bridge.JevRequestError("malformed_response")
    choice = answer.get("choice")
    probabilities = answer.get("probabilities")
    confidence = answer.get("confidence")
    if choice not in ROUTE_BUNDLES or not isinstance(probabilities, Mapping) or set(probabilities) != set(ROUTE_BUNDLES):
        raise bridge.JevRequestError("malformed_response")
    clean_probabilities: dict[str, float] = {}
    for label in ROUTE_BUNDLES:
        candidate = probabilities[label]
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or not 0 <= float(candidate) <= 1:
            raise bridge.JevRequestError("malformed_response")
        clean_probabilities[label] = float(candidate)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise bridge.JevRequestError("malformed_response")
    model = value.get("model", bridge.PRIMARY_PROVIDER.model)
    if not isinstance(model, str) or not model.strip():
        raise bridge.JevRequestError("malformed_response")
    return {
        "route": choice,
        "probabilities": clean_probabilities,
        "confidence": float(confidence),
        "model": _clean_text(model, 128),
        "usage": bridge._safe_usage(value.get("usage")),
    }


def _cost_from_usage(usage: Any) -> str:
    if not isinstance(usage, Mapping) or usage.get("cost") is None:
        return "not_available"
    value = usage.get("cost")
    return _clean_text(value, 64) if isinstance(value, (str, int, float)) else "not_available"


def _fixture_response(route: str) -> dict[str, Any]:
    return {
        "model": "jev-reading-fixture",
        "answers": {
            "reference_route": {
                "choice": route,
                "probabilities": {label: (1.0 if label == route else 0.0) for label in ROUTE_BUNDLES},
                "confidence": 0.91,
            }
        },
        "usage": None,
    }


def _request_with_fallback(
    request_body: Mapping[str, Any],
    *,
    primary: bridge.ProviderSpec,
    fallback: bridge.ProviderSpec,
    timeout: float,
    post_fn: Callable[[bridge.ProviderSpec, Mapping[str, Any], float], Any] | None = None,
) -> tuple[dict[str, Any], bridge.ProviderSpec, bool, int]:
    sender = post_fn or bridge._post_provider
    attempts = 0

    def call(spec: bridge.ProviderSpec) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        body = dict(request_body)
        body["model"] = spec.model
        return _parse_reference_response(sender(spec, body, timeout))

    try:
        return call(primary), primary, False, attempts
    except bridge.JevRequestError as exc:
        if not bridge._fallback_allowed(exc):
            raise
    return call(fallback), fallback, True, attempts


def _request_body(request: str, *, request_id: str, platform: str, model: str) -> dict[str, Any]:
    state = {
        "request_id": request_id,
        "request": request,
        "platform": _clean_text(platform, 64, allow_empty=True),
        "advisory_only": True,
        "selection_scope": "Choose the minimal hermes-agent reference bundle for the current request. Do not execute tools, mutate files, or change authority.",
        "hub": SKILL_NAME,
        "manifest": {label: list(paths) for label, paths in ROUTE_BUNDLES.items()},
        "hard_invariants": list(HARD_INVARIANTS),
        "expected_output": {
            "route": "one manifest label",
            "candidate_references": "derived locally from route",
            "required_reads": "SKILL.md plus candidates and local mandatory reads",
            "hard_invariants": "preserved locally",
            "confidence": "number in [0,1]",
            "reason_code": "local reason code",
            "request_id": request_id,
        },
    }
    encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_STATE_BYTES:
        raise bridge.BridgeValidationError("targeted-reading request exceeds bounded size")
    return {
        "model": _clean_text(model, 128),
        "state": encoded,
        "questions": {
            "reference_route": {
                "type": "choice",
                "instructions": "Choose exactly one advisory reference route label. The local loader remains authoritative.",
                "criteria": ROUTE_CRITERIA,
            }
        },
    }


class TargetedReadingAdvisor:
    """One advisory preflight plus read-observation logging for a profile."""

    def __init__(
        self,
        ctx: Any,
        store: bridge.BridgeStore,
        *,
        profile_name: str = "default",
        post_fn: Callable[[bridge.ProviderSpec, Mapping[str, Any], float], Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.ctx = ctx
        self.store = store
        self.profile_name = profile_name or "default"
        self.post_fn = post_fn
        self.clock = clock or time.monotonic
        self.reading_path = store.root / "skill-reading-advisory.jsonl"
        self.enabled = _as_bool(_config_value(ctx, "targeted_reading.enabled", True), True)
        self.once_per_session = _as_bool(_config_value(ctx, "targeted_reading.once_per_session", True), True)
        self.timeout = _as_float(_config_value(ctx, "targeted_reading.timeout_seconds", DEFAULT_TIMEOUT_SECONDS), DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS)
        self.max_request_chars = _as_int(_config_value(ctx, "targeted_reading.max_request_chars", MAX_REQUEST_CHARS), MAX_REQUEST_CHARS, 8_000)
        self.max_context_chars = _as_int(_config_value(ctx, "targeted_reading.max_context_chars", MAX_CONTEXT_CHARS), MAX_CONTEXT_CHARS, 8_000)
        self._lock = threading.RLock()
        self._seen_sessions: set[str] = set()
        self._latest: dict[str, dict[str, Any]] = {}
        self._latest_by_session: dict[str, dict[str, Any]] = {}
        self._actual_reads: dict[str, set[str]] = {}
        # ``pre_llm_call`` frames each user turn.  Capture the bounded request
        # there; later ``transform_tool_result`` calls use the exact turn key
        # to refresh advisory guidance only when hermes-agent is actually read.
        self._request_context: dict[str, dict[str, Any]] = {}
        self._active_turn_by_session: dict[str, str] = {}
        self._attempted_turns: set[str] = set()
        # ``post_tool_call`` runs before ``transform_tool_result``.  A later
        # turn has no advisory yet, so stage that read and bind it once the
        # transform callback obtains the current turn's advisory.
        self._pending_observations: dict[str, list[dict[str, Any]]] = {}

    def _session_key(self, session_id: Any, turn_id: Any) -> str:
        session = _clean_text(session_id, 300, allow_empty=True)
        if session:
            return "session:" + session
        turn = _clean_text(turn_id, 300, allow_empty=True)
        if turn:
            return "turn:" + turn
        return "anonymous:" + uuid.uuid4().hex

    def _turn_key(self, session_id: Any, turn_id: Any) -> str:
        session = _clean_text(session_id, 300, allow_empty=True)
        turn = _clean_text(turn_id, 300, allow_empty=True)
        if session and turn:
            return f"session:{session}:turn:{turn}"
        if turn:
            return "turn:" + turn
        if session:
            return "session:" + session
        return "anonymous:" + uuid.uuid4().hex

    def _capture_request(
        self,
        *,
        session_id: Any,
        task_id: Any,
        turn_id: Any,
        user_message: Any,
        platform: Any,
    ) -> tuple[str, str, str] | None:
        request = _user_text(user_message)
        if not request:
            return None
        request = request[: self.max_request_chars]
        session_key = self._session_key(session_id, turn_id)
        turn_key = self._turn_key(session_id, turn_id)
        with self._lock:
            previous_turn = self._active_turn_by_session.get(session_key)
            self._active_turn_by_session[session_key] = turn_key
            if previous_turn and previous_turn != turn_key:
                # A timed-out transform from an older turn must not later
                # attach its staged observation to the new turn.
                self._pending_observations.pop(previous_turn, None)
            self._request_context[turn_key] = {
                "session_key": session_key,
                "task_id": task_id,
                "turn_id": turn_id,
                "request": request,
                "platform": _clean_text(platform, 64, allow_empty=True),
            }
        return turn_key, session_key, request

    @staticmethod
    def _targeted_skill_view(tool_name: Any, args: Any) -> bool:
        if tool_name != "skill_view" or not isinstance(args, Mapping):
            return False
        name = args.get("name")
        return isinstance(name, str) and name.strip().casefold() in {
            SKILL_NAME,
            "autonomous-ai-agents/hermes-agent",
        }

    def _append(self, record: Mapping[str, Any]) -> None:
        try:
            with self.store.interprocess_lock():
                bridge._append_jsonl(self.reading_path, dict(record))
        except Exception:
            log.warning("Jev targeted-reading ledger append failed", exc_info=True)

    def _reserve_budget(self, request_id: str, session_hash: str) -> bool:
        try:
            with self.store.interprocess_lock():
                bridge._append_jsonl(
                    self.reading_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "skill_reading_reservation",
                        "request_id": request_id,
                        "session_hash": session_hash,
                        "day": _jst_day(),
                        "reserved_calls": MAX_PROVIDER_ATTEMPTS,
                        "created_at": _now(),
                    },
                )
                return True
        except Exception:
            log.warning("Jev targeted-reading budget read failed; skipping advisory", exc_info=True)
            return False

    def _result_record(
        self,
        *,
        request_id: str,
        session_key: str,
        task_id: Any,
        turn_id: Any,
        request: str,
        status: str,
        started: float,
        provider: str = "not_available",
        transport: str = "not_available",
        fallback_used: bool = False,
        provider_attempts: int = 0,
        decision: Mapping[str, Any] | None = None,
        reason_code: str = "not_available",
        error_reason: str | None = None,
    ) -> dict[str, Any]:
        decision = decision or {}
        route = decision.get("route")
        candidates = _files_for_route(route) if isinstance(route, str) else []
        required, mandatory_reason = _required_reads(route if isinstance(route, str) else "main_only", request)
        if mandatory_reason and reason_code == "jev_route":
            reason_code = mandatory_reason
        usage = decision.get("usage")
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "skill_reading_advisory",
            "request_id": request_id,
            "profile": self.profile_name,
            "session_hash": _hash_value(session_key),
            "task_hash": _hash_value(task_id),
            "turn_hash": _hash_value(turn_id),
            "status": status,
            "route": route or "main_only",
            "candidate_references": candidates,
            "required_reads": required,
            "hard_invariants": list(HARD_INVARIANTS),
            "confidence": float(decision.get("confidence", 0.0)) if isinstance(decision.get("confidence", 0.0), (int, float)) else 0.0,
            "reason_code": reason_code,
            "provider": provider,
            "transport": transport,
            "fallback_used": bool(fallback_used),
            "provider_attempts": int(provider_attempts),
            "usage": usage if isinstance(usage, Mapping) else None,
            "cost": _cost_from_usage(usage),
            "payload_bytes": len(request.encode("utf-8", "replace")),
            "latency_ms": round(max(0.0, self.clock() - started) * 1000, 2),
            "actual_reads_observed": False,
            "comparison_status": "pending" if status in {"advisory", "fallback"} else "not_observed",
            "request_text_logged": False,
            "error_reason": _clean_text(error_reason, 96, allow_empty=True) if error_reason else None,
            "created_at": _now(),
        }
        return record

    def _format_context(self, record: Mapping[str, Any]) -> str:
        def joined(key: str) -> str:
            values = record.get(key, [])
            return ", ".join(str(value) for value in values) or "(none)"

        text = (
            "[Jev targeted-reading advisory; advisory only; worker authority unchanged]\n"
            f"request_id={record['request_id']} route={record['route']} confidence={float(record['confidence']):.3f} reason_code={record['reason_code']}\n"
            f"candidate_references={joined('candidate_references')}\n"
            f"required_reads={joined('required_reads')}\n"
            f"hard_invariants={joined('hard_invariants')}\n"
            "Read the hermes-agent hub first, then use the candidates only as a starting point. "
            "The existing loader may read additional material; Jev does not authorize tools, configuration, permissions, or skill changes."
        )
        return text[: self.max_context_chars]

    def _advisory_field(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Return the bounded, additive JSON field delivered with skill_view."""
        return {
            "advisory_only": True,
            "request_id": record.get("request_id"),
            "route": record.get("route", "main_only"),
            "candidate_references": list(record.get("candidate_references", [])),
            "required_reads": list(record.get("required_reads", [])),
            "hard_invariants": list(record.get("hard_invariants", [])),
            "confidence": record.get("confidence", 0.0),
            "reason_code": record.get("reason_code", "not_available"),
            "status": record.get("status", "advisory"),
            "context": self._format_context(record),
        }

    def _run_advisory(
        self,
        *,
        turn_key: str,
        session_key: str,
        task_id: Any,
        turn_id: Any,
        request: str,
        platform: str,
        deliver: bool,
    ) -> dict[str, str] | None:
        with self._lock:
            if turn_key in self._attempted_turns:
                existing = self._latest.get(turn_key) if deliver else None
                context = self._format_context(existing) if existing else None
                return {"context": context} if context else None
            # Reserve before the provider call so parallel hub/reference calls
            # in the same turn coalesce without holding the lock during I/O.
            self._attempted_turns.add(turn_key)
        request_id = "jread-" + uuid.uuid4().hex[:20]
        session_hash = _hash_value(session_key)
        started = self.clock()
        if not self._reserve_budget(request_id, session_hash):
            record = self._result_record(
                request_id=request_id, session_key=session_key, task_id=task_id, turn_id=turn_id,
                request=request, status="skipped", started=started, reason_code="reservation_unavailable",
            )
            self._append(record)
            return None
        try:
            primary = bridge._configured_provider(self.ctx, "primary", bridge.PRIMARY_PROVIDER)
            fallback = bridge._configured_provider(self.ctx, "fallback", bridge.FALLBACK_PROVIDER)
            body = _request_body(request, request_id=request_id, platform=platform, model=primary.model)
            decision, provider, fallback_used, attempts = _request_with_fallback(
                body, primary=primary, fallback=fallback, timeout=self.timeout, post_fn=self.post_fn,
            )
            route = str(decision["route"])
            candidates = _files_for_route(route)
            required, mandatory_reason = _required_reads(route, request)
            reason_code = mandatory_reason or "jev_route"
            enriched = dict(decision)
            enriched.update({
                "route": route,
                "candidate_references": candidates,
                "required_reads": required,
                "hard_invariants": list(HARD_INVARIANTS),
                "reason_code": reason_code,
                "request_id": request_id,
            })
            record = self._result_record(
                request_id=request_id, session_key=session_key, task_id=task_id, turn_id=turn_id,
                request=request, status="fallback" if fallback_used else "advisory", started=started,
                provider=provider.name, transport=provider.transport, fallback_used=fallback_used,
                provider_attempts=attempts, decision=enriched, reason_code=reason_code,
            )
            context = self._format_context(record)
            with self._lock:
                current = self._active_turn_by_session.get(session_key) == turn_key
                if current:
                    self._latest[turn_key] = record
                    self._latest_by_session[session_key] = record
                    self._actual_reads.setdefault(turn_key, set())
            if not current:
                # The provider completed after a newer turn became active.
                # Keep the result out of all delivery/read-binding state.
                return None
            self._append(record)
            return {"context": context} if deliver else None
        except bridge.JevRequestError as exc:
            status = "malformed" if exc.reason == "malformed_response" else "unavailable"
            record = self._result_record(
                request_id=request_id, session_key=session_key, task_id=task_id, turn_id=turn_id,
                request=request, status=status, started=started, reason_code=exc.reason,
                error_reason=exc.reason,
            )
            self._append(record)
        except (bridge.BridgeValidationError, TypeError, ValueError, KeyError) as exc:
            record = self._result_record(
                request_id=request_id, session_key=session_key, task_id=task_id, turn_id=turn_id,
                request=request, status="malformed", started=started, reason_code="malformed_response",
                error_reason=type(exc).__name__,
            )
            self._append(record)
        except Exception as exc:
            # Advisory failure must never change the ordinary skill-reading path.
            record = self._result_record(
                request_id=request_id, session_key=session_key, task_id=task_id, turn_id=turn_id,
                request=request, status="skipped", started=started, reason_code="unexpected_advisory_error",
                error_reason=type(exc).__name__,
            )
            self._append(record)
            log.warning("Jev targeted-reading advisory skipped", exc_info=True)
        return None

    def pre_llm_call(
        self,
        *,
        session_id: str = "",
        task_id: str = "",
        turn_id: str = "",
        user_message: Any = None,
        is_first_turn: bool = False,
        platform: str = "",
        **_: Any,
    ) -> dict[str, str] | None:
        if not self.enabled:
            return None
        captured = self._capture_request(
            session_id=session_id, task_id=task_id, turn_id=turn_id,
            user_message=user_message, platform=platform,
        )
        if captured is None:
            return None
        turn_key, session_key, request = captured
        with self._lock:
            if self.once_per_session and not is_first_turn:
                return None
            if self.once_per_session and session_key in self._seen_sessions:
                return None
            if self.once_per_session:
                self._seen_sessions.add(session_key)
        return self._run_advisory(
            turn_key=turn_key, session_key=session_key, task_id=task_id,
            turn_id=turn_id, request=request, platform=platform, deliver=True,
        )

    def _append_read_observation(
        self,
        advisory: Mapping[str, Any],
        *,
        turn_key: str,
        relative: str,
        status: Any,
        error_type: Any,
    ) -> None:
        candidates = set(advisory.get("candidate_references", []))
        required = set(advisory.get("required_reads", []))
        with self._lock:
            seen = self._actual_reads.setdefault(turn_key, set())
            reread = relative in seen
            seen.add(relative)
        event = {
            "schema_version": SCHEMA_VERSION,
            "kind": "skill_read_observation",
            "request_id": advisory["request_id"],
            "profile": self.profile_name,
            "session_hash": advisory["session_hash"],
            "path": relative,
            "candidate_hit": relative in candidates,
            "required_hit": relative in required,
            "additional_read": relative not in candidates and relative != "SKILL.md",
            "re_read": reread,
            "status": _status_value(status, error_type),
            "content_logged": False,
            "created_at": _now(),
        }
        self._append(event)

    def _flush_pending_observations(self, turn_key: str, advisory: Mapping[str, Any]) -> None:
        with self._lock:
            pending = self._pending_observations.pop(turn_key, [])
        for item in pending:
            self._append_read_observation(
                advisory,
                turn_key=turn_key,
                relative=item["relative"],
                status=item.get("status"),
                error_type=item.get("error_type"),
            )

    def transform_tool_result(
        self,
        *,
        tool_name: str = "",
        args: Any = None,
        result: Any = None,
        session_id: str = "",
        task_id: str = "",
        turn_id: str = "",
        **_: Any,
    ) -> str | None:
        """Add the current Jev advisory to exact hermes-agent skill reads.

        Hermes invokes this after ``post_tool_call`` and before appending the
        result to model context.  A later turn refreshes its advisory here;
        the host's bounded hook runner keeps provider failure/timeout fail-open.
        """
        if not self.enabled or not self._targeted_skill_view(tool_name, args):
            return None
        if not isinstance(result, str) or (not session_id and not turn_id):
            return None
        session_key = self._session_key(session_id, turn_id)
        turn_key = self._turn_key(session_id, turn_id)
        with self._lock:
            context = self._request_context.get(turn_key)
            advisory = self._latest.get(turn_key)
            current = self._active_turn_by_session.get(session_key) == turn_key
        if context is None or not current:
            return None
        if advisory is None:
            self._run_advisory(
                turn_key=turn_key,
                session_key=session_key,
                task_id=context.get("task_id", task_id),
                turn_id=context.get("turn_id", turn_id),
                request=context.get("request", ""),
                platform=context.get("platform", ""),
                deliver=False,
            )
            with self._lock:
                advisory = self._latest.get(turn_key)
                current = self._active_turn_by_session.get(session_key) == turn_key
        if advisory is None or not current:
            return None

        # The post hook has already observed this read.  If this was the
        # first read of a later turn, it staged the event until this advisory
        # existed; flush it with the same request_id before returning.
        self._flush_pending_observations(turn_key, advisory)
        try:
            transformed = json.loads(result)
        except (TypeError, ValueError):
            return None
        if not isinstance(transformed, dict) or "jev_advisory" in transformed:
            return None
        transformed["jev_advisory"] = self._advisory_field(advisory)
        try:
            return json.dumps(transformed, ensure_ascii=False)
        except (TypeError, ValueError):
            return None

    def post_tool_call(
        self,
        *,
        tool_name: str = "",
        args: Any = None,
        session_id: str = "",
        turn_id: str = "",
        status: Any = None,
        error_type: Any = None,
        **_: Any,
    ) -> None:
        if tool_name != "skill_view" or not isinstance(args, Mapping):
            return
        relative = _normalise_skill_read(args.get("name"), args.get("file_path"))
        if relative is None:
            return
        session_key = self._session_key(session_id, turn_id)
        turn_key = self._turn_key(session_id, turn_id)
        explicit_turn = bool(_clean_text(turn_id, 300, allow_empty=True))
        explicit_identity = explicit_turn or bool(_clean_text(session_id, 300, allow_empty=True))
        with self._lock:
            advisory = self._latest.get(turn_key)
            # A missing turn id is the only case where the legacy session-level
            # association is safe.  Never let a late callback from one turn
            # annotate a different, explicitly identified turn.
            if advisory is None and not _clean_text(turn_id, 300, allow_empty=True):
                advisory = self._latest_by_session.get(session_key)
            if explicit_turn and self._active_turn_by_session.get(session_key) not in {None, turn_key}:
                return
            if advisory is None:
                if explicit_identity:
                    self._pending_observations.setdefault(turn_key, []).append({
                        "relative": relative,
                        "status": status,
                        "error_type": error_type,
                    })
                return
        self._append_read_observation(
            advisory,
            turn_key=turn_key,
            relative=relative,
            status=status,
            error_type=error_type,
        )


def make_fixture_response(route: str = "configuration_reference") -> dict[str, Any]:
    """Small deterministic fixture used by the focused shadow tests."""
    if route not in ROUTE_BUNDLES:
        raise ValueError(route)
    return _fixture_response(route)


__all__ = [
    "ALL_REFERENCE_FILES",
    "HARD_INVARIANTS",
    "ROUTE_BUNDLES",
    "TargetedReadingAdvisor",
    "_mandatory_references",
    "_parse_reference_response",
    "_request_body",
    "_request_with_fallback",
    "make_fixture_response",
]
