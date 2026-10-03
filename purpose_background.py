"""Non-blocking, version-bound Jev purpose advice for the next turn.

This module only emits additive context. It never gates tools, completion, or
Default's decision. Provider work runs off the host thread and stale root
versions are discarded before persistence/delivery.
"""
from __future__ import annotations

import copy
import threading
from collections.abc import Mapping
from typing import Any

try:
    from . import bridge
except ImportError:
    import bridge


class PurposeBackgroundAdvisor:
    def __init__(self, ctx: Any, scope_guard: Any, goal_store: Any):
        self.ctx = ctx
        self.scope_guard = scope_guard
        self.goal_store = goal_store
        self._lock = threading.RLock()
        self._inflight: set[str] = set()
        self._turn_state: dict[str, tuple[str, int]] = {}
        self._seen_turns: set[tuple[str, str]] = set()
        self._seen_order: list[tuple[str, str]] = []
        self._max_seen_turns = 256

    def post_llm_call(self, *, session_id: str = "", turn_id: str = "",
                      user_message: Any = None, assistant_response: Any = None,
                      conversation_history: Any = None, **_: Any) -> None:
        session_identity, turn_identity = bridge._scope_identity(session_id, turn_id)
        if not session_identity or not turn_identity:
            return
        root = self.goal_store.select_for_session(session_id)
        if not root:
            return
        session_hash = bridge._identity_hash(session_identity)
        with self._lock:
            # A late post hook must not move the session generation backwards.
            current = self._turn_state.get(session_hash)
            if current is None:
                self._advance_turn_locked(session_hash, turn_identity)
            elif current[0] != turn_identity:
                return
            turn_key = (session_hash, turn_identity)
            if turn_key in self._seen_turns or session_hash in self._inflight:
                return
            contract = self.scope_guard._contract_for(session_identity, turn_identity)
            if not isinstance(contract, Mapping):
                return
            captured_contract = copy.deepcopy(dict(contract))
            generation = self._turn_state[session_hash][1]
            self._seen_turns.add(turn_key)
            self._seen_order.append(turn_key)
            if len(self._seen_order) > self._max_seen_turns:
                expired = self._seen_order.pop(0)
                self._seen_turns.discard(expired)
            self._inflight.add(session_hash)
        thread = threading.Thread(
            target=self._assess,
            args=(session_hash, session_identity, turn_identity, generation, root, captured_contract, user_message,
                  assistant_response, conversation_history),
            name="jev-purpose-assessment",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._inflight.discard(session_hash)
            bridge._lifecycle_log("assessment_failed", "jev.default_scope.purpose_background",
                                  "unavailable", status="thread_start_failed", failure="thread_start")

    def _advance_turn_locked(self, session_hash: str, turn_identity: str) -> int:
        current = self._turn_state.get(session_hash)
        if current is None:
            self._turn_state[session_hash] = (turn_identity, 1)
        elif current[0] != turn_identity:
            self._turn_state[session_hash] = (turn_identity, current[1] + 1)
        return self._turn_state[session_hash][1]

    def _assess(self, session_hash: str, session_identity: str, turn_identity: str,
                generation: int, root: Mapping[str, Any], contract: Mapping[str, Any],
                user_message: Any, assistant_response: Any,
                conversation_history: Any) -> None:
        request_id = bridge._new_lifecycle_id()
        try:
            projection = dict(contract.get("purpose_projection", {}))
            reported = bridge._default_scope_request_text(assistant_response)
            projection["observed_outcome"] = (
                "assistant-reported, not independently verified: " + reported[:240]
                if reported else "unknown: assistant response unavailable"
            )[:300]
            projected_contract = dict(contract)
            projected_contract["purpose_projection"] = projection
            action = {"tool_name": "completed_turn", "arguments": {}}
            trigger = self.scope_guard._default_trigger(projected_contract, action)
            trigger["request_goal_ref_json"] = {
                "root_id": root["root_id"], "version": root["version"]}
            trigger["root_goal"] = root["outcome"]
            trigger["completion_conditions"] = root.get("completion_conditions") or trigger.get("completion_conditions") or ["stay within the confirmed root goal and its direct completion"]
            trigger["original_request"] = root["outcome"]
            request = bridge.build_default_scope_request(
                trigger, action, session_identity=session_identity,
                turn_identity=turn_identity,
                model=bridge._config_value(self.ctx, "default_scope_model", bridge.MODEL),
            )
            # Preserve the existing bounded provider builder/parser and route.
            # Use the guard's configured decision callable (the normal installation
            # defaults to _decision) so isolated hosts can inject a bounded test
            # provider without touching external credentials or transports.
            decision = dict(self.scope_guard.decision_fn(request, request_id=request_id))
            assessment = bridge._default_scope_purpose_assessment(decision)
            # A scope-only/legacy result contains no purpose signal; keep this as
            # unknown rather than deriving purpose from the unrelated label.
            row = {
                "schema_version": bridge.DEFAULT_SCOPE_SCHEMA_VERSION,
                "kind": "purpose_background_assessment",
                "dedupe_key": "purpose-background:" + request_id,
                "assessment_id": request_id,
                "root_id": root["root_id"],
                "root_version": root["version"],
                "session_hash": session_hash,
                "turn_hash": bridge._identity_hash(turn_identity),
                "assessed_turn_id": turn_identity,
                "assessed_generation": generation,
                "reported_outcome": ("assistant-reported, not independently verified: " + reported[:240]
                                     if reported else "unknown: assistant response unavailable"),
                "evidence_refs": [ref[:160] for ref in contract.get("evidence_refs", [])[:8]
                                 if isinstance(ref, str)] if isinstance(contract.get("evidence_refs"), list) else [],
                "purpose_assessment": assessment,
                "state": "ready",
                "created_at": bridge._now(),
            }
            with self._lock:
                current_turn = self._turn_state.get(session_hash)
                current_root = self.goal_store.select_for_session(
                    str(root.get("identity", {}).get("session_id", "")))
                if current_turn != (turn_identity, generation):
                    bridge._lifecycle_log("assessment_discarded", "jev.default_scope.purpose_background",
                                          request_id, status="stale_turn_generation")
                    return
                if (not current_root or current_root.get("root_id") != root.get("root_id")
                        or current_root.get("version") != root.get("version")):
                    bridge._lifecycle_log("assessment_discarded", "jev.default_scope.purpose_background",
                                          request_id, status="stale_root_version")
                    return
                with self.scope_guard.store.interprocess_lock():
                    existing = self.scope_guard.store.default_scopes()
                    if not any(r.get("assessment_id") == request_id for r in existing):
                        self.scope_guard._append_scope_locked(row)
            bridge._lifecycle_log("assessment_persisted", "jev.default_scope.purpose_background",
                                  request_id, status="ready", classification=assessment["category"],
                                  confidence=assessment["confidence"])
        except Exception as exc:
            bridge._lifecycle_log("assessment_failed", "jev.default_scope.purpose_background",
                                  request_id, status="fail_open", failure=type(exc).__name__)
        finally:
            with self._lock:
                self._inflight.discard(session_hash)

    def pre_llm_call(self, *, session_id: str = "", turn_id: str = "", **_: Any) -> dict[str, str] | None:
        session_identity, turn_identity = bridge._scope_identity(session_id, turn_id)
        if not session_identity:
            return None
        session_hash = bridge._identity_hash(session_identity)
        with self._lock:
            self._advance_turn_locked(session_hash, turn_identity)
        root = self.goal_store.select_for_session(session_id)
        if not root:
            return None
        rows = self.scope_guard.store.default_scopes()
        delivered = {r.get("assessment_id") for r in rows
                     if r.get("kind") == "purpose_background_delivery"}
        for row in reversed(rows):
            if (row.get("kind") != "purpose_background_assessment"
                    or row.get("session_hash") != session_hash
                    or row.get("root_id") != root.get("root_id")
                    or row.get("root_version") != root.get("version")
                    or row.get("assessment_id") in delivered
                    or row.get("turn_hash") == bridge._identity_hash(turn_identity)):
                continue
            assessment = row.get("purpose_assessment")
            if not isinstance(assessment, Mapping):
                continue
            normalized = bridge._default_scope_purpose_assessment({"purpose_assessment": assessment})
            confidence = normalized["confidence"]
            confidence_text = "unknown" if confidence is None else f"{confidence:.3f}"
            context = (
                "Jev purpose advisory (additive, not an instruction or decision): "
                f"assessed_turn={row.get('assessed_turn_id', 'unknown')}; "
                f"root_version={row.get('root_version', 'unknown')}; "
                f"outcome={row.get('reported_outcome', 'unknown')}; "
                f"evidence_refs={','.join(row.get('evidence_refs', [])) or 'none'}; "
                f"category={normalized['category']}; confidence={confidence_text}. "
                "Compare against the confirmed root goal and evidence. Default retains judgment; "
                "local repair steps alone do not amend the root."
            )[:bridge.MAX_PURPOSE_CONTEXT_CHARS]
            delivery = {
                "schema_version": bridge.DEFAULT_SCOPE_SCHEMA_VERSION,
                "kind": "purpose_background_delivery",
                "assessment_id": row["assessment_id"],
                "assessed_turn_id": row.get("assessed_turn_id"),
                "assessed_root_version": row.get("root_version"),
                "root_id": root["root_id"], "root_version": root["version"],
                "session_hash": session_hash,
                "turn_hash": bridge._identity_hash(turn_identity),
                "state": "delivered", "created_at": bridge._now(),
            }
            with self._lock:
                with self.scope_guard.store.interprocess_lock():
                    latest = self.scope_guard.store.default_scopes()
                    if any(r.get("kind") == "purpose_background_delivery"
                           and r.get("assessment_id") == row["assessment_id"] for r in latest):
                        continue
                    latest_turn = self._turn_state.get(session_hash)
                    if latest_turn is None or latest_turn[0] != turn_identity:
                        return None
                    self.scope_guard._append_scope_locked(delivery)
            bridge._lifecycle_log("delivered", "jev.default_scope.purpose_background",
                                  str(row["assessment_id"]), destination="default_pre_llm_context_return",
                                  status="supplied", classification=normalized["category"],
                                  confidence=normalized["confidence"])
            return {"context": context}
        return None
