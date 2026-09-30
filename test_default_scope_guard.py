from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "jev_route_screening_default_scope_guard_test"
_spec = importlib.util.spec_from_file_location(
    PACKAGE_NAME,
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("could not load the installed Jev plugin package")
_package = importlib.util.module_from_spec(_spec)
sys.modules[PACKAGE_NAME] = _package
_spec.loader.exec_module(_package)
bridge = sys.modules[f"{PACKAGE_NAME}.bridge"]


class FakeContext:
    profile_name = "default"

    def __init__(self, root: Path) -> None:
        self._config = {
            "bridge_dir": str(root),
            "role": "consumer",
            "consumer_profiles": ["default"],
            "producer_profiles": ["ops"],
            "default_scope.enabled": True,
        }

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)


def _trigger() -> dict[str, Any]:
    return {
        "schema_version": bridge.TRIGGER_SCHEMA_VERSION,
        "trigger_id": "trg-default-scope-001",
        "dedupe_key": "task-default:run-1:checkpoint-1:scope_drift",
        "task_id": "task-default",
        "run_id": "run-1",
        "checkpoint_id": "checkpoint-1",
        "source_profile": "ops",
        "original_request": "complete the bounded configuration readback",
        "completion_conditions": ["read the requested configuration", "record the exact readback"],
        "observed_actions": [{"action": "read_file", "result": "verified=true", "evidence_changed": True, "evidence_refs": ["workspace://readback"]}],
        "evidence_refs": ["kanban://task/task-default"],
        "worker_conclusion": {"status": "unmeasured", "reason": "worker checkpoint has no final conclusion"},
        "current_unknown": "whether the next default action remains within the bounded readback",
        "next_action": "read back the remaining completion evidence",
        "drift_category": "scope_drift",
        "reason": "next action may exceed the admitted scope",
        "route_label": "scope_drift",
        "state": "pending_review",
        "created_at": "2026-09-21T00:00:00Z",
        "default_decision": "pending_readback",
    }


def _readback(trigger: dict[str, Any], *, session_id: str, turn_id: str) -> dict[str, Any]:
    return {
        "scope": "default_scope",
        "session_id": session_id,
        "turn_id": turn_id,
        "task_id": trigger["task_id"],
        "run_id": trigger["run_id"],
        "original_request": trigger["original_request"],
        "completion_conditions": trigger["completion_conditions"],
        "evidence_refs": trigger["evidence_refs"],
        "next_action": trigger["next_action"],
        "worker_conclusion": {"status": "unmeasured", "reason": "default readback contains no worker conclusion"},
    }


class DefaultScopeGuardTests(unittest.TestCase):
    def _make(self, decision_fn):
        temporary = tempfile.TemporaryDirectory(prefix="jev-default-scope-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = bridge.BridgeStore(root)
        context = FakeContext(root)
        consumer = bridge.Consumer(context, store)
        guard = bridge.DefaultScopeGuard(context, store, decision_fn=decision_fn)
        guard.attach_consumer(consumer)
        consumer.scope_guard = guard
        trigger = _trigger()
        store.append_trigger(trigger)
        with store.interprocess_lock():
            consumer._append_control_locked(
                trigger,
                "provisional_stop",
                source="jev",
                reason="worker candidate requires default readback",
            )
        return store, consumer, guard, trigger

    def _make_default(self, decision_fn, *, tool_name: str, action_args: dict[str, Any], session_id: str, turn_id: str, user_message: str = "complete the bounded default readback"):
        temporary = tempfile.TemporaryDirectory(prefix="jev-default-request-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = bridge.BridgeStore(root)
        context = FakeContext(root)
        consumer = bridge.Consumer(context, store)
        guard = bridge.DefaultScopeGuard(context, store, decision_fn=decision_fn)
        guard.attach_consumer(consumer)
        consumer.scope_guard = guard
        guard.pre_llm_call(
            session_id=session_id,
            turn_id=turn_id,
            user_message=user_message,
            completion_conditions=["read only", "record the exact readback"],
        )
        session_identity, turn_identity = bridge._scope_identity(session_id, turn_id)
        contract = guard._contract_for(session_identity, turn_identity)
        self.assertIsNotNone(contract)
        action = bridge._default_scope_action(tool_name, action_args)
        self.assertIsNotNone(action)
        trigger = guard._default_trigger(contract or {}, action or {})
        return store, consumer, guard, trigger

    @staticmethod
    def _scope_drift(_request: dict[str, Any]) -> dict[str, Any]:
        return {
            "label": "scope_drift",
            "route_label": "scope_drift",
            "confidence": 0.94,
            "reason": "the proposed default action is outside the bounded readback",
        }

    def test_scope_drift_blocks_before_tool_and_matching_readback_releases(self) -> None:
        store, consumer, guard, trigger = self._make_default(
            self._scope_drift,
            tool_name="write_file",
            action_args={"path": "readback.txt", "content": "MUST_NOT_PERSIST"},
            session_id="default-session-1",
            turn_id="turn-1",
        )
        first = guard.pre_tool_call(
            "write_file",
            {"path": "readback.txt", "content": "MUST_NOT_PERSIST"},
            session_id="default-session-1",
            turn_id="turn-1",
        )
        self.assertEqual(first["action"], "block")
        self.assertIn(bridge.DEFAULT_SCOPE_BLOCK_MESSAGE, first["message"])
        self.assertIn("Jev判定: scope_drift", first["message"])
        self.assertIn("confidence: 0.940", first["message"])
        self.assertIn("masterへ示して確認する", first["message"])
        self.assertEqual(len([row for row in store.default_scopes() if row.get("state") == "candidate"]), 1)
        self.assertEqual(len([row for row in store.controls() if row.get("scope") == "default"]), 1)
        with store.interprocess_lock():
            consumer._append_control_locked(
                trigger,
                "provisional_stop",
                source="jev",
                reason="independent worker fixture",
                scope="worker",
            )

        duplicate = guard.pre_tool_call(
            "write_file",
            {"path": "readback.txt", "content": "DIFFERENT_UNLOGGED_CONTENT"},
            session_id="default-session-1",
            turn_id="turn-1",
        )
        self.assertEqual(duplicate, first)
        self.assertEqual(len([row for row in store.default_scopes() if row.get("state") == "candidate"]), 1)

        with self.assertRaises(bridge.BridgeValidationError):
            consumer.review_tool(
                args={
                    "trigger_id": trigger["trigger_id"],
                    "decision": "no_intervention",
                    "readback": _readback(trigger, session_id="different-session", turn_id="turn-1"),
                }
            )
        self.assertIsNotNone(bridge._active_scope_control(store, trigger, scope="default"))

        released = json.loads(
            consumer.review_tool(
                args={
                    "trigger_id": trigger["trigger_id"],
                    "decision": "no_intervention",
                    "readback": _readback(trigger, session_id="default-session-1", turn_id="turn-2"),
                }
            )
        )
        self.assertEqual(released["state"], "no_intervention")
        self.assertEqual(released["default_decision"], "no_intervention")
        self.assertIsNone(bridge._active_scope_control(store, trigger, scope="default"))
        self.assertIsNone(
            guard.pre_tool_call(
                "write_file",
                {"path": "readback.txt", "content": "allowed-after-release"},
                session_id="default-session-1",
                turn_id="turn-3",
            )
        )
        self.assertIsNotNone(bridge._active_scope_control(store, trigger, scope="worker"))

    def test_worker_trigger_stays_on_worker_review_route_without_default_identity_block(self) -> None:
        store, consumer, guard, trigger = self._make(self._scope_drift)
        calls: list[dict[str, Any]] = []

        def unexpected_default_screen(request: dict[str, Any]) -> dict[str, Any]:
            calls.append(request)
            return self._scope_drift(request)

        guard.decision_fn = unexpected_default_screen
        result = guard.pre_tool_call(
            "read_file",
            {"path": "worker-readback.txt"},
            session_id="different-default-session",
            turn_id="worker-turn-1",
        )
        self.assertIsNone(result)
        self.assertEqual(calls, [])
        self.assertEqual(store.default_scopes(), [])
        self.assertIsNone(bridge._active_scope_control(store, trigger, scope="default"))
        self.assertIsNotNone(bridge._active_scope_control(store, trigger, scope="worker"))

        pending = json.loads(consumer.review_tool(args={}, session_id="default-session"))
        self.assertEqual([row["trigger_id"] for row in pending], [trigger["trigger_id"]])
        worker_readback = _readback(trigger, session_id="default-session", turn_id="worker-review")
        worker_readback["scope"] = "worker"
        released = json.loads(
            consumer.review_tool(
                args={
                    "scope": "worker",
                    "trigger_id": trigger["trigger_id"],
                    "decision": "no_intervention",
                    "readback": worker_readback,
                },
                session_id="default-session",
            )
        )
        self.assertEqual(released["state"], "no_intervention")
        self.assertIsNone(bridge._active_scope_control(store, trigger, scope="worker"))
        self.assertIsNone(bridge._active_scope_control(store, trigger, scope="default"))

    def test_default_request_is_screened_without_worker_trigger(self) -> None:
        calls: list[dict[str, Any]] = []

        def scope_drift(request: dict[str, Any]) -> dict[str, Any]:
            calls.append(request)
            return {
                "label": "scope_drift",
                "route_label": "scope_drift",
                "confidence": 0.91,
                "reason": "the proposed default write is not part of the frozen request",
            }

        temporary = tempfile.TemporaryDirectory(prefix="jev-default-request-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = bridge.BridgeStore(root)
        context = FakeContext(root)
        consumer = bridge.Consumer(context, store)
        guard = bridge.DefaultScopeGuard(context, store, decision_fn=scope_drift)
        guard.attach_consumer(consumer)
        consumer.scope_guard = guard

        guard.pre_llm_call(
            session_id="default-session-2",
            turn_id="turn-2",
            user_message="read the existing configuration and report it",
            completion_conditions=["read only", "report the exact values"],
        )
        blocked = guard.pre_tool_call(
            "write_file",
            {"path": "must-not-write.txt", "content": "SIDE_EFFECT"},
            session_id="default-session-2",
            turn_id="turn-2",
        )
        self.assertEqual(blocked["action"], "block")
        self.assertIn(bridge.DEFAULT_SCOPE_BLOCK_MESSAGE, blocked["message"])
        self.assertIn("Jev判定: scope_drift", blocked["message"])
        self.assertIn("confidence: 0.910", blocked["message"])
        self.assertEqual(len(calls), 1)
        state = json.loads(calls[0]["state"])
        self.assertEqual(state["original_request"], "read the existing configuration and report it")
        self.assertEqual(state["completion_conditions"], ["read only", "report the exact values"])
        self.assertEqual(state["current_action"]["tool_name"], "write_file")
        self.assertNotIn("SIDE_EFFECT", calls[0]["state"])

        trigger = [row for row in store.triggers() if row.get("scope") == "default_scope"][-1]
        released = json.loads(
            consumer.review_tool(
                args={
                    "scope": "default_scope",
                    "trigger_id": trigger["trigger_id"],
                    "decision": "no_intervention",
                    "readback": _readback(trigger, session_id="default-session-2", turn_id="turn-2"),
                }
            )
        )
        self.assertEqual(released["state"], "no_intervention")
        self.assertIsNone(
            guard.pre_tool_call(
                "write_file",
                {"path": "must-not-write.txt", "content": "allowed-after-readback"},
                session_id="default-session-2",
                turn_id="turn-2",
            )
        )

    def test_default_no_candidate_and_failure_fail_open_without_worker_event(self) -> None:
        calls: list[dict[str, Any]] = []

        def insufficient(request: dict[str, Any]) -> dict[str, Any]:
            calls.append(request)
            return {"label": "insufficient_information", "confidence": 0.2, "reason": "not enough context"}

        temporary = tempfile.TemporaryDirectory(prefix="jev-default-open-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = bridge.BridgeStore(root)
        context = FakeContext(root)
        consumer = bridge.Consumer(context, store)
        guard = bridge.DefaultScopeGuard(context, store, decision_fn=insufficient)
        guard.attach_consumer(consumer)
        guard.pre_llm_call(session_id="s", turn_id="t", user_message="inspect one file")
        self.assertIsNone(guard.pre_tool_call("read_file", {"path": "one.txt"}, session_id="s", turn_id="t"))
        self.assertIsNone(guard.pre_tool_call("read_file", {"path": "one.txt"}, session_id="s", turn_id="t"))
        self.assertEqual(len(calls), 1)
        self.assertFalse(store.triggers())
        self.assertEqual(store.default_scopes()[-1]["state"], "advisory")

        def timeout(_request: dict[str, Any]) -> dict[str, Any]:
            raise bridge.JevRequestError("timeout")

        store2 = bridge.BridgeStore(root / "timeout")
        consumer2 = bridge.Consumer(context, store2)
        guard2 = bridge.DefaultScopeGuard(context, store2, decision_fn=timeout)
        guard2.attach_consumer(consumer2)
        guard2.pre_llm_call(session_id="s2", turn_id="t2", user_message="inspect another file")
        self.assertIsNone(guard2.pre_tool_call("read_file", {"path": "two.txt"}, session_id="s2", turn_id="t2"))
        self.assertEqual(store2.default_scopes()[-1]["state"], "fail_open")
        self.assertFalse(store2.triggers())

    def test_insufficient_information_and_failure_fail_open_without_stop(self) -> None:
        calls: list[dict[str, Any]] = []

        def insufficient(request: dict[str, Any]) -> dict[str, Any]:
            calls.append(request)
            return {"label": "insufficient_information", "confidence": 0.25, "reason": "bounded action context is incomplete"}

        store, _consumer, guard, trigger = self._make_default(
            insufficient,
            tool_name="read_file",
            action_args={"path": "config.yaml"},
            session_id="s",
            turn_id="t",
        )
        self.assertIsNone(guard.pre_tool_call("read_file", {"path": "config.yaml"}, session_id="s", turn_id="t"))
        self.assertIsNone(guard.pre_tool_call("read_file", {"path": "config.yaml"}, session_id="s", turn_id="t"))
        self.assertEqual(len(calls), 1, "same action/session must be deduplicated")
        self.assertIsNone(bridge._active_scope_control(store, trigger, scope="default"))
        self.assertEqual(store.default_scopes()[-1]["state"], "advisory")

        def timeout(_request: dict[str, Any]) -> dict[str, Any]:
            raise bridge.JevRequestError("timeout")

        store2, _consumer2, guard2, trigger2 = self._make_default(
            timeout,
            tool_name="read_file",
            action_args={"path": "other.txt"},
            session_id="s",
            turn_id="t",
        )
        self.assertIsNone(guard2.pre_tool_call("read_file", {"path": "other.txt"}, session_id="s", turn_id="t"))
        self.assertIsNone(bridge._active_scope_control(store2, trigger2, scope="default"))
        self.assertEqual(store2.default_scopes()[-1]["state"], "fail_open")

    def test_projection_contains_only_bounded_contract_and_action(self) -> None:
        trigger = _trigger()
        action = bridge._default_scope_action("patch", {"path": "x.txt", "content": "SECRET_SENTINEL"})
        self.assertIsNotNone(action)
        request = bridge.build_default_scope_request(trigger, action or {}, session_identity="session:s", turn_identity="turn:t")
        state = json.loads(request["state"])
        self.assertEqual(state["original_request"], trigger["original_request"])
        self.assertEqual(state["completion_conditions"], trigger["completion_conditions"])
        self.assertEqual(state["remaining_unknowns"], trigger["current_unknown"])
        self.assertEqual(state["expected_next_action"], trigger["next_action"])
        self.assertEqual(state["current_action"]["tool_name"], "patch")
        self.assertNotIn("SECRET_SENTINEL", request["state"])
        self.assertIn("scope_drift", request["questions"]["route"]["criteria"])

    def test_confidence_below_threshold_is_advisory_and_does_not_block(self) -> None:
        def low_confidence(_request: dict[str, Any]) -> dict[str, Any]:
            return {
                "label": "scope_drift",
                "route_label": "scope_drift",
                "confidence": 0.79,
                "reason": "the action may be outside the frozen request",
            }

        store, _consumer, guard, trigger = self._make_default(
            low_confidence,
            tool_name="write_file",
            action_args={"path": "readback.txt", "content": "bounded"},
            session_id="low-session",
            turn_id="low-turn",
        )
        result = guard.pre_tool_call(
            "write_file",
            {"path": "readback.txt", "content": "bounded"},
            session_id="low-session",
            turn_id="low-turn",
        )
        self.assertIsNone(result)
        self.assertIsNone(bridge._active_scope_control(store, trigger, scope="default"))
        outcome = store.default_scopes()[-1]
        self.assertEqual(outcome["state"], "advisory")
        self.assertEqual(outcome["label"], "scope_drift")
        self.assertEqual(outcome["confidence"], 0.79)
        self.assertFalse(outcome["confidence_gate"]["confirmation_required"])
        self.assertEqual(outcome["confidence_gate"]["next_action"], "readback_original_request_and_default_decision")

    def test_confidence_at_threshold_preserves_existing_block_and_readback_gate(self) -> None:
        def threshold_confidence(_request: dict[str, Any]) -> dict[str, Any]:
            return {
                "label": "scope_drift",
                "route_label": "scope_drift",
                "confidence": bridge.DEFAULT_SCOPE_CONFIDENCE_THRESHOLD,
                "reason": "the action is likely outside the frozen request",
            }

        store, _consumer, guard, trigger = self._make_default(
            threshold_confidence,
            tool_name="write_file",
            action_args={"path": "threshold.txt", "content": "bounded"},
            session_id="threshold-session",
            turn_id="threshold-turn",
        )
        result = guard.pre_tool_call(
            "write_file",
            {"path": "threshold.txt", "content": "bounded"},
            session_id="threshold-session",
            turn_id="threshold-turn",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "block")
        self.assertIn("confidence: 0.800", result["message"])
        self.assertIn("masterへ示して確認する", result["message"])
        self.assertIsNotNone(bridge._active_scope_control(store, trigger, scope="default"))
        self.assertEqual(store.default_scopes()[-1]["state"], "candidate")

    def test_invalid_confidence_preserves_fail_safe_without_default_stop(self) -> None:
        def invalid_confidence(_request: dict[str, Any]) -> dict[str, Any]:
            return {
                "label": "scope_drift",
                "route_label": "scope_drift",
                "confidence": "0.79",
                "reason": "the action may be outside the frozen request",
            }

        store, _consumer, guard, trigger = self._make_default(
            invalid_confidence,
            tool_name="write_file",
            action_args={"path": "invalid.txt", "content": "bounded"},
            session_id="invalid-session",
            turn_id="invalid-turn",
        )
        result = guard.pre_tool_call(
            "write_file",
            {"path": "invalid.txt", "content": "bounded"},
            session_id="invalid-session",
            turn_id="invalid-turn",
        )
        self.assertIsNone(result)
        self.assertIsNone(bridge._active_scope_control(store, trigger, scope="default"))
        outcome = store.default_scopes()[-1]
        self.assertEqual(outcome["state"], "fail_open")
        self.assertEqual(outcome["error_reason"], "malformed_response")

    def test_unknown_confidence_is_display_only_and_does_not_stop(self) -> None:
        def unknown_confidence(_request: dict[str, Any]) -> dict[str, Any]:
            return {
                "label": "scope_drift",
                "route_label": "scope_drift",
                "reason": "the action may be outside the frozen request",
            }

        store, _consumer, guard, trigger = self._make_default(
            unknown_confidence,
            tool_name="read_file",
            action_args={"path": "readback.txt"},
            session_id="unknown-session",
            turn_id="unknown-turn",
        )
        result = guard.pre_tool_call(
            "read_file",
            {"path": "readback.txt"},
            session_id="unknown-session",
            turn_id="unknown-turn",
        )
        self.assertIsNone(result)
        self.assertIsNone(bridge._active_scope_control(store, trigger, scope="default"))
        outcome = store.default_scopes()[-1]
        self.assertEqual(outcome["state"], "advisory")
        self.assertIsNone(outcome["confidence"])
        self.assertEqual(outcome["confidence_display"], bridge.DEFAULT_SCOPE_UNKNOWN_CONFIDENCE)
        self.assertFalse(outcome["confidence_gate"]["confirmation_required"])
        self.assertEqual(outcome["confidence_gate"]["next_action"], "display_unknown_only")

    def test_multiturn_purpose_projection_is_persisted_and_delivered_next_turn(self) -> None:
        calls: list[dict[str, Any]] = []

        def classify(request: dict[str, Any]) -> dict[str, Any]:
            calls.append(request)
            return {
                "label": "progressing",
                "confidence": 0.99,
                "reason": "route remains advisory",
                "purpose_assessment": {
                    "category": "procedure_success_but_outcome_missing",
                    "confidence": 0.91,
                },
            }

        store, _consumer, guard, _trigger = self._make_default(
            classify,
            tool_name="web_search",
            action_args={"query": "current news"},
            session_id="purpose-session",
            turn_id="initial-turn",
            user_message="Find useful, diverse current news across several areas.",
        )
        history = [
            {"role": "user", "content": "Find useful, diverse current news across several areas."},
            {"role": "assistant", "content": "I will search current publications and compare useful results."},
        ]
        for index in range(5):
            history.extend([
                {"role": "user", "content": f"Continue the broad news search, update {index}."},
                {"role": "assistant", "content": f"I will continue collecting useful current items, update {index}."},
            ])
        history.extend([
            {"role": "user", "content": "The local request failed; fix the error so I can continue the news search."},
            {"role": "tool", "name": "run", "status": "error", "content": "SECRET_TOOL_BODY_MUST_NOT_BE_COPIED"},
        ])
        guard.pre_llm_call(
            session_id="purpose-session",
            turn_id="repair-turn",
            user_message=history[-2]["content"],
            conversation_history=history,
            completion_conditions=["restrict retrieval to one media source", "use an exact 24-hour window"],
            current_unknown="whether these restrictions still yield useful coverage",
        )
        blocked = guard.pre_tool_call(
            "web_search",
            {"query": "current news"},
            session_id="purpose-session",
            turn_id="repair-turn",
        )
        self.assertIsNone(blocked, "purpose classification is advisory and must not add a stop")
        request_state = json.loads(calls[0]["state"])
        purpose_context = request_state["purpose_context"]
        self.assertIn("Find useful, diverse current news", purpose_context["observed_first_user_goal_not_assumed_active"])
        self.assertIn("fix the error", purpose_context["current_user_instruction"])
        self.assertEqual(purpose_context["recent_user_attributed_history"][0]["role"], "user")
        self.assertEqual(purpose_context["recent_user_attributed_history"][-1]["role"], "tool_observation")
        self.assertNotIn("SECRET_TOOL_BODY_MUST_NOT_BE_COPIED", calls[0]["state"])
        self.assertIn("authorship/necessity not established", purpose_context["default_derived_requirement_hypotheses"][0]["source"])
        self.assertIn("explicit user goal change or cancellation", purpose_context["instructions"])

        assessment_row = next(row for row in reversed(store.default_scopes()) if row.get("purpose_assessment"))
        self.assertEqual(assessment_row["purpose_assessment"]["category"], "procedure_success_but_outcome_missing")
        self.assertEqual(assessment_row["purpose_assessment"]["confidence"], 0.91)
        self.assertEqual(assessment_row["purpose_diagnostics"]["provider"], "returned")
        self.assertFalse(store.controls(), "purpose confidence/category must not create a control")

        delivered = guard.pre_llm_call(
            session_id="purpose-session",
            turn_id="next-turn",
            user_message="Continue searching for the original news outcome.",
            conversation_history=history,
        )
        self.assertIsInstance(delivered, dict)
        self.assertIn("procedure_success_but_outcome_missing", delivered["context"])
        self.assertIn("confidence=0.910", delivered["context"])
        self.assertIn("Default retains final judgment", delivered["context"])
        delivery_rows = [row for row in store.default_scopes() if row.get("kind") == "default_scope_context_delivery"]
        self.assertEqual(len(delivery_rows), 1)
        self.assertEqual(delivery_rows[0]["state"], "purpose_advisory_delivered")
        self.assertIsNone(guard.pre_llm_call(session_id="purpose-session", turn_id="repeat-turn", user_message="Continue."))
        self.assertEqual(len([row for row in store.default_scopes() if row.get("kind") == "default_scope_context_delivery"]), 1)

    def test_purpose_parser_is_typed_and_malformed_classification_fails_open(self) -> None:
        route_probs = {label: (1.0 if label == "progressing" else 0.0) for label in bridge.LABELS}
        purpose_probs = {label: (1.0 if label == "purpose_critical_prerequisite" else 0.0) for label in bridge.PURPOSE_LABELS}
        parsed = bridge.parse_jev_response({
            "answers": {
                "route": {"choice": "progressing", "probabilities": route_probs, "confidence": 0.9},
                "purpose": {"choice": "purpose_critical_prerequisite", "probabilities": purpose_probs, "confidence": 0.87},
            },
            "model": "local-stub",
        })
        self.assertEqual(parsed["purpose_assessment"]["category"], "purpose_critical_prerequisite")
        self.assertEqual(parsed["purpose_assessment"]["confidence"], 0.87)

        def malformed(_request: dict[str, Any]) -> dict[str, Any]:
            return {"label": "progressing", "confidence": 0.9, "purpose_assessment": {"category": "invented", "confidence": 0.99}}

        store, _consumer, guard, trigger = self._make_default(
            malformed,
            tool_name="read_file",
            action_args={"path": "fixture.txt"},
            session_id="malformed-purpose-session",
            turn_id="malformed-purpose-turn",
        )
        self.assertIsNone(guard.pre_tool_call("read_file", {"path": "fixture.txt"}, session_id="malformed-purpose-session", turn_id="malformed-purpose-turn"))
        self.assertEqual(store.default_scopes()[-1]["state"], "fail_open")
        self.assertIsNone(bridge._active_scope_control(store, trigger, scope="default"))

    def test_explicit_goal_change_is_kept_distinct_from_prior_goal(self) -> None:
        calls: list[dict[str, Any]] = []

        def classify(request: dict[str, Any]) -> dict[str, Any]:
            calls.append(request)
            return {"label": "progressing", "confidence": 0.9}

        store, _consumer, guard, _trigger = self._make_default(
            classify,
            tool_name="read_file",
            action_args={"path": "goal.txt"},
            session_id="goal-change-session",
            turn_id="goal-1",
            user_message="Find and summarize useful current news.",
        )
        history = [
            {"role": "user", "content": "Find and summarize useful current news."},
            {"role": "assistant", "content": "I will gather current items."},
            {"role": "user", "content": "Cancel the news task. New goal: report the current configuration only."},
        ]
        guard.pre_llm_call(
            session_id="goal-change-session",
            turn_id="goal-2",
            user_message=history[-1]["content"],
            conversation_history=history,
        )
        guard.pre_tool_call("read_file", {"path": "config.yaml"}, session_id="goal-change-session", turn_id="goal-2")
        state = json.loads(calls[0]["state"])
        purpose = state["purpose_context"]
        self.assertIn("Find and summarize useful current news", purpose["observed_first_user_goal_not_assumed_active"])
        self.assertIn("Cancel the news task", purpose["current_user_instruction"])
        self.assertEqual(purpose["recent_user_attributed_history"][-1]["role"], "user")
        row = next(row for row in reversed(store.default_scopes()) if row.get("purpose_assessment"))
        self.assertEqual(row["purpose_assessment"]["category"], "unknown")
        self.assertIsNone(row["purpose_assessment"]["confidence"])


class PurposeWireRegressionTests(unittest.TestCase):
    def test_all_purpose_categories_preserve_typed_advisory(self) -> None:
        for category in bridge.PURPOSE_LABELS:
            with self.subTest(category=category):
                response = {
                    "answers": {
                        "route": {"choice": "progressing", "probabilities": {label: float(label == "progressing") for label in bridge.LABELS}, "confidence": 0.9},
                        "purpose": {"choice": category, "probabilities": {label: float(label == category) for label in bridge.PURPOSE_LABELS}, "confidence": 0.9},
                    },
                    "model": "offline-fixture",
                }
                self.assertEqual(bridge.parse_jev_response(response)["purpose_assessment"]["category"], category)
                for bad in [True, "1.0", float("nan"), float("inf")]:
                    response["answers"]["purpose"]["probabilities"][category] = bad
                    with self.assertRaises(bridge.JevRequestError):
                        bridge.parse_jev_response(response)

    def test_full_history_goal_precedes_bounded_local_repairs(self) -> None:
        goal = "Find diverse useful news across regions."
        history = [{"role": "user", "content": goal}] + [{"role": "user", "content": "Fix this local error."} for _ in range(20)]
        self.assertEqual(bridge._default_scope_first_user_message(history, "Continue"), goal)
        self.assertLessEqual(len(bridge._default_scope_history(history, "Continue")), bridge.MAX_PURPOSE_HISTORY_MESSAGES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
