from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from .point_state import POINT_STATES


ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "jev_route_screening_point_state_candidate"


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)
    return package


def _event(bridge, *, task_id="task-point", run_id="run-point", checkpoint_id="checkpoint-1"):
    raw = {
        "schema_version": 2,
        "projection_version": 1,
        "source": "worker_hook",
        "category": "worker_checkpoint",
        "profile": "ops",
        "task_id": task_id,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "criteria": ["read the named evidence", "record the exact readback"],
        "observation": "bounded readback present",
        "recent_steps": [{
            "action": "read_file",
            "result": "verified=true",
            "evidence_changed": True,
            "evidence_refs": ["workspace://readback"],
        }],
        "evidence_refs": ["workspace://readback"],
        "completed_tools": 3,
        "tools_since_previous": 3,
        "changed_evidence": True,
        "changed_evidence_age_seconds": 120.0,
        "original_request": "verify the named evidence",
        "current_unknown": "whether the exact readback is complete",
        "next_action": "read the named verification",
        "evidence_fingerprint": f"fingerprint-{checkpoint_id}",
        "sensitive": False,
        "projection_safe": True,
    }
    event = bridge.validate_checkpoint(raw)
    event["state"] = "pending"
    return event


class ContextFactory:
    def __init__(self, plugin, bridge, PluginContext, PluginManifest, PluginManager, root: Path, *, point_state_enabled: bool, profile_name: str = "default", role: str = "consumer", manager_scope: Path | None = None):
        self.plugin = plugin
        self.bridge = bridge
        self.PluginContext = PluginContext
        self.PluginManifest = PluginManifest
        self.PluginManager = PluginManager
        self.root = root
        self.point_state_enabled = point_state_enabled
        self.profile_name_value = profile_name
        self.role = role
        self.manager_scope = manager_scope

    def make(self):
        manager = self.PluginManager(scope_key=str(self.manager_scope or self.root))
        bridge = self.bridge
        root = self.root
        point_state_enabled = self.point_state_enabled
        profile_name = self.profile_name_value
        role = self.role

        class Context(self.PluginContext):
            @property
            def profile_name(self):
                return profile_name

            def get_config(self, key, default=None):
                return {
                    "bridge_dir": str(root / "bridge"),
                    "role": role,
                    "consumer_profiles": ["default"],
                    "producer_profiles": ["ops"],
                    "default_scope.enabled": True,
                    "point_state_enabled": point_state_enabled,
                    "targeted_reading.enabled": False,
                }.get(key, default)

        context = Context(
            self.PluginManifest(name="jev-route-screening", source="local", path=str(ROOT)),
            manager,
        )
        bridge.register(context)
        return manager, context


def _consumer_and_guard(manager, bridge):
    consumer = next(
        hook.__self__
        for hook in manager._hooks["post_tool_call"]
        if isinstance(getattr(hook, "__self__", None), bridge.Consumer)
    )
    guard = next(
        hook.__self__
        for hook in manager._hooks["pre_tool_call"]
        if isinstance(getattr(hook, "__self__", None), bridge.DefaultScopeGuard)
    )
    return consumer, guard


def _point_sender(calls, choice="ordinary_action_ready", confidence=0.8):
    def sender(provider, body, timeout):
        del timeout
        calls.append({"provider": provider.name, "body": body})
        return {
            "model": provider.model,
            "answers": {"next_action_state": {"choice": choice, "confidence": confidence}},
        }

    return sender


def _legacy_sender(calls, label="scope_drift", confidence=0.9):
    def sender(provider, body, timeout):
        del timeout
        calls.append({"provider": provider.name, "body": body})
        labels = tuple(body["questions"]["route"]["criteria"])
        return {
            "model": provider.model,
            "answers": {
                "route": {
                    "choice": label,
                    "probabilities": {name: (1.0 if name == label else 0.0) for name in labels},
                    "confidence": confidence,
                    "reason": "legacy default scope fixture",
                }
            },
        }

    return sender


def test_real_plugin_manager_point_state_route_and_legacy_default_branch():
    plugin = _load_plugin()
    bridge = sys.modules[f"{PACKAGE_NAME}.bridge"]
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    with tempfile.TemporaryDirectory(prefix="jev-point-state-registry-") as temporary:
        root = Path(temporary)
        manager, context = ContextFactory(plugin, bridge, PluginContext, PluginManifest, PluginManager, root, point_state_enabled=True).make()
        consumer, guard = _consumer_and_guard(manager, bridge)
        assert consumer.point_state_enabled is True
        # DefaultScopeGuard keeps the established legacy scope classifier.
        assert guard.point_state_enabled is False

        event = _event(bridge)
        consumer.store.append_event(event)
        calls = []
        with patch.object(bridge, "_post_provider", side_effect=_point_sender(calls)):
            assert consumer.process_pending() is True

        assert len(calls) == 1
        body = calls[0]["body"]
        assert set(body) == {"model", "state", "questions"}
        assert body["model"]
        assert "point_state_enabled" not in body
        assert isinstance(body["state"], str)
        assert isinstance(body["questions"], dict)
        question = body["questions"]["next_action_state"]
        assert set(question) == {"type", "instructions", "criteria"}
        assert question["type"] == "choice"
        assert isinstance(question["criteria"], dict)
        assert set(question["criteria"]) == set(POINT_STATES)
        assert "acceptance_ready" in question["instructions"]
        review = consumer.store.reviews()[-1]
        assert review["point_state"] == "ordinary_action_ready"
        assert review["point_state_action"] == "continue_scoped_action"
        assert review["state"] == "completed"
        assert not consumer.store.triggers()

        # The same plugin still exposes the established default readback/release seam.
        guard.pre_llm_call(
            session_id="default-session",
            turn_id="turn-1",
            user_message="read the named evidence and report it",
            completion_conditions=["read only", "report exact values"],
        )
        with patch.object(bridge, "_post_provider", side_effect=_legacy_sender(calls, "scope_drift", 0.9)):
            blocked = guard.pre_tool_call(
                "write_file",
                {"path": "must-not-write.txt", "content": "SIDE_EFFECT"},
                session_id="default-session",
                turn_id="turn-1",
            )
        assert blocked is not None and blocked["action"] == "block"
        candidate = guard.store.default_scopes()[-1]
        assert candidate["state"] == "candidate"
        assert candidate["label"] == "scope_drift"
        trigger = [row for row in guard.store.triggers() if row.get("scope") == "default_scope"][-1]
        readback = {
            "scope": "default_scope",
            "session_id": "default-session",
            "turn_id": "turn-2",
            "task_id": trigger["task_id"],
            "run_id": trigger["run_id"],
            "original_request": trigger["original_request"],
            "completion_conditions": trigger["completion_conditions"],
            "evidence_refs": trigger["evidence_refs"],
            "next_action": trigger["next_action"],
            "worker_conclusion": {"status": "unmeasured", "reason": "default readback has no worker conclusion"},
        }
        released = json.loads(consumer.review_tool(
            args={
                "scope": "default_scope",
                "trigger_id": trigger["trigger_id"],
                "decision": "no_intervention",
                "readback": readback,
            },
            session_id="default-session",
        ))
        assert released["state"] == "no_intervention"
        assert released["default_decision"] == "no_intervention"
        assert bridge._active_scope_control(guard.store, trigger, scope="default") is None
        guard.decision_fn = lambda _request: {"label": "insufficient_information", "confidence": 0.5, "reason": "released legacy advisory"}
        assert guard.pre_tool_call(
            "read_file",
            {"path": "allowed-after-release.txt"},
            session_id="default-session",
            turn_id="turn-3",
        ) is None

        # Omit the new flag: the existing legacy consumer branch remains selected.
        legacy_root = root / "legacy"
        legacy_manager, _ = ContextFactory(plugin, bridge, PluginContext, PluginManifest, PluginManager, legacy_root, point_state_enabled=False).make()
        legacy_consumer, legacy_guard = _consumer_and_guard(legacy_manager, bridge)
        assert legacy_consumer.point_state_enabled is False
        assert legacy_guard.point_state_enabled is False


def test_registered_noncontinue_effect_holds_worker_and_budget_survives_checkpoint_release():
    """The opt-in route is real registry code; controls remain guidance-only and bounded."""
    plugin = _load_plugin()
    bridge = sys.modules[f"{PACKAGE_NAME}.bridge"]
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    with tempfile.TemporaryDirectory(prefix="jev-point-state-hold-registry-") as temporary:
        root = Path(temporary)
        default_manager, _ = ContextFactory(
            plugin,
            bridge,
            PluginContext,
            PluginManifest,
            PluginManager,
            root,
            point_state_enabled=True,
            profile_name="default",
            role="consumer",
            manager_scope=root / "default-manager",
        ).make()
        consumer, guard = _consumer_and_guard(default_manager, bridge)
        producer_manager, _ = ContextFactory(
            plugin,
            bridge,
            PluginContext,
            PluginManifest,
            PluginManager,
            root,
            point_state_enabled=True,
            profile_name="ops",
            role="producer",
            manager_scope=root / "ops-manager",
        ).make()
        producer = next(
            hook.__self__
            for hook in producer_manager._hooks["pre_tool_call"]
            if isinstance(getattr(hook, "__self__", None), bridge.Producer)
        )
        assert consumer.point_state_enabled is True
        assert guard.point_state_enabled is False

        first_event = _event(bridge, task_id="task-hold", run_id="run-hold", checkpoint_id="checkpoint-1")
        consumer.store.append_event(first_event)
        calls = []
        with patch.object(bridge, "_post_provider", side_effect=_point_sender(calls, "input_error_unresolved", 0.8)):
            assert consumer.process_pending() is True

        first_trigger = consumer.store.triggers()[-1]
        assert first_trigger["drift_category"] == "evidence_mismatch"
        assert first_trigger["point_state_action"] == "correct_input"
        assert first_trigger["point_state_reason"] == "first_unrepaired_error"
        assert first_trigger["point_state_instructions"] == ["read the named verification"]
        assert first_trigger["point_state_guidance"]["instructions"] == ["read the named verification"]
        assert first_trigger["point_state_milestone"] == "current-task"
        assert bridge._active_provisional_stop(consumer.store, "task-hold", "run-hold") is not None

        with patch.dict(
            os.environ,
            {"HERMES_PROFILE": "ops", "HERMES_KANBAN_TASK": "task-hold", "HERMES_KANBAN_RUN_ID": "run-hold"},
            clear=False,
        ):
            blocked = producer.pre_tool_call("write_file", {"path": "must-not-run"})
        assert blocked is not None and blocked["action"] == "block"
        assert bridge.PROVISIONAL_STOP_BLOCK_MESSAGE in blocked["message"]

        readback = {
            "scope": "worker",
            "task_id": first_trigger["task_id"],
            "run_id": first_trigger["run_id"],
            "original_request": first_trigger["original_request"],
            "completion_conditions": first_trigger["completion_conditions"],
            "evidence_refs": first_trigger["evidence_refs"],
            "next_action": first_trigger["next_action"],
            "worker_conclusion": {"status": "unmeasured", "reason": "worker test has no final conclusion"},
        }
        released = json.loads(
            consumer.review_tool(
                args={
                    "scope": "worker",
                    "trigger_id": first_trigger["trigger_id"],
                    "decision": "no_intervention",
                    "readback": readback,
                }
            )
        )
        assert released["state"] == "no_intervention"
        assert bridge._active_provisional_stop(consumer.store, "task-hold", "run-hold") is None
        with patch.dict(
            os.environ,
            {"HERMES_PROFILE": "ops", "HERMES_KANBAN_TASK": "task-hold", "HERMES_KANBAN_RUN_ID": "run-hold"},
            clear=False,
        ):
            assert producer.pre_tool_call("read_file", {"path": "after-release"}) is None

        # A new event id/checkpoint is allowed, but the stable current-task
        # correction budget is not reset by that new delivery event.
        second_event = _event(bridge, task_id="task-hold", run_id="run-hold", checkpoint_id="checkpoint-2")
        second_event["observation"] = "bounded readback changed after the first correction"
        second_event["recent_steps"][0]["result"] = "verified=true; changed"
        consumer.store.append_event(second_event)
        with patch.object(bridge, "_post_provider", side_effect=_point_sender(calls, "input_error_unresolved", 0.8)):
            assert consumer.process_pending() is True
        second_trigger = consumer.store.triggers()[-1]
        assert second_trigger["checkpoint_id"] == "checkpoint-2"
        assert second_trigger["point_state_milestone"] == "current-task"
        assert second_trigger["point_state_action"] == "review"
        assert second_trigger["point_state_reason"] == "correction_budget_exhausted"
        assert second_trigger["point_state_instructions"] == []
        assert len(calls) == 2


def test_registered_unknown_state_is_visible_review_guidance():
    plugin = _load_plugin()
    bridge = sys.modules[f"{PACKAGE_NAME}.bridge"]
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    with tempfile.TemporaryDirectory(prefix="jev-point-state-unknown-registry-") as temporary:
        root = Path(temporary)
        manager, _ = ContextFactory(
            plugin,
            bridge,
            PluginContext,
            PluginManifest,
            PluginManager,
            root,
            point_state_enabled=True,
            profile_name="default",
            role="consumer",
        ).make()
        consumer, guard = _consumer_and_guard(manager, bridge)
        assert guard.point_state_enabled is False
        consumer.store.append_event(_event(bridge, task_id="task-unknown", run_id="run-unknown", checkpoint_id="unknown-1"))
        with patch.object(bridge, "_post_provider", side_effect=_point_sender([], "unknown", 0.9)):
            assert consumer.process_pending() is True
        review = consumer.store.reviews()[-1]
        trigger = consumer.store.triggers()[-1]
        assert review["point_state_action"] == "refresh_evidence_then_review"
        assert review["state"] == "triggered"
        assert trigger["point_state"] == "unknown"
        assert trigger["point_state_reason"] == "unknown_state"
        assert trigger["point_state_instructions"] == [
            "Perform one bounded refresh of the named evidence, then return to default review; do not execute an arbitrary command."
        ]


def test_provider_exception_is_failed_not_false_pass():
    plugin = _load_plugin()
    bridge = sys.modules[f"{PACKAGE_NAME}.bridge"]
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    with tempfile.TemporaryDirectory(prefix="jev-point-state-provider-error-") as temporary:
        root = Path(temporary)
        manager, _ = ContextFactory(plugin, bridge, PluginContext, PluginManifest, PluginManager, root, point_state_enabled=True).make()
        consumer, _ = _consumer_and_guard(manager, bridge)
        consumer.store.append_event(_event(bridge, task_id="task-error", run_id="run-error", checkpoint_id="cp-error"))

        def provider_error(_event):
            raise bridge.JevRequestError("timeout")

        consumer.decision_fn = provider_error
        assert consumer.process_pending() is True
        review = consumer.store.reviews()[-1]
        assert review["state"] == "failed"
        assert review.get("reason") == "timeout"
        assert review["state"] != "completed"
        assert not consumer.store.triggers()
