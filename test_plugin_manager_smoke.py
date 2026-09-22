from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "jev_route_screening_plugin_manager_smoke_isolated"


def _load_plugin() -> Any:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the installed Jev plugin package")
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)
    return package


def fixture_sender(provider: Any, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    del provider, timeout
    route_labels = tuple(body["questions"]["reference_route"]["criteria"])
    route = "configuration_reference"
    return {
        "model": "jev-reading-plugin-isolated-smoke",
        "answers": {
            "reference_route": {
                "choice": route,
                "probabilities": {label: (1.0 if label == route else 0.0) for label in route_labels},
                "confidence": 0.9,
            }
        },
        "usage": None,
    }


class IsolatedContext:
    """Small host context that keeps plugin state in a temporary store only."""

    def __init__(self, profile_name: str, bridge_dir: Path, role: str) -> None:
        self.profile_name = profile_name
        self._config = {
            "bridge_dir": str(bridge_dir),
            "role": role,
            "producer_profiles": ["ops"],
            "consumer_profiles": ["default"],
            "targeted_reading.enabled": True,
            "targeted_reading.once_per_session": True,
            "targeted_reading.timeout_seconds": 5,
            "targeted_reading.max_request_chars": 4000,
            "targeted_reading.max_context_chars": 6000,
        }
        self.tools: dict[str, Any] = {}
        self.hooks: dict[str, list[Any]] = {}

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def register_tool(self, **kwargs: Any) -> None:
        self.tools[kwargs["name"]] = kwargs["handler"]

    def register_hook(self, name: str, callback: Any) -> None:
        self.hooks.setdefault(name, []).append(callback)


def _owners(context: IsolatedContext, class_name: str, hook_name: str) -> list[Any]:
    return [
        callback
        for callback in context.hooks.get(hook_name, [])
        if getattr(getattr(callback, "__self__", None), "__class__", type(None)).__name__ == class_name
    ]


def main() -> None:
    plugin = _load_plugin()
    with tempfile.TemporaryDirectory(prefix="jev-plugin-manager-smoke-") as temporary:
        root = Path(temporary)
        default_context = IsolatedContext("default", root / "default", "consumer")
        plugin.register(default_context)

        consumers = _owners(default_context, "Consumer", "post_tool_call")
        scope_guards = _owners(default_context, "DefaultScopeGuard", "pre_tool_call")
        scope_guard_llm_hooks = _owners(default_context, "DefaultScopeGuard", "pre_llm_call")
        advisors = _owners(default_context, "TargetedReadingAdvisor", "pre_llm_call")
        if len(consumers) != 1 or len(scope_guards) != 1 or len(scope_guard_llm_hooks) != 1 or len(advisors) != 1:
            raise SystemExit(
                f"isolated default registration missing consumer/scope/advisor hook: consumers={len(consumers)} scope_guards={len(scope_guards)} scope_guard_llm_hooks={len(scope_guard_llm_hooks)} advisors={len(advisors)}"
            )
        if "jev_bridge_review" not in default_context.tools or "jev_bridge_checkpoint" in default_context.tools:
            raise SystemExit("isolated default registration exposed the wrong bridge tool")

        advisor = advisors[0].__self__
        advisor.post_fn = fixture_sender
        result = advisor.pre_llm_call(
            session_id="isolated-plugin-manager-smoke-session",
            task_id="isolated-plugin-manager-smoke-task",
            turn_id="isolated-plugin-manager-smoke-turn",
            user_message="show the current configuration reference",
            is_first_turn=True,
            model="smoke-model",
            platform="local",
        )
        if not result or "configuration_reference" not in result.get("context", ""):
            raise SystemExit(f"isolated advisor invocation returned no targeted-reading context: {result!r}")

        ledger = Path(advisor.reading_path)
        if not ledger.is_file() or root not in ledger.resolve().parents:
            raise SystemExit(f"advisory ledger escaped the temporary smoke store: {ledger}")
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        if not any(row.get("kind") == "skill_reading_advisory" and row.get("status") == "advisory" for row in rows):
            raise SystemExit("isolated default handler did not persist an advisory ledger row")

        ops_context = IsolatedContext("ops", root / "ops", "producer")
        plugin.register(ops_context)
        ops_advisors = _owners(ops_context, "TargetedReadingAdvisor", "pre_llm_call")
        if ops_advisors:
            raise SystemExit("ops producer unexpectedly registered TargetedReadingAdvisor")
        if "jev_bridge_checkpoint" not in ops_context.tools or "jev_bridge_review" in ops_context.tools:
            raise SystemExit("isolated ops registration did not expose the producer bridge tool")

        print(
            json.dumps(
                {
                    "isolated_plugin_registration": "pass",
                    "default_consumer_hook": True,
                    "default_scope_guard_hook": True,
                    "default_scope_contract_hook": True,
                    "default_targeted_reading_advisor": True,
                    "ops_producer_advisor_absent": True,
                    "plugin_manager_invocation": "isolated_context_only",
                    "unrelated_live_hooks_invoked": False,
                    "ledger": str(ledger),
                    "ledger_rows": len(rows),
                    "ledger_mode": oct(ledger.stat().st_mode & 0o777),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def test_real_registry_stopped_review_recovery():
    """Exercise the host registration/dispatch seam, never the live bridge store."""
    import os
    from unittest.mock import patch
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry

    plugin = _load_plugin()
    bridge = sys.modules[f"{PACKAGE_NAME}.bridge"]
    with tempfile.TemporaryDirectory(prefix="jev-review-registry-") as temporary:
        root = Path(temporary)
        with patch.dict(os.environ, {"HERMES_HOME": str(root)}):
            manager = PluginManager(scope_key=str(root))

            class Context(PluginContext):
                @property
                def profile_name(self):
                    return "default"

                def get_config(self, key, default=None):
                    return {
                        "bridge_dir": str(root / "bridge"), "role": "consumer",
                        "consumer_profiles": ["default"], "producer_profiles": ["ops"],
                        "default_scope.enabled": True,
                    }.get(key, default)

            context = Context(PluginManifest(name="jev-route-screening", source="local", path=str(ROOT)), manager)
            bridge.register(context)
            schema = registry.get_definitions({"jev_bridge_review"})[0]["function"]
            assert schema["description"]
            assert {"scope", "trigger_id", "decision", "readback"} <= set(schema["parameters"]["properties"])
            guard = next(h.__self__ for h in manager._hooks["pre_tool_call"] if isinstance(h.__self__, bridge.DefaultScopeGuard))
            guard.decision_fn = lambda _: {"label": "scope_drift", "confidence": 0.9, "reason": "fixture drift"}
            session = "fixture-default-session"
            manager.invoke_hook("pre_llm_call", session_id=session, turn_id="turn-1", user_message="Read configuration only", completion_conditions=["report exact configuration"])
            blocked = manager.invoke_hook("pre_tool_call", tool_name="write_file", args={"path": str(root / "must-not-write")}, session_id=session, turn_id="turn-1")
            assert any(row and row.get("action") == "block" for row in blocked)
            review_allowed = manager.invoke_hook("pre_tool_call", tool_name="jev_bridge_review", args={}, session_id=session, turn_id="turn-2")
            assert not any(row and row.get("action") == "block" for row in review_allowed)

            def dispatch(payload, caller=session):
                return json.loads(registry.dispatch("jev_bridge_review", payload, scope=str(root), session_id=caller))

            pending = dispatch({})
            assert len(pending) == 1
            row = pending[0]
            assert dispatch({}, "different-session") == []
            payload = {"scope": "default_scope", "trigger_id": row["trigger_id"], "decision": "no_intervention", "readback": row["readback_template"]}
            assert "error" in dispatch(payload, "different-session")
            invalid = dict(payload, readback=dict(payload["readback"], original_request="wrong request"))
            assert "error" in dispatch(invalid)
            assert bridge._active_scope_control(guard.store, row, scope="default") is not None
            assert dispatch(dict(payload, decision="intervene"))["state"] == "intervention_requested"
            assert bridge._active_scope_control(guard.store, row, scope="default") is not None
            # The release must not settle an independent worker stop.
            with guard.store.interprocess_lock():
                guard.consumer._append_control_locked(row, "provisional_stop", source="jev", reason="worker fixture", scope="worker")
            assert dispatch(payload)["state"] == "no_intervention"
            assert dispatch(payload)["state"] == "already_settled"
            assert dispatch({}) == []
            assert bridge._active_scope_control(guard.store, row, scope="default") is None
            assert bridge._active_scope_control(guard.store, row, scope="worker") is not None
            guard.decision_fn = lambda _: {"label": "insufficient_information", "confidence": 0.5, "reason": "fixture advisory"}
            continued = manager.invoke_hook("pre_tool_call", tool_name="read_file", args={"path": str(root / "harmless")}, session_id=session, turn_id="turn-3")
            assert not any(result and result.get("action") == "block" for result in continued)
            assert not (root / "must-not-write").exists()


if __name__ == "__main__":
    main()
