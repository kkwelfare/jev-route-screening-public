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


if __name__ == "__main__":
    main()
