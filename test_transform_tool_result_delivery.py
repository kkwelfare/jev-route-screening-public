"""Isolated feasibility probe for the host transform_tool_result seam.

This test intentionally does not change the Jev production plugin.  Each case
runs in a fresh Python process with a temporary Hermes home containing one
mock Jev plugin and one minimal hermes-agent skill.  The real
model_tools.handle_function_call("skill_view", ...) path then exercises the
host pre/post/transform dispatch around the built-in skill_view tool.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SCRATCH = Path(os.environ.get("JEV_TEST_SCRATCH") or tempfile.gettempdir())
SCRIPT = Path(__file__).resolve()
_SENTINEL = "RESULT_JSON="
_PROVIDER_SLEEP_SECONDS = 0.8
_PROMPT_RETURN_FRACTION = 0.75


_PLUGIN_YAML = """name: transform_jev_probe
version: 0.0.1
provides_hooks:
  - transform_tool_result
"""

_PLUGIN_INIT = r'''"""Temporary test-only mock Jev transform plugin."""

from __future__ import annotations

import json
import os
import time


def _mock_jev_provider(*, tool_name: str, args: dict, result: str) -> dict:
    mode = os.environ.get("JEV_PROBE_CASE", "success")
    if mode == "raise":
        raise RuntimeError("mock_jev_provider_failure")
    if mode == "timeout":
        time.sleep(float(os.environ.get("JEV_PROBE_SLEEP_SECONDS", "0.8")))
    return {
        "source": "mock-jev",
        "tool_name": tool_name,
        "skill_name": args.get("name"),
        "result_was_json": isinstance(result, str),
        "route": "configuration_reference",
    }


def _transform_tool_result(tool_name: str = "", args: dict | None = None,
                           result: Any = None, **_: Any) -> str | None:
    if tool_name != "skill_view" or not isinstance(args, dict) or not isinstance(result, str):
        return None
    advisory = _mock_jev_provider(tool_name=tool_name, args=args, result=result)
    original = json.loads(result)
    if not isinstance(original, dict):
        raise AssertionError("skill_view result was not a JSON object")
    original["jev_advisory"] = advisory
    return json.dumps(original, sort_keys=True)


def register(ctx) -> None:
    ctx.register_hook("transform_tool_result", _transform_tool_result)
'''

_SKILL_MD = """---
name: hermes-agent
description: Isolated transform seam probe skill
metadata:
  hermes:
    tags: [test, seam]
---
# Hermes agent probe

This content is a local fixture used only to verify the host skill_view result.
"""


def _write_fixture(home: Path, *, timeout: float) -> None:
    plugin_dir = home / "plugins" / "transform_jev_probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(_PLUGIN_YAML, encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(_PLUGIN_INIT, encoding="utf-8")

    skill_dir = home / "skills" / "autonomous-ai-agents" / "hermes-agent"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")

    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - transform_jev_probe\n"
        f"  hook_callback_timeout: {timeout}\n",
        encoding="utf-8",
    )


def _write_production_fixture(home: Path, *, timeout: float) -> Path:
    """Expose the installed default plugin bytes through fresh discovery."""
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True)
    (plugins_dir / "jev-route-screening").symlink_to(SCRIPT.parent, target_is_directory=True)
    skill_dir = home / "skills" / "autonomous-ai-agents" / "hermes-agent"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    bridge_dir = home / "jev-bridge"
    config = {
        "plugins": {
            "enabled": ["jev-route-screening"],
            "hook_callback_timeout": timeout,
            "entries": {
                "jev-route-screening": {
                    "settings": {
                        "bridge_dir": str(bridge_dir),
                        "role": "consumer",
                        "consumer_profiles": ["default"],
                        "targeted_reading": {
                            "enabled": True,
                            "once_per_session": True,
                            "timeout_seconds": 5.0,
                        },
                    }
                }
            },
        }
    }
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    return bridge_dir


def _production_fixture_response(router_module, route: str) -> dict[str, Any]:
    return {
        "model": "jev-production-fixture",
        "answers": {
            "reference_route": {
                "choice": route,
                "probabilities": {
                    label: (1.0 if label == route else 0.0)
                    for label in router_module.ROUTE_BUNDLES
                },
                "confidence": 0.93,
            }
        },
        "usage": None,
    }


def _production_child_case(case: str) -> dict[str, Any]:
    # These imports and discovery calls intentionally happen in a fresh child
    # so installed-plugin registration cannot be satisfied by this test module.
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins_mod
    import model_tools
    from tools.registry import registry

    if not plugins_mod.has_hook("transform_tool_result"):
        loaded = plugins_mod.get_plugin_manager()._plugins.get("jev-route-screening")
        from hermes_cli.config import load_config_readonly
        probe = plugins_mod.PluginContext(loaded.manifest, plugins_mod.get_plugin_manager()) if loaded else None
        probe_values = {
            key: (probe.get_config(key, "<missing>") if probe else "<no-loaded-plugin>")
            for key in ("bridge_dir", "role", "consumer_profiles", "targeted_reading", "targeted_reading.enabled")
        }
        trace = []
        class _TraceContext:
            profile_name = "default"
            def get_config(self, key, default=None):
                return probe.get_config(key, default)
            def register_hook(self, name, callback):
                trace.append(("hook", name, getattr(callback, "__name__", type(callback).__name__)))
            def register_tool(self, *args, **kwargs):
                trace.append(("tool", args, sorted(kwargs)))
        if loaded:
            loaded.module.register(_TraceContext())
        raise AssertionError(
            f"fresh production plugin did not register transform_tool_result: "
            f"loaded={loaded!r} config={Path(os.environ['HERMES_HOME']) / 'config.yaml'} "
            f"parsed={load_config_readonly()!r} probe={probe_values!r} "
            f"profile_env={os.environ.get('HERMES_PROFILE')!r} module={getattr(loaded, 'module', None)!r} trace={trace!r}"
        )
    if not plugins_mod.has_hook("pre_tool_call"):
        raise AssertionError("production plugin did not register the required default scope guard hook")
    scope_guard_hooks = [
        callback
        for callback in plugins_mod.get_plugin_manager()._hooks.get("pre_tool_call", [])
        if getattr(getattr(callback, "__self__", None), "__class__", type(None)).__name__ == "DefaultScopeGuard"
    ]
    if len(scope_guard_hooks) != 1 or not getattr(scope_guard_hooks[0].__self__, "enabled", False):
        raise AssertionError("production plugin did not expose one enabled default scope guard")
    # Keep this transform probe focused on the host seam while still dispatching
    # the enabled guard hook.  The guard's behavior has dedicated regression
    # coverage; this offline decision avoids counting a second provider path.
    scope_guard_hooks[0].__self__.decision_fn = lambda _request: {
        "label": "progressing",
        "confidence": 0.9,
        "reason": "offline transform seam scope fixture",
    }
    router_module = next(
        module for module in list(sys.modules.values())
        if module is not None and hasattr(module, "TargetedReadingAdvisor") and hasattr(module, "ROUTE_BUNDLES")
    )
    bridge_dir = Path(os.environ["JEV_PRODUCTION_BRIDGE_DIR"])
    calls: list[dict[str, Any]] = []

    def _mock_provider(spec, body, timeout):
        calls.append({"model": spec.model, "timeout": timeout})
        if case == "raise":
            raise RuntimeError("mock_production_provider_failure")
        if case == "timeout":
            time.sleep(float(os.environ.get("JEV_PROBE_SLEEP_SECONDS", "0.8")))
        state = json.loads(body["state"])
        request = str(state["request"]).casefold()
        route = "cli_reference" if "cli" in request else "configuration_reference"
        return _production_fixture_response(router_module, route)

    router_module.bridge._post_provider = _mock_provider
    task_id = "production-task"
    session_id = "production-session"

    if case == "success":
        initial = lifecycle.invoke_hook(
            "pre_llm_call", session_id=session_id, task_id=task_id, turn_id="turn-1",
            user_message="read configuration settings", is_first_turn=True, platform="test",
        )
        if not any(isinstance(item, dict) and "context" in item for item in initial):
            raise AssertionError(f"initial targeted-reading trigger missing: {initial!r}")
        first = model_tools.handle_function_call(
            "skill_view", {"name": "hermes-agent"}, task_id=task_id,
            tool_call_id="production-tool-1", session_id=session_id, turn_id="turn-1",
        )
        first_obj = json.loads(first)
        if first_obj.get("success") is not True:
            raise AssertionError(f"initial production skill_view failed: {first_obj!r}")
        if first_obj.get("jev_advisory", {}).get("route") != "configuration_reference":
            raise AssertionError("initial production advisory was not attached separately")
        if "jev_advisory" in first_obj.get("content", ""):
            raise AssertionError("initial advisory contaminated original skill content")
        if len(calls) != 1:
            raise AssertionError(f"initial transform made unexpected provider calls: {len(calls)}")

        later = lifecycle.invoke_hook(
            "pre_llm_call", session_id=session_id, task_id=task_id, turn_id="turn-2",
            user_message="show the CLI reference", is_first_turn=False, platform="test",
        )
        if later:
            raise AssertionError(f"ordinary later turn unexpectedly returned context: {later!r}")
        second = model_tools.handle_function_call(
            "skill_view", {"name": "hermes-agent"}, task_id=task_id,
            tool_call_id="production-tool-2", session_id=session_id, turn_id="turn-2",
        )
        second_obj = json.loads(second)
        if second_obj.get("success") is not True:
            raise AssertionError(f"later production skill_view failed: {second_obj!r}")
        if second_obj.get("jev_advisory", {}).get("route") != "cli_reference":
            raise AssertionError("later production advisory did not refresh")
        if len(calls) != 2:
            raise AssertionError(f"later turn did not make exactly one refresh call: {len(calls)}")
        duplicate = model_tools.handle_function_call(
            "skill_view", {"name": "hermes-agent", "file_path": "references/cli-reference.md"},
            task_id=task_id, tool_call_id="production-tool-3", session_id=session_id, turn_id="turn-2",
        )
        duplicate_obj = json.loads(duplicate)
        if duplicate_obj.get("jev_advisory", {}).get("route") != "cli_reference":
            raise AssertionError("same-turn duplicate lost the advisory")
        if len(calls) != 2:
            raise AssertionError("same-turn hub/reference duplicate triggered another provider call")
        lifecycle.invoke_hook(
            "pre_llm_call", session_id=session_id, task_id=task_id, turn_id="turn-3",
            user_message="ordinary chat", is_first_turn=False, platform="test",
        )
        model_tools.handle_function_call(
            "skill_view", {"name": "other-skill"}, task_id=task_id,
            tool_call_id="production-tool-4", session_id=session_id, turn_id="turn-3",
        )
        if len(calls) != 2:
            raise AssertionError("ordinary chat or another skill triggered a provider call")
        rows = [
            json.loads(line)
            for line in (bridge_dir / "skill-reading-advisory.jsonl").read_text().splitlines()
        ]
        advisories = [row for row in rows if row.get("kind") == "skill_reading_advisory"]
        observations = [row for row in rows if row.get("kind") == "skill_read_observation"]
        if len(advisories) != 2:
            raise AssertionError(f"unexpected production advisory count: {len(advisories)}")
        latest_id = advisories[-1]["request_id"]
        if not any(row.get("request_id") == latest_id for row in observations):
            raise AssertionError("post_tool observation was not bound to the later transformed read")
        return {
            "case": case, "fresh_process": True, "hook_registered": True,
            "skill_read_success": True, "initial_and_later_refresh": True,
            "same_turn_deduplicated": True, "ordinary_and_other_skill_ignored": True,
            "observation_bound": True, "provider_calls": len(calls),
        }

    baseline = registry.dispatch(
        "skill_view", {"name": "hermes-agent"}, task_id="failure-baseline-task", session_id="failure-baseline-session",
    )
    lifecycle.invoke_hook(
        "pre_llm_call", session_id=session_id, task_id=task_id, turn_id="turn-failure",
        user_message="show the CLI reference", is_first_turn=False, platform="test",
    )
    started = time.monotonic()
    observed = model_tools.handle_function_call(
        "skill_view", {"name": "hermes-agent"}, task_id=task_id,
        tool_call_id="production-failure-tool", session_id=session_id, turn_id="turn-failure",
    )
    elapsed = time.monotonic() - started
    if observed != baseline:
        raise AssertionError(f"{case} production path changed the original result: baseline={baseline!r} observed={observed!r}")
    provider_sleep = float(os.environ.get("JEV_PROBE_SLEEP_SECONDS", str(_PROVIDER_SLEEP_SECONDS)))
    prompt_limit = provider_sleep * _PROMPT_RETURN_FRACTION
    if elapsed >= prompt_limit:
        raise AssertionError(
            f"{case} production transform did not fail open before the provider completed: "
            f"{elapsed:.3f}s >= {prompt_limit:.3f}s"
        )
    return {
        "case": case, "fresh_process": True, "hook_registered": True,
        "skill_read_success": True, "original_result_preserved": True,
        "fail_open": True, "elapsed_seconds": round(elapsed, 4), "provider_calls": len(calls),
    }


def _child_case(case: str) -> dict[str, Any]:
    # Imports are deliberately inside the child path so every case gets a
    # fresh PluginManager and a fresh registry/discovery cycle.
    import hermes_cli.plugins as plugins_mod
    import model_tools
    from tools.registry import registry

    if not plugins_mod.has_hook("transform_tool_result"):
        raise AssertionError("fresh PluginManager did not register transform_tool_result")

    raw = registry.dispatch(
        "skill_view",
        {"name": "hermes-agent"},
        task_id="baseline-task",
        session_id="baseline-session",
    )
    if not isinstance(raw, str):
        raise AssertionError("baseline skill_view dispatch did not return a string")
    raw_obj = json.loads(raw)
    if not isinstance(raw_obj, dict) or raw_obj.get("success") is not True:
        raise AssertionError(f"baseline skill_view read failed: {raw_obj!r}")

    started = time.monotonic()
    observed = model_tools.handle_function_call(
        "skill_view",
        {"name": "hermes-agent"},
        task_id="hook-task",
        tool_call_id="hook-tool-call",
        session_id="hook-session",
        turn_id="hook-turn",
    )
    elapsed = time.monotonic() - started
    if not isinstance(observed, str):
        raise AssertionError("real handle_function_call did not return a string")
    observed_obj = json.loads(observed)
    if not isinstance(observed_obj, dict) or observed_obj.get("success") is not True:
        raise AssertionError(f"real skill_view read failed: {observed_obj!r}")

    result: dict[str, Any] = {
        "case": case,
        "fresh_process": True,
        "hook_registered": True,
        "skill_read_success": True,
        "elapsed_seconds": round(elapsed, 4),
    }

    if case == "success":
        if set(observed_obj) != set(raw_obj) | {"jev_advisory"}:
            raise AssertionError("successful transform did not preserve the original key set")
        for key, value in raw_obj.items():
            if observed_obj.get(key) != value:
                raise AssertionError(f"successful transform changed original field {key!r}")
        advisory = observed_obj.get("jev_advisory")
        if not isinstance(advisory, dict) or advisory.get("source") != "mock-jev":
            raise AssertionError("mock Jev advisory was not delivered separately")
        if advisory.get("tool_name") != "skill_view":
            raise AssertionError("transform hook did not receive the real skill_view name")
        if advisory.get("skill_name") != "hermes-agent":
            raise AssertionError("transform hook did not receive the real skill_view args")
        if advisory.get("route") != "configuration_reference":
            raise AssertionError("mock Jev provider result was not delivered")
        if "jev_advisory" in observed_obj.get("content", ""):
            raise AssertionError("advisory was not kept separate from original skill content")
        result["original_fields_preserved"] = True
        result["advisory_separate"] = True
        result["model_context_result"] = "transformed skill_view string"
    elif case in {"raise", "timeout"}:
        if observed != raw:
            raise AssertionError(f"{case} path changed the tool result instead of failing open")
        if "jev_advisory" in observed_obj:
            raise AssertionError(f"{case} path delivered an advisory despite provider failure")
        result["original_result_preserved"] = True
        result["fail_open"] = True
        if case == "timeout":
            provider_sleep = float(os.environ.get("JEV_PROBE_SLEEP_SECONDS", str(_PROVIDER_SLEEP_SECONDS)))
            prompt_limit = provider_sleep * _PROMPT_RETURN_FRACTION
            if elapsed >= prompt_limit:
                raise AssertionError(
                    f"transform timeout blocked the skill read until the provider completed: "
                    f"{elapsed:.3f}s >= {prompt_limit:.3f}s"
                )
            result["timeout_returned_promptly"] = True
    else:
        raise AssertionError(f"unknown child case: {case}")

    return result


def _run_parent() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for case in ("success", "raise", "timeout"):
        with tempfile.TemporaryDirectory(prefix="jev-transform-seam-", dir=SCRATCH) as root:
            home = Path(root) / "hermes-home"
            home.mkdir()
            _write_fixture(home, timeout=0.05)
            env = os.environ.copy()
            env.update({
                "HERMES_HOME": str(home),
                "HERMES_PROFILE": "default",
                "JEV_PROBE_CASE": case,
                "JEV_PROBE_SLEEP_SECONDS": str(_PROVIDER_SLEEP_SECONDS),
                "PYTHONPATH": env.get("PYTHONPATH", ""),
            })
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--child", case],
                cwd=str(SCRIPT.parent),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise AssertionError(
                    f"child case {case} failed with exit {completed.returncode}\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            line = next((line for line in completed.stdout.splitlines() if line.startswith(_SENTINEL)), None)
            if line is None:
                raise AssertionError(
                    f"child case {case} produced no result sentinel\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            cases.append(json.loads(line[len(_SENTINEL):]))

    production_cases: list[dict[str, Any]] = []
    for case in ("success", "raise", "timeout"):
        with tempfile.TemporaryDirectory(prefix="jev-production-transform-", dir=SCRATCH) as root:
            home = Path(root) / ".hermes"
            home.mkdir()
            bridge_dir = _write_production_fixture(home, timeout=0.05 if case == "timeout" else 0.5)
            env = os.environ.copy()
            env.update({
                "HOME": str(root),
                "HERMES_HOME": str(home),
                "HERMES_PROFILE": "default",
                "JEV_PRODUCTION_BRIDGE_DIR": str(bridge_dir),
                "JEV_PROBE_SLEEP_SECONDS": str(_PROVIDER_SLEEP_SECONDS),
                "PYTHONPATH": env.get("PYTHONPATH", ""),
            })
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--production-child", case],
                cwd=str(SCRIPT.parent), env=env, text=True, capture_output=True, check=False,
            )
            if completed.returncode != 0:
                raise AssertionError(
                    f"production child case {case} failed with exit {completed.returncode}\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            line = next((line for line in completed.stdout.splitlines() if line.startswith(_SENTINEL)), None)
            if line is None:
                raise AssertionError(
                    f"production child case {case} produced no result sentinel\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            production_cases.append(json.loads(line[len(_SENTINEL):]))

    summary = {
        "status": "pass",
        "cases": cases,
        "production_cases": production_cases,
        "production_files_changed": False,
        "live_api_calls": False,
        "gateway_restart": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("success", "raise", "timeout"))
    parser.add_argument("--production-child", choices=("success", "raise", "timeout"))
    args = parser.parse_args()
    if args.child:
        print(_SENTINEL + json.dumps(_child_case(args.child), sort_keys=True))
        return 0
    if args.production_child:
        print(_SENTINEL + json.dumps(_production_child_case(args.production_child), sort_keys=True))
        return 0
    return _run_parent()


if __name__ == "__main__":
    raise SystemExit(main())
