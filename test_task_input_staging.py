from __future__ import annotations

import datetime as dt
import importlib.util
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "jev_task_input_staging_test_plugin"


def _load_staging_module():
    spec = importlib.util.spec_from_file_location("jev_task_input_staging_module", ROOT / "task_input_staging.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load task-input staging module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Jev plugin package")
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)
    return package


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _decision_input() -> dict[str, Any]:
    now = _now()
    return {
        "schema_version": 1,
        "scope": "one bounded test task",
        "objective": "exercise optional task-input staging",
        "target_candidates": ["https://example.test/reference"],
        "requested_outcome": "verify safe task-bound persistence",
        "action_mode": "production",
        "allowed_actions": ["read test fixture"],
        "forbidden_actions": ["publish", "merge"],
        "decision_points": ["read the exact marker back"],
        "freshness": {"status": "not_time_sensitive", "observed_at": now, "valid_until": None, "rule": "fixed fixture"},
        "deduplication": {"status": "checked", "basis": "one test invocation", "checked_at": now},
        "baseline": {"fixture": "offline"},
        "completion_conditions": ["marker is task-bound and read back before unblock"],
        "prior_context": {},
        "signals": [],
        "feedback": [],
        "output_requirements": {"format": "json"},
        "uncertainty": [],
        "provenance": {"source": "offline test fixture", "captured_at": now, "source_references": ["https://example.test/reference"]},
        "domain_extension": None,
    }


class FakeKanban:
    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.by_key: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.unblock_count = 0
        self.terminal_exit_code = 0
        self.corrupt_body_after_edit = False

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(args)))
        if name == "kanban_create":
            key = args["idempotency_key"]
            task_id = self.by_key.get(key)
            if task_id is None:
                task_id = "t_fixture1234"
                self.by_key[key] = task_id
                fields = {key: value for key, value in args.items() if key not in {"initial_status", "idempotency_key", "board"}}
                self.tasks[task_id] = {"id": task_id, **fields, "status": args["initial_status"]}
            task = self.tasks[task_id]
            return {"ok": True, "task_id": task_id, "status": task["status"]}
        if name == "kanban_show":
            task = self.tasks.get(args["task_id"])
            return {"task": dict(task)} if task is not None else {"error": "missing"}
        if name == "terminal":
            words = shlex.split(args["command"])
            edit_index = words.index("edit")
            task_id = words[edit_index + 1]
            body_index = words.index("--body")
            body = words[body_index + 1]
            if self.terminal_exit_code == 0:
                self.tasks[task_id]["body"] = body
                if self.corrupt_body_after_edit:
                    self.tasks[task_id]["body"] += "jev_decision_input_json: {}\n"
            return {"output": "", "exit_code": self.terminal_exit_code}
        if name == "kanban_unblock":
            task = self.tasks[args["task_id"]]
            if task["status"] != "blocked":
                return {"ok": False, "error": "not blocked"}
            task["status"] = "ready"
            self.unblock_count += 1
            return {"ok": True, "task_id": args["task_id"]}
        raise AssertionError(f"unexpected tool dispatch: {name}")


class StagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_staging_module()
        self.params = {"task": {"title": "Fixture task", "assignee": "ops", "body": "Bounded fixture."}, "decision_input": _decision_input()}

    def test_success_binds_generated_task_id_and_reads_back_before_unblock(self) -> None:
        kanban = FakeKanban()
        result = json.loads(self.module.stage_task_input(self.params, kanban.dispatch))
        task = kanban.tasks["t_fixture1234"]
        persisted = self.module._extract_persisted(task["body"])

        self.assertTrue(result["ok"])
        self.assertEqual(result["task_id"], "t_fixture1234")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(persisted["provenance"]["task_id"], "t_fixture1234")
        expected = dict(self.params["decision_input"])
        expected["provenance"] = dict(expected["provenance"], task_id="t_fixture1234")
        self.assertEqual(persisted, expected)
        self.assertEqual(task["body"].count(self.module.MARKER), 1)
        self.assertEqual(task["status"], "ready")
        names = [name for name, _ in kanban.calls]
        self.assertLess(names.index("kanban_show"), names.index("terminal"))
        self.assertLess(names.index("terminal"), names.index("kanban_unblock"))
        create_args = next(args for name, args in kanban.calls if name == "kanban_create")
        self.assertEqual(create_args["initial_status"], "blocked")
        self.assertTrue(create_args["idempotency_key"].startswith("jev-input-stage-"))

    def test_identical_retry_reuses_one_card_and_does_not_unblock_twice(self) -> None:
        kanban = FakeKanban()
        first = json.loads(self.module.stage_task_input(self.params, kanban.dispatch))
        second = json.loads(self.module.stage_task_input(self.params, kanban.dispatch))
        self.assertTrue(first["ok"] and second["ok"])
        self.assertTrue(second["already_staged"])
        self.assertEqual(len(kanban.tasks), 1)
        self.assertEqual(kanban.unblock_count, 1)

    def test_missing_or_invalid_input_fails_before_task_creation(self) -> None:
        kanban = FakeKanban()
        missing = json.loads(self.module.stage_task_input({"task": self.params["task"]}, kanban.dispatch))
        invalid_input = dict(self.params["decision_input"])
        invalid_input.pop("output_requirements")
        invalid = json.loads(self.module.stage_task_input({"task": self.params["task"], "decision_input": invalid_input}, kanban.dispatch))
        self.assertFalse(missing["ok"])
        self.assertFalse(invalid["ok"])
        self.assertEqual(kanban.calls, [])
        self.assertEqual(kanban.tasks, {})

    def test_edit_or_persisted_readback_failure_leaves_task_blocked(self) -> None:
        for failure in ("edit", "readback"):
            with self.subTest(failure=failure):
                kanban = FakeKanban()
                if failure == "edit":
                    kanban.terminal_exit_code = 1
                else:
                    kanban.corrupt_body_after_edit = True
                result = json.loads(self.module.stage_task_input(self.params, kanban.dispatch))
                task = kanban.tasks["t_fixture1234"]
                self.assertFalse(result["ok"])
                self.assertEqual(task["status"], "blocked")
                self.assertEqual(kanban.unblock_count, 0)

    def test_optional_registration_is_default_consumer_only_and_preserves_existing_tools(self) -> None:
        plugin = _load_plugin()

        class Context:
            def __init__(self, profile: str, role: str, enabled: bool, root: Path) -> None:
                self.profile_name = profile
                self._config = {
                    "bridge_dir": str(root), "role": role,
                    "producer_profiles": ["ops"], "consumer_profiles": ["default"],
                    "targeted_reading.enabled": False,
                    "task_input_staging_enabled": enabled,
                }
                self.tools: dict[str, Any] = {}
                self.hooks: dict[str, list[Any]] = {}

            def get_config(self, key: str, default: Any = None) -> Any:
                return self._config.get(key, default)

            def register_tool(self, **kwargs: Any) -> None:
                self.tools[kwargs["name"]] = kwargs

            def register_hook(self, name: str, callback: Any) -> None:
                self.hooks.setdefault(name, []).append(callback)

            def dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
                raise AssertionError("registration test must not execute tool side effects")

        with tempfile.TemporaryDirectory(prefix="jev-task-staging-plugin-") as temporary:
            root = Path(temporary)
            enabled_default = Context("default", "consumer", True, root / "default-enabled")
            plugin.register(enabled_default)
            self.assertIn("jev_stage_task_input", enabled_default.tools)
            self.assertIn("jev_bridge_review", enabled_default.tools)
            self.assertNotIn("jev_bridge_checkpoint", enabled_default.tools)

            disabled_default = Context("default", "consumer", False, root / "default-disabled")
            plugin.register(disabled_default)
            self.assertNotIn("jev_stage_task_input", disabled_default.tools)
            self.assertIn("jev_bridge_review", disabled_default.tools)

            enabled_worker = Context("ops", "producer", True, root / "worker")
            plugin.register(enabled_worker)
            self.assertNotIn("jev_stage_task_input", enabled_worker.tools)
            self.assertIn("jev_bridge_checkpoint", enabled_worker.tools)
            self.assertNotIn("jev_bridge_review", enabled_worker.tools)


if __name__ == "__main__":
    unittest.main()
