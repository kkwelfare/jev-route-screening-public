from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parent
OPS_ROOT = Path(os.environ.get("JEV_OPS_PLUGIN_ROOT") or DEFAULT_ROOT).expanduser().resolve()


def _load_package(package_name: str, root: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        package_name,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load plugin package from {root}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return package


def _load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeContext:
    def __init__(self, bridge_dir: Path) -> None:
        self.profile_name = "default"
        self._config = {
            "bridge_dir": str(bridge_dir),
            "role": "consumer",
            "consumer_profiles": ["default"],
            "producer_profiles": ["ops"],
            "targeted_reading.enabled": False,
        }
        self.tools: dict[str, Any] = {}
        self.hooks: dict[str, list[Any]] = {}

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def register_tool(self, **kwargs: Any) -> None:
        self.tools[kwargs["name"]] = kwargs["handler"]

    def register_hook(self, name: str, callback: Any) -> None:
        self.hooks.setdefault(name, []).append(callback)


@contextmanager
def _environment(**values: str):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value == "":
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class WorkerBindingContextTests(unittest.TestCase):
    def test_dispatcher_observer_binds_current_ops_worker_without_env_identity(self) -> None:
        default_plugin = _load_package("jev_default_worker_binding_target", DEFAULT_ROOT)
        ops_bridge = _load_module("jev_ops_worker_binding_target", OPS_ROOT / "bridge.py")
        with tempfile.TemporaryDirectory(prefix="jev-worker-binding-") as temporary:
            root = Path(temporary) / "bridge"
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            context = FakeContext(root)
            with _environment(
                HERMES_PROFILE="default",
                HERMES_KANBAN_TASK="",
                HERMES_KANBAN_RUN_ID="",
                HERMES_KANBAN_WORKSPACE="",
            ):
                default_plugin.register(context)
                callbacks = context.hooks.get("on_kanban_worker_spawned", [])
                self.assertEqual(len(callbacks), 1)
                payload = {
                    "task_id": "t_binding_regression",
                    "run_id": 3142,
                    "profile_name": "default",
                    "assignee": "ops",
                    "board": "default",
                    "worker_pid": os.getpid(),
                    "workspace_path": str(workspace),
                }
                callbacks[0](**payload)
                callbacks[0](**payload)

            store = ops_bridge.BridgeStore(root)
            rows = store.worker_bindings()
            self.assertEqual(len(rows), 1, "observer callback must be idempotent for one spawn")
            self.assertEqual(rows[0]["source"], "on_kanban_worker_spawned")
            self.assertEqual(rows[0]["task_id"], "t_binding_regression")
            self.assertEqual(rows[0]["run_id"], "3142")
            self.assertEqual(rows[0]["worker_pid"], os.getpid())
            self.assertEqual(rows[0]["worker_profile"], "ops")
            self.assertEqual(Path(rows[0]["workspace_path"]).resolve(), workspace.resolve())

            with _environment(
                HERMES_PROFILE="ops",
                HERMES_KANBAN_TASK="",
                HERMES_KANBAN_RUN_ID="",
                HERMES_KANBAN_WORKSPACE=str(workspace),
            ):
                binding = ops_bridge._load_current_worker_binding(root, "ops")

            self.assertIsNotNone(binding)
            self.assertEqual(binding["task_id"], "t_binding_regression")
            self.assertEqual(binding["run_id"], "3142")
            self.assertEqual(binding["worker_pid"], os.getpid())

            with _environment(
                HERMES_PROFILE="ops",
                HERMES_KANBAN_TASK="",
                HERMES_KANBAN_RUN_ID="",
                HERMES_KANBAN_WORKSPACE=str(workspace / "other"),
            ):
                self.assertIsNone(ops_bridge._load_current_worker_binding(root, "ops"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
