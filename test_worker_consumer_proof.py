from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "jev_route_screening_worker_consumer_proof_target"


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


plugin = _load_plugin()
bridge = sys.modules[f"{PACKAGE_NAME}.bridge"]


class FakeContext:
    """Minimal host context that exercises the plugin's real register seam."""

    def __init__(self, profile_name: str, bridge_dir: Path, role: str) -> None:
        self.profile_name = profile_name
        self._config = {
            "bridge_dir": str(bridge_dir),
            "role": role,
            "producer_profiles": ["ops"],
            "consumer_profiles": ["default"],
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


def _bound_method(callback: Any, class_name: str) -> Any:
    owner = getattr(callback, "__self__", None)
    if owner is not None and owner.__class__.__name__ == class_name:
        return callback
    return None


class WorkerConsumerProofTests(unittest.TestCase):
    def test_registered_worker_hook_to_consumer_and_default_readback(self) -> None:
        secret_sentinel = "PROOF_SECRET_SENTINEL_MUST_NOT_PERSIST"
        with tempfile.TemporaryDirectory(prefix="jev-worker-consumer-proof-") as temporary:
            root = Path(temporary)
            contract = bridge.TaskContract(
                task_id="proof-task-001",
                run_id="9001",
                profile="ops",
                started_at=1000.0,
                original_request="prove the bounded producer to consumer route",
                criteria=("event identity is bound", "consumer decision is recorded"),
                current_unknown="whether the same event reaches default readback",
                next_action="read back the decision and release any provisional stop",
                evidence_refs=("kanban://task/proof-task-001",),
            )

            producer_context = FakeContext("ops", root, "producer")
            with _environment(
                HERMES_PROFILE="ops",
                HERMES_KANBAN_TASK=contract.task_id,
                HERMES_KANBAN_RUN_ID=contract.run_id,
            ):
                plugin.register(producer_context)
                producer_callback = next(
                    (
                        callback
                        for callback in producer_context.hooks.get("post_tool_call", [])
                        if _bound_method(callback, "Producer") is not None
                    ),
                    None,
                )
                self.assertIsNotNone(producer_callback)
                self.assertIn("jev_bridge_checkpoint", producer_context.tools)
                self.assertNotIn("jev_bridge_review", producer_context.tools)

                producer = producer_callback.__self__
                producer.contract_loader = lambda: contract
                producer.clock = lambda: 1300.0
                producer.cadence_seconds = 0.0

                # These are normal post-tool callbacks, not manual checkpoint input.
                # The sentinel is supplied through a content field that must never be
                # copied into the bounded projection or any JSONL ledger.
                for index in range(3):
                    producer_callback(
                        tool_name="write_file",
                        args={
                            "path": f"proof-output-{index}.txt",
                            "content": secret_sentinel,
                        },
                        task_id="runtime-callback-id-is-not-board-identity",
                        result={"ok": True, "changed": True},
                        status="ok",
                    )

            store = bridge.BridgeStore(root)
            events = store.events()
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event["schema_version"], bridge.SCHEMA_VERSION)
            self.assertEqual(event["category"], bridge.WORKER_CATEGORY)
            self.assertFalse(event["synthetic"])
            self.assertEqual(event["source"], "worker_hook")
            self.assertTrue(event["projection_safe"])
            self.assertEqual(event["profile"], "ops")
            self.assertEqual(event["producer_profile"], "ops")
            self.assertEqual(event["task_id"], contract.task_id)
            self.assertEqual(str(event["run_id"]), contract.run_id)
            self.assertEqual(event["event_key"], f"{contract.task_id}:{contract.run_id}:{event['checkpoint_id']}")
            self.assertGreaterEqual(event["completed_tools"], bridge.MIN_COMPLETED_TOOLS)
            self.assertGreaterEqual(event["tools_since_previous"], bridge.MIN_COMPLETED_TOOLS)
            self.assertGreaterEqual(event["changed_evidence_age_seconds"], bridge.CHECKPOINT_CADENCE_SECONDS)

            default_context = FakeContext("default", root, "consumer")
            with _environment(
                HERMES_PROFILE="default",
                HERMES_KANBAN_TASK="",
                HERMES_KANBAN_RUN_ID="",
            ):
                plugin.register(default_context)

            self.assertNotIn("jev_bridge_checkpoint", default_context.tools)
            self.assertIn("jev_bridge_review", default_context.tools)
            consumer_callback = next(
                (
                    callback
                    for callback in default_context.hooks.get("post_tool_call", [])
                    if _bound_method(callback, "Consumer") is not None
                ),
                None,
            )
            self.assertIsNotNone(consumer_callback)
            consumer = consumer_callback.__self__

            decision = {
                "label": "completion_candidate",
                "route_label": "completion_candidate",
                "probabilities": {
                    label: (1.0 if label == "completion_candidate" else 0.0)
                    for label in bridge.LABELS
                },
                "confidence": 0.91,
                "model": "jev-proof-offline",
                "usage": None,
            }
            consumer.decision_fn = lambda _event: decision
            self.assertTrue(consumer.process_pending())
            self.assertFalse(consumer.process_pending(), "the same event must not be consumed twice")

            reviews = store.reviews()
            ledger = [row for row in store.ledger() if row.get("event_key") == event["event_key"]]
            triggers = store.triggers()
            self.assertEqual(len(reviews), 1)
            self.assertEqual(reviews[0]["event_key"], event["event_key"])
            self.assertEqual(reviews[0]["label"], "completion_candidate")
            self.assertEqual(reviews[0]["state"], "triggered")
            self.assertEqual(len(ledger), 2, "one reservation and one final ledger state are expected")
            self.assertEqual(ledger[-1]["state"], "triggered")
            self.assertEqual(len(triggers), 1)
            trigger = triggers[0]
            self.assertEqual(trigger["task_id"], event["task_id"])
            self.assertEqual(str(trigger["run_id"]), str(event["run_id"]))
            self.assertEqual(trigger["checkpoint_id"], event["checkpoint_id"])
            self.assertEqual(trigger["state"], "pending_review")

            readback = {
                "task_id": event["task_id"],
                "run_id": event["run_id"],
                "original_request": event["original_request"],
                "completion_conditions": event["criteria"],
                "evidence_refs": event["evidence_refs"],
                "next_action": event["next_action"],
                "worker_conclusion": {
                    "status": "unmeasured",
                    "reason": "proof harness has no final worker conclusion",
                },
            }
            response = json.loads(
                default_context.tools["jev_bridge_review"](
                    args={
                        "trigger_id": trigger["trigger_id"],
                        "decision": "no_intervention",
                        "readback": readback,
                    }
                )
            )
            self.assertEqual(response["state"], "no_intervention")
            self.assertEqual(response["default_decision"]["decision"], "no_intervention")
            self.assertEqual(response["task_id"], event["task_id"])
            self.assertEqual(str(response["run_id"]), str(event["run_id"]))
            self.assertEqual(store.triggers()[-1]["state"], "no_intervention")
            self.assertEqual(store.controls()[-1]["control"], "release")

            persisted = "\n".join(
                path.read_text(encoding="utf-8")
                for path in root.glob("*.jsonl")
                if path.is_file()
            )
            self.assertNotIn(secret_sentinel, persisted)
            self.assertNotIn(secret_sentinel, json.dumps(event, ensure_ascii=False, sort_keys=True))

    def test_consumer_discards_finished_post_tool_threads(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jev-thread-cleanup-regression-") as temporary:
            root = Path(temporary)
            for should_fail in (False, True):
                started = threading.Event()
                release = threading.Event()
                consumer = bridge.Consumer(
                    FakeContext("default", root, "consumer"),
                    bridge.BridgeStore(root),
                )

                def bounded_process_pending(fails: bool = should_fail) -> None:
                    started.set()
                    if not release.wait(timeout=2.0):
                        raise AssertionError("thread cleanup probe was not released")
                    if fails:
                        raise RuntimeError("thread cleanup probe")

                consumer.process_pending = bounded_process_pending
                consumer.post_tool_call(tool_name="thread-cleanup-probe")
                self.assertTrue(started.wait(timeout=1.0))
                with consumer._lock:
                    threads = list(consumer._threads)
                self.assertEqual(len(threads), 1)
                thread = threads[0]

                if should_fail:
                    with self.assertLogs(bridge.log, level="WARNING"):
                        release.set()
                        thread.join(timeout=2.0)
                else:
                    release.set()
                    thread.join(timeout=2.0)
                self.assertFalse(thread.is_alive())
                with consumer._lock:
                    self.assertNotIn(thread, consumer._threads)
                    self.assertEqual(consumer._threads, set())

    def test_consumer_processes_more_than_daily_cap_across_distinct_runs_offline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jev-daily-cap-regression-") as temporary:
            root = Path(temporary)
            store = bridge.BridgeStore(root)
            task_id = "daily-cap-regression-task"
            run_ids = [f"run-{index}" for index in range(11)]
            for index, run_id in enumerate(run_ids):
                event = bridge.validate_checkpoint(
                    {
                        "schema_version": bridge.LEGACY_SCHEMA_VERSION,
                        "synthetic": True,
                        "category": bridge.LEGACY_CATEGORY,
                        "task_id": task_id,
                        "run_id": run_id,
                        "checkpoint_id": f"checkpoint-{index}",
                        "criteria": ["daily eligibility is bounded only by per-run safeguards"],
                        "observation": "offline quota regression event",
                        "evidence_refs": [f"temporary://event/{index}"],
                        "completed_tools": bridge.MIN_COMPLETED_TOOLS,
                        "changed_evidence_age_seconds": bridge.CHECKPOINT_CADENCE_SECONDS,
                        "original_request": "prove distinct run events are not blocked by a daily cap",
                        "current_unknown": "none",
                        "next_action": "record the offline decision",
                    }
                )
                event["state"] = "pending"
                store.append_event(event)

            decision_calls: list[tuple[str, str]] = []
            decision = {
                "label": "progressing",
                "route_label": "progressing",
                "probabilities": {
                    label: (1.0 if label == "progressing" else 0.0)
                    for label in bridge.LABELS
                },
                "confidence": 0.9,
                "model": "jev-regression-offline",
                "usage": None,
            }

            def offline_decision(event: dict[str, Any]) -> dict[str, Any]:
                decision_calls.append((str(event["task_id"]), str(event["run_id"])))
                return decision

            consumer = bridge.Consumer(
                FakeContext("default", root, "consumer"),
                store,
                decision_fn=offline_decision,
            )

            self.assertEqual(bridge.MAX_CALLS_PER_RUN, 3)
            self.assertEqual(bridge.MAX_LIVE_CALLS_PER_RUN, 3)
            for _ in run_ids:
                self.assertTrue(consumer.process_pending())
            self.assertFalse(consumer.process_pending())

            self.assertEqual(decision_calls, [(task_id, run_id) for run_id in run_ids])
            self.assertEqual(len(store.reviews()), len(run_ids))
            ledger = store.ledger()
            self.assertEqual(len(ledger), len(run_ids) * 2)
            self.assertEqual(
                {row["run_key"] for row in ledger},
                {f"{task_id}:{run_id}" for run_id in run_ids},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
