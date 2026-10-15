"""Regression: the default pre_tool_call pending probe must use the ledger.

events.jsonl rows keep ``state == "pending"`` forever; completion lives in the
ledger.  The probe must report no work when every pending event already has an
attempt row, so ordinary default tool calls do not run process_pending.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from test_worker_consumer_proof import FakeContext, bridge  # noqa: F401


class PendingProbeTests(unittest.TestCase):
    def _store(self, root: Path) -> "bridge.BridgeStore":
        return bridge.BridgeStore(root)

    def _write(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def test_probe_ignores_events_already_in_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            events = [{"event_key": f"t_x:1:cp-{i}", "task_id": "t_x", "run_id": "1", "state": "pending"} for i in range(3)]
            self._write(store.events_path, events)
            self._write(store.ledger_path, [{"event_key": e["event_key"], "run_key": "t_x:1", "state": "completed"} for e in events])
            consumer = bridge.Consumer(FakeContext("default", root, "consumer"), store)
            self.assertFalse(consumer.has_unprocessed_pending())
            self.assertFalse(consumer.process_pending())

    def test_probe_reports_event_without_attempt_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            events = [{"event_key": f"t_x:1:cp-{i}", "task_id": "t_x", "run_id": "1", "state": "pending"} for i in range(2)]
            self._write(store.events_path, events)
            self._write(store.ledger_path, [{"event_key": events[0]["event_key"], "run_key": "t_x:1", "state": "completed"}])
            consumer = bridge.Consumer(FakeContext("default", root, "consumer"), store)
            self.assertTrue(consumer.has_unprocessed_pending())

    def test_scope_guard_skips_process_pending_when_probe_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self._store(root)
            self._write(store.events_path, [{"event_key": "t_x:1:cp-0", "task_id": "t_x", "run_id": "1", "state": "pending"}])
            calls: list[int] = []
            guard = bridge.DefaultScopeGuard(
                FakeContext("default", root, "consumer"), store,
                process_pending=lambda: calls.append(1),
                has_pending=lambda: False,
            )
            guard.pre_tool_call(tool_name="read_file", args={"path": "/tmp/x"}, session_id="", turn_id="")
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
