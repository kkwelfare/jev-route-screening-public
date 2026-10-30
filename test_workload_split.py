"""Offline tests for the advisory jev_workload_split tool."""
from __future__ import annotations

import importlib.util
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "jev_workload_split_test_pkg", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
)
assert spec is not None and spec.loader is not None
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
bridge = sys.modules[spec.name + ".bridge"]
ws = importlib.import_module(spec.name + ".workload_split")


class Context:
    profile_name = "default"

    def __init__(self, root=""):
        self.root = str(root)
        self.tools = {}
        self.hooks = {}

    def get_config(self, key, default=None):
        return {"bridge_dir": self.root, "role": "consumer", "consumer_profiles": ["default"]}.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs

    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)


def response(many, runtime, divisible):
    return {"answers": {
        "many_items": {"type": "noul", "noul": many},
        "exceeds_runtime": {"type": "noul", "noul": runtime},
        "divisible": {"type": "noul", "noul": divisible},
    }}


class Sender:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, provider, body, timeout):
        self.calls.append((provider, dict(body)))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class WorkloadSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ctx = Context(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, args, sender=None):
        return json.loads(ws.workload_split_tool(args, ctx=self.ctx, post_fn=sender, log_root=self.root))

    def log_rows(self):
        path = self.root / ws.LOG_NAME
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_count_rule_skips_jev(self):
        sender = Sender()
        over = self.run_tool({"request_summary": "Fix five skills", "target_count": 5, "executor": "delegation"}, sender)
        within = self.run_tool({"request_summary": "Fix two skills", "target_count": 3, "executor": "delegation"}, sender)
        self.assertEqual((over["signal"], over["route"]), ("recommend_split", "count_rule"))
        self.assertEqual(within["signal"], "proceed")
        self.assertEqual(sender.calls, [])
        self.assertEqual(len(self.log_rows()), 2)

    def test_jev_route_truth_table(self):
        cases = [
            ((0.9, 0.2, 0.9), "recommend_split"),
            ((0.2, 0.8, 0.1), "consider_runtime"),
            ((0.1, 0.1, 0.9), "proceed"),
            ((0.5, 0.5, 0.9), "proceed"),
            ((0.9, 0.9, 0.5), "proceed"),
        ]
        for nouls, expected in cases:
            with self.subTest(nouls=nouls):
                result = self.run_tool({"request_summary": "List support groups around Arakawa", "executor": "kanban_research"}, Sender(response(*nouls)))
                self.assertEqual(result["signal"], expected)
                self.assertEqual(result["route"], "jev")

    def test_request_shape_has_three_nouls_and_no_ids(self):
        sender = Sender(response(0.1, 0.1, 0.1))
        self.run_tool({"request_summary": "Summarize one page", "executor": "kanban_research", "label": "t_x", "runtime_limit_minutes": 30}, sender)
        body = sender.calls[0][1]
        self.assertEqual(set(body["questions"]), {"many_items", "exceeds_runtime", "divisible"})
        self.assertTrue(all(q["type"] == "noul" for q in body["questions"].values()))
        self.assertEqual(body["state"], {"request": {"summary": "Summarize one page", "runtime_limit_minutes": 30}})

    def test_provider_failure_and_malformed_fail_open(self):
        failure = self.run_tool({"request_summary": "x", "executor": "other"},
                                Sender(bridge.JevRequestError("http_error", 400)))
        self.assertEqual((failure["signal"], failure["reason_code"]), ("proceed", "jev_unavailable"))
        malformed = self.run_tool({"request_summary": "x", "executor": "other"}, Sender({"answers": {}}))
        self.assertEqual(malformed["signal"], "proceed")

    def test_fallback_on_timeout(self):
        sender = Sender(bridge.JevRequestError("timeout"), response(0.9, 0.9, 0.9))
        result = self.run_tool({"request_summary": "Big list", "executor": "kanban_research"}, sender)
        self.assertEqual(result["signal"], "recommend_split")
        self.assertEqual(len(sender.calls), 2)

    def test_invalid_input(self):
        for args in ({}, {"request_summary": "x", "executor": "bad"}, {"request_summary": "x", "executor": "other", "target_count": 0},
                     {"request_summary": "x", "executor": "other", "extra": 1}):
            with self.subTest(args=args):
                self.assertEqual(self.run_tool(args)["signal"], "invalid_input")

    def test_log_has_no_summary_text(self):
        self.run_tool({"request_summary": "secret-ish summary text", "target_count": 9, "executor": "delegation"})
        row = self.log_rows()[0]
        self.assertNotIn("secret-ish", json.dumps(row))
        self.assertIn("summary_sha256", row)

    def test_registered_for_default_consumer(self):
        ctx = Context(self.root)
        plugin.register(ctx)
        self.assertIn("jev_workload_split", ctx.tools)


if __name__ == "__main__":
    unittest.main()
