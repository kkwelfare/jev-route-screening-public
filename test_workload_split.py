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


def response(many, runtime, divisible, synthesis=0.1, quantitative=0.1):
    answers = {
        "many_items": {"type": "noul", "noul": many},
        "exceeds_runtime": {"type": "noul", "noul": runtime},
        "divisible": {"type": "noul", "noul": divisible},
    }
    if synthesis is not None:
        answers["cross_source_synthesis"] = {"type": "noul", "noul": synthesis}
    if quantitative is not None:
        answers["quantitative_analysis"] = {"type": "noul", "noul": quantitative}
    return {"answers": answers}


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
        self.assertEqual((over["signal"], over["route"]), ("recommend_split", "count_rule"))
        self.assertEqual(over["model_signal"], "not_judged")
        self.assertEqual(sender.calls, [])
        self.assertEqual(len(self.log_rows()), 1)

    def test_count_within_threshold_keeps_rule_and_asks_model_judgment(self):
        # Jev says large and divisible, but the stated count settles the split signal.
        sender = Sender(response(0.9, 0.9, 0.1, synthesis=0.9))
        within = self.run_tool({"request_summary": "Compare two laws", "target_count": 2, "executor": "kanban_research"}, sender)
        self.assertEqual((within["signal"], within["reason_code"], within["route"]), ("proceed", "count_within_threshold", "count_rule"))
        self.assertEqual(within["model_signal"], "consider_stronger_model")
        self.assertEqual(len(sender.calls), 1)
        failed = self.run_tool({"request_summary": "Fix two skills", "target_count": 3, "executor": "delegation"},
                               Sender(bridge.JevRequestError("http_error", 400)))
        self.assertEqual((failed["signal"], failed["reason_code"], failed["route"]), ("proceed", "count_within_threshold", "count_rule"))
        self.assertEqual(failed["model_signal"], "not_judged")

    def test_model_signal_truth_table(self):
        cases = [
            # (divisible, synthesis, quantitative) -> model_signal, reason
            ((0.1, 0.9, 0.1), "consider_stronger_model", "jev_cross_source_synthesis"),
            ((0.1, 0.1, 0.9), "consider_stronger_model", "jev_quantitative_analysis"),
            ((0.5, 0.9, 0.9), "consider_stronger_model", "jev_synthesis_and_quantitative"),
            ((0.9, 0.9, 0.9), "none", "jev_divisible"),
            ((0.1, 0.5, 0.6), "none", "jev_no_hard_judgment"),
            ((0.1, 0.1, 0.1), "none", "jev_no_hard_judgment"),
        ]
        for (divisible, synthesis, quantitative), expected, reason in cases:
            with self.subTest(divisible=divisible, synthesis=synthesis, quantitative=quantitative):
                result = self.run_tool({"request_summary": "Assess a policy", "executor": "kanban_research"},
                                       Sender(response(0.1, 0.1, divisible, synthesis, quantitative)))
                self.assertEqual((result["model_signal"], result["model_reason_code"]), (expected, reason))
                self.assertEqual(result["signal"], "proceed")

    def test_legacy_three_answer_response_keeps_split_signal(self):
        result = self.run_tool({"request_summary": "Big list", "executor": "kanban_research"},
                               Sender(response(0.9, 0.9, 0.9, synthesis=None, quantitative=None)))
        self.assertEqual(result["signal"], "recommend_split")
        self.assertEqual((result["model_signal"], result["model_reason_code"]), ("not_judged", "jev_model_judgment_missing"))

    def test_model_signal_is_logged(self):
        self.run_tool({"request_summary": "Assess a policy", "executor": "kanban_research", "label": "t_x"},
                      Sender(response(0.1, 0.1, 0.1, 0.9, 0.1)))
        self.assertEqual(self.log_rows()[0]["model_signal"], "consider_stronger_model")

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

    def test_request_shape_has_five_nouls_and_no_ids(self):
        sender = Sender(response(0.1, 0.1, 0.1))
        self.run_tool({"request_summary": "Summarize one page", "executor": "kanban_research", "label": "t_x", "runtime_limit_minutes": 30}, sender)
        body = sender.calls[0][1]
        self.assertEqual(set(body["questions"]), {"many_items", "exceeds_runtime", "divisible", "cross_source_synthesis", "quantitative_analysis"})
        self.assertTrue(all(q["type"] == "noul" for q in body["questions"].values()))
        self.assertEqual(body["state"], {"request": {"summary": "Summarize one page", "runtime_limit_minutes": 30}})

    def test_provider_failure_and_malformed_fail_open(self):
        failure = self.run_tool({"request_summary": "x", "executor": "other"},
                                Sender(bridge.JevRequestError("http_error", 400)))
        self.assertEqual((failure["signal"], failure["reason_code"]), ("proceed", "jev_unavailable"))
        self.assertEqual(failure["model_signal"], "not_judged")
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
