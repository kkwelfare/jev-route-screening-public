"""Focused offline tests for Jev news search-advice token routing."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "jev_search_advice_test_pkg",
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
assert spec is not None and spec.loader is not None
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
bridge = sys.modules[spec.name + ".bridge"]

VALID = {
    "snapshot_version": bridge.NEWS_SEARCH_ADVICE_EXTENSION,
    "candidates": [
        {"candidate_id": "c1", "entity_terms": ["OpenAI"], "number_terms": ["$500"], "name_terms": ["Pro Max"]},
        {"candidate_id": "c2", "entity_terms": ["Anthropic"], "number_terms": ["$4"], "name_terms": ["Opus 5.5"]},
    ],
}


def response(choice: str, ids=("c1", "c2"), confidence: float = 0.95):
    choices = {f"candidate_{candidate_id}" for candidate_id in ids} | {"no_preference"}
    selected = choice if choice in choices else "candidate_c1"
    return {
        "model": "test-provider-model",
        "answers": {
            "news_search_priority": {
                "choice": selected,
                "probabilities": {name: (1.0 if name == selected else 0.0) for name in choices},
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 1, "output_tokens": 1, "cost": "0"},
    }


class Sender:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, provider, body, timeout):
        self.calls.append((provider, dict(body), timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class Context:
    profile_name = "default"

    def __init__(self, root=None, *, role="consumer", profiles=None):
        self.root = str(root or "")
        self.role = role
        self.profiles = profiles or ["default"]
        self.tools = {}
        self.hooks = {}

    def get_config(self, key, default=None):
        return {
            "bridge_dir": self.root,
            "role": self.role,
            "producer_profiles": ["ops"],
            "consumer_profiles": self.profiles,
        }.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs

    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)


def reset_call_guard():
    with bridge._SEARCH_ADVICE_LIVE_CALL_LOCK:
        bridge._search_advice_live_call_used = False


class SearchAdviceToolTests(unittest.TestCase):
    def setUp(self):
        reset_call_guard()

    def test_input_and_provider_state_only_contain_exact_terms(self):
        clean = bridge.validate_search_advice_candidates(VALID)
        self.assertEqual(clean["candidates"], VALID["candidates"])
        request = bridge.build_jev_search_advice_request(VALID)
        state = json.loads(request["state"])
        self.assertEqual(set(state), {"snapshot_version", "candidates"})
        self.assertEqual(set(state["candidates"][0]), {"candidate_id", "entity_terms", "number_terms", "name_terms", "exact_query"})
        self.assertEqual(state["candidates"][0]["exact_query"], '"OpenAI" "$500" "Pro Max"')
        self.assertNotIn("https://", request["state"])
        self.assertNotIn("post_text", request["state"])
        self.assertNotIn("author_handle", request["state"])

    def test_unknown_fields_urls_handles_duplicates_and_unbounded_batches_reject(self):
        invalid = (
            {**VALID, "post_text": "full post"},
            {**VALID, "candidates": [{**VALID["candidates"][0], "url": "https://x.com/a/status/1"}]},
            {**VALID, "candidates": [{**VALID["candidates"][0], "entity_terms": ["@account"]}]},
            {**VALID, "candidates": [{**VALID["candidates"][0], "number_terms": ["$500", "$500"]}]},
            {**VALID, "candidates": [*VALID["candidates"]] * 5},
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(bridge.BridgeValidationError):
                    bridge.validate_search_advice_candidates(value)

    def test_jev_choice_maps_back_to_locally_composed_exact_query(self):
        sender = Sender(response("candidate_c2"))
        result = json.loads(bridge.jev_search_advice_tool(VALID, ctx=Context(), post_fn=sender))
        self.assertEqual(result["status"], "query_advice_ready")
        self.assertEqual(result["selection_source"], "jev_advisory")
        self.assertEqual(result["candidate_id"], "c2")
        self.assertEqual(result["search_query"], '"Anthropic" "$4" "Opus 5.5"')
        self.assertEqual(result["exact_fallback_query"], '"OpenAI" "$500" "Pro Max"')
        self.assertEqual(len(sender.calls), 1)

    def test_ambiguous_provider_result_falls_back_without_dropping_lead(self):
        sender = Sender({"answers": {"unexpected": {}}})
        result = json.loads(bridge.jev_search_advice_tool(VALID, ctx=Context(), post_fn=sender))
        self.assertEqual(result["selection_source"], "local_fallback")
        self.assertEqual(result["candidate_id"], "c1")
        self.assertIn('"$500"', result["search_query"])
        self.assertEqual(len(sender.calls), 1)

    def test_only_one_live_provider_evaluation_per_process(self):
        first_sender = Sender(response("candidate_c2"))
        bridge.jev_search_advice_tool(VALID, ctx=Context(), post_fn=first_sender)
        second_sender = Sender(response("candidate_c2"))
        second = json.loads(bridge.jev_search_advice_tool(VALID, ctx=Context(), post_fn=second_sender))
        self.assertEqual(len(first_sender.calls), 1)
        self.assertEqual(second_sender.calls, [])
        self.assertEqual(second["selection_source"], "local_fallback")
        self.assertIn('"$500"', second["exact_fallback_query"])

    def test_tool_is_registered_only_for_default_consumer(self):
        with tempfile.TemporaryDirectory(prefix="jev-search-advice-register-") as directory:
            default = Context(directory, role="consumer", profiles=["default"])
            bridge.register(default)
            self.assertIn("jev_search_advice", default.tools)
            self.assertEqual(default.tools["jev_search_advice"]["toolset"], "jev_search_advice")
            schema = default.tools["jev_search_advice"]["schema"]
            self.assertEqual(schema["name"], "jev_search_advice")
            self.assertFalse(schema["parameters"]["additionalProperties"])

        with tempfile.TemporaryDirectory(prefix="jev-search-advice-producer-") as directory:
            ops = Context(directory, role="producer", profiles=["ops"])
            ops.profile_name = "ops"
            bridge.register(ops)
            self.assertNotIn("jev_search_advice", ops.tools)


if __name__ == "__main__":
    unittest.main()
