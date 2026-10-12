"""Offline tests for the aggregate-only Jev news-sufficiency advisory."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "jev_sufficiency_test_pkg",
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
assert spec is not None and spec.loader is not None
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
bridge = sys.modules[spec.name + ".bridge"]

VALID = {
    "snapshot_version": bridge.NEWS_DOMAIN_EXTENSION,
    "broad_search_calls_used": 1,
    "broad_search_call_cap": 3,
    "total_verified_eligible_candidate_count": 2,
    "verified_counts_by_broad_category": {"ai": 1, "business": 1},
    "required_non_ai_communication_count": 1,
    "required_non_ai_communication_condition_met": True,
    "unresolved_freshness_check_count": 0,
    "unresolved_source_check_count": 0,
    "unresolved_dedup_check_count": 0,
    "coverage_status": "complete",
    "consecutive_no_gain_count": 0,
}


def provider_response(choice: str, confidence: float = 0.95, **extra):
    return {
        "model": "test-provider-model",
        "answers": {
            "news_sufficiency": {
                "choice": choice,
                "probabilities": {
                    name: (1.0 if name == choice else 0.0)
                    for name in bridge.SUFFICIENCY_CHOICES
                },
                "confidence": confidence,
                **extra,
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


def reset_sufficiency_live_call_guard():
    with bridge._SUFFICIENCY_LIVE_CALL_LOCK:
        bridge._sufficiency_live_call_used = False


class SufficiencyToolTests(unittest.TestCase):
    def setUp(self):
        reset_sufficiency_live_call_guard()

    def test_input_allowlist_limits_and_partition_integrity(self):
        clean = bridge.validate_sufficiency_snapshot(VALID)
        self.assertEqual(set(clean), set(bridge.SUFFICIENCY_SNAPSHOT_FIELDS))
        self.assertEqual(clean["verified_counts_by_broad_category"], {"ai": 1, "business": 1})
        overlapping = {**VALID, "verified_counts_by_broad_category": {"ai": 2, "business": 2}}
        self.assertEqual(bridge.validate_sufficiency_snapshot(overlapping)["total_verified_eligible_candidate_count"], 2)

        invalid_limits = (
            {**VALID, "total_verified_eligible_candidate_count": bridge.MAX_SUFFICIENCY_COUNT + 1},
            {**VALID, "broad_search_calls_used": 4},
            {**VALID, "broad_search_call_cap": 4},
            {**VALID, "verified_counts_by_broad_category": {"x" * 49: 1}},
        )
        for snapshot in invalid_limits:
            with self.subTest(snapshot=snapshot):
                with self.assertRaises(bridge.BridgeValidationError):
                    bridge.validate_sufficiency_snapshot(snapshot)

        original_byte_limit = bridge.MAX_SUFFICIENCY_INPUT_BYTES
        try:
            bridge.MAX_SUFFICIENCY_INPUT_BYTES = len(json.dumps(VALID, separators=(",", ":")).encode("utf-8")) - 1
            with self.assertRaises(bridge.BridgeValidationError):
                bridge.validate_sufficiency_snapshot(VALID)
        finally:
            bridge.MAX_SUFFICIENCY_INPUT_BYTES = original_byte_limit

        for change in (
            {"candidate_title": "private title"},
            {"candidate_urls": ["https://example.invalid/" ]},
            {"personal_interest_text": "private"},
            {"broad_search_calls_used": 4},
            {"broad_search_call_cap": 4},
            {"verified_counts_by_broad_category": {"ai": 3}},
            {"verified_counts_by_broad_category": {"private interest text": 2}},
            {"coverage_status": "probably_complete"},
            {"consecutive_no_gain_count": 3},
        ):
            with self.subTest(change=change):
                with self.assertRaises(bridge.BridgeValidationError):
                    bridge.validate_sufficiency_snapshot({**VALID, **change})

        oversize = {**VALID, "verified_counts_by_broad_category": {f"category_{i}": 1 for i in range(25)}}
        oversize["total_verified_eligible_candidate_count"] = 25
        oversize["required_non_ai_communication_count"] = 1
        with self.assertRaises(bridge.BridgeValidationError):
            bridge.validate_sufficiency_snapshot(oversize)

    def test_provider_state_contains_only_approved_aggregate_fields(self):
        request = bridge.build_jev_sufficiency_request(VALID)
        self.assertEqual(set(request), {"model", "state", "questions"})
        state = json.loads(request["state"])
        self.assertEqual(set(state), set(bridge.SUFFICIENCY_SNAPSHOT_FIELDS))
        self.assertEqual(set(request["questions"]), {"news_sufficiency"})
        self.assertIn("deterministic", request["questions"]["news_sufficiency"]["instructions"])
        self.assertNotIn("title", request["state"])
        self.assertNotIn("url", request["state"])
        self.assertNotIn("private personal interest text", request["state"])
        self.assertNotIn("candidate title", request["state"])

    def test_provider_success_is_reduced_to_two_advisory_fields(self):
        sender = Sender(provider_response("snapshot_sufficient", leaked_reason="private article title"))
        result = json.loads(bridge.jev_sufficiency_tool(VALID, ctx=Context(), post_fn=sender))
        self.assertEqual(result, {"status": "snapshot_sufficient", "reason_code": "aggregate_consistent"})
        self.assertEqual(set(result), {"status", "reason_code"})
        self.assertEqual([call[0].name for call in sender.calls], ["typesafe"])
        request_state = json.loads(sender.calls[0][1]["state"])
        self.assertEqual(set(request_state), set(bridge.SUFFICIENCY_SNAPSHOT_FIELDS))

    def test_only_one_live_evaluation_is_allowed_per_process(self):
        first_sender = Sender(provider_response("snapshot_sufficient"))
        first = json.loads(bridge.jev_sufficiency_tool(VALID, ctx=Context(), post_fn=first_sender))
        second_sender = Sender(provider_response("snapshot_sufficient"))
        second = json.loads(bridge.jev_sufficiency_tool(VALID, ctx=Context(), post_fn=second_sender))
        invalid_sender = Sender(provider_response("snapshot_sufficient"))
        after_checked_invalid = json.loads(bridge.jev_sufficiency_tool({"unexpected": "field"}, ctx=Context(), post_fn=invalid_sender))

        self.assertEqual(first, {"status": "snapshot_sufficient", "reason_code": "aggregate_consistent"})
        self.assertEqual(len(first_sender.calls), 1)
        self.assertEqual(second, {"status": "more_evidence_needed", "reason_code": "already_checked"})
        self.assertEqual(second_sender.calls, [])
        self.assertEqual(after_checked_invalid, {"status": "more_evidence_needed", "reason_code": "already_checked"})
        self.assertEqual(invalid_sender.calls, [])

    def test_primary_timeout_uses_existing_openrouter_fallback(self):
        sender = Sender(
            bridge.JevRequestError("timeout"),
            provider_response("more_evidence_aggregate_needs_review"),
        )
        result = json.loads(bridge.jev_sufficiency_tool(VALID, ctx=Context(), post_fn=sender))
        self.assertEqual(result, {"status": "more_evidence_needed", "reason_code": "aggregate_needs_review"})
        self.assertEqual([call[0].name for call in sender.calls], ["typesafe", "openrouter"])
        self.assertEqual(sender.calls[1][1]["model"], bridge.FALLBACK_PROVIDER.model)
        for _provider, body, _timeout in sender.calls:
            self.assertEqual(set(json.loads(body["state"])), set(bridge.SUFFICIENCY_SNAPSHOT_FIELDS))

    def test_provider_unavailable_or_ambiguous_never_confirms_sufficiency(self):
        unavailable = Sender(
            bridge.JevRequestError("timeout"),
            bridge.JevRequestError("credential_unavailable"),
        )
        result = json.loads(bridge.jev_sufficiency_tool(VALID, ctx=Context(), post_fn=unavailable))
        self.assertEqual(result, {"status": "more_evidence_needed", "reason_code": "provider_unavailable"})

        # Each provider outcome below models a separate plugin process/run.
        reset_sufficiency_live_call_guard()
        ambiguous = Sender(provider_response("snapshot_sufficient", confidence=0.5))
        result = json.loads(bridge.jev_sufficiency_tool(VALID, ctx=Context(), post_fn=ambiguous))
        self.assertEqual(result, {"status": "more_evidence_needed", "reason_code": "provider_ambiguous"})

        reset_sufficiency_live_call_guard()
        malformed = Sender({"answers": {"news_sufficiency": {"choice": "snapshot_sufficient"}}})
        result = json.loads(bridge.jev_sufficiency_tool(VALID, ctx=Context(), post_fn=malformed))
        self.assertEqual(result, {"status": "more_evidence_needed", "reason_code": "provider_ambiguous"})

        reset_sufficiency_live_call_guard()
        unsupported_distribution = {
            name: 0.0 for name in bridge.SUFFICIENCY_CHOICES
        }
        inconsistent = Sender(provider_response("snapshot_sufficient", probabilities=unsupported_distribution))
        result = json.loads(bridge.jev_sufficiency_tool(VALID, ctx=Context(), post_fn=inconsistent))
        self.assertEqual(result, {"status": "more_evidence_needed", "reason_code": "provider_ambiguous"})

        reset_sufficiency_live_call_guard()
        weak_selected_choice = {
            name: (0.2 if name == "snapshot_sufficient" else 0.8 if name == "more_evidence_aggregate_needs_review" else 0.0)
            for name in bridge.SUFFICIENCY_CHOICES
        }
        contradictory = Sender(provider_response("snapshot_sufficient", confidence=0.99, probabilities=weak_selected_choice))
        result = json.loads(bridge.jev_sufficiency_tool(VALID, ctx=Context(), post_fn=contradictory))
        self.assertEqual(result, {"status": "more_evidence_needed", "reason_code": "provider_ambiguous"})

    def test_deterministic_safeguards_short_circuit_before_provider(self):
        cases = (
            ({"coverage_status": "incomplete"}, "coverage_incomplete"),
            ({"unresolved_source_check_count": 1}, "unresolved_checks"),
            ({"required_non_ai_communication_condition_met": False}, "required_mix_missing"),
        )
        for change, reason in cases:
            with self.subTest(reason=reason):
                sender = Sender(provider_response("snapshot_sufficient"))
                result = json.loads(bridge.jev_sufficiency_tool({**VALID, **change}, ctx=Context(), post_fn=sender))
                self.assertEqual(result, {"status": "more_evidence_needed", "reason_code": reason})
                self.assertEqual(sender.calls, [])

        empty_not_exhausted = {
            **VALID,
            "broad_search_calls_used": 1,
            "total_verified_eligible_candidate_count": 0,
            "verified_counts_by_broad_category": {},
            "required_non_ai_communication_count": 1,
            "consecutive_no_gain_count": 0,
        }
        sender = Sender(provider_response("snapshot_sufficient"))
        result = json.loads(bridge.jev_sufficiency_tool(empty_not_exhausted, ctx=Context(), post_fn=sender))
        self.assertEqual(result, {"status": "more_evidence_needed", "reason_code": "search_not_exhausted"})
        self.assertEqual(sender.calls, [])

        exhausted_empty_snapshot = {
            **empty_not_exhausted,
            "broad_search_calls_used": 2,
            "consecutive_no_gain_count": 2,
        }
        sender = Sender(provider_response("snapshot_sufficient"))
        result = json.loads(bridge.jev_sufficiency_tool(exhausted_empty_snapshot, ctx=Context(), post_fn=sender))
        self.assertEqual(result, {"status": "snapshot_sufficient", "reason_code": "aggregate_consistent"})
        self.assertEqual(len(sender.calls), 1)

    def test_invalid_input_returns_restricted_result_without_sending(self):
        sender = Sender(provider_response("snapshot_sufficient"))
        result = json.loads(bridge.jev_sufficiency_tool({**VALID, "candidate_title": "private"}, ctx=Context(), post_fn=sender))
        self.assertEqual(result, {"status": "invalid_snapshot", "reason_code": "invalid_input"})
        self.assertEqual(set(result), {"status", "reason_code"})
        self.assertEqual(sender.calls, [])

        valid_sender = Sender(provider_response("snapshot_sufficient"))
        valid_result = json.loads(bridge.jev_sufficiency_tool(VALID, ctx=Context(), post_fn=valid_sender))
        self.assertEqual(valid_result, {"status": "snapshot_sufficient", "reason_code": "aggregate_consistent"})
        self.assertEqual(len(valid_sender.calls), 1)

    def test_tool_is_registered_only_for_default_consumer(self):
        with tempfile.TemporaryDirectory(prefix="jev-sufficiency-register-") as directory:
            default = Context(directory, role="consumer", profiles=["default"])
            bridge.register(default)
            self.assertIn("jev_sufficiency", default.tools)
            self.assertEqual(default.tools["jev_sufficiency"]["toolset"], "jev_sufficiency")
            self.assertEqual(default.tools["jev_bridge_review"]["toolset"], "jev_route_screening")
            schema = default.tools["jev_sufficiency"]["schema"]
            self.assertEqual(schema["name"], "jev_sufficiency")
            self.assertFalse(schema["parameters"]["additionalProperties"])
            self.assertEqual(set(schema["parameters"]["required"]), set(bridge.SUFFICIENCY_SNAPSHOT_FIELDS))

        with tempfile.TemporaryDirectory(prefix="jev-sufficiency-producer-") as directory:
            ops = Context(directory, role="producer", profiles=["ops"])
            ops.profile_name = "ops"
            bridge.register(ops)
            self.assertNotIn("jev_sufficiency", ops.tools)
            self.assertEqual(ops.tools["jev_bridge_checkpoint"]["toolset"], "jev_route_screening")


if __name__ == "__main__":
    unittest.main()
