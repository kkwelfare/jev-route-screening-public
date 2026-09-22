import json
import math
import tempfile
import unittest
from pathlib import Path

from .point_state import (
    CONFIDENCE_THRESHOLD,
    POINT_STATES,
    PointIdentity,
    PointStatePolicy,
    build_point_state_request,
    derive_exclusive_state,
    parse_point_state_response,
)


class PointStateContractTests(unittest.TestCase):
    def test_request_uses_frozen_choice_contract_and_json_state(self):
        request = build_point_state_request({"goal": "g", "expected_next_action": "read"})
        self.assertEqual(set(request), {"model", "state", "questions"})
        self.assertTrue(request["model"])
        self.assertIsInstance(request["state"], str)
        question = request["questions"]["next_action_state"]
        self.assertEqual(set(question), {"type", "instructions", "criteria"})
        self.assertEqual(question["type"], "choice")
        self.assertIsInstance(question["criteria"], dict)
        self.assertEqual(set(question["criteria"]), set(POINT_STATES))
        self.assertIn("acceptance_ready", question["instructions"])
        self.assertIn("ordinary_action_ready", question["instructions"])

    def test_real_request_response_shape_is_parsed(self):
        request = build_point_state_request({"goal": "g", "scope": ["read"], "expected_next_action": "read"})
        self.assertIsInstance(request["questions"], dict)
        self.assertIn("next_action_state", request["questions"])
        response = {
            "model": request["model"],
            "answers": {"next_action_state": {"choice": "named_verification_ready", "confidence": 0.8}},
        }
        decision = parse_point_state_response(response)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.state, "named_verification_ready")

    def test_exclusive_precedence_first_match_wins(self):
        snapshot = {
            "scope_or_authorization_blocked": True,
            "input_error_unresolved": True,
            "evidence_missing": True,
            "expected_next_action": "read",
        }
        self.assertEqual(derive_exclusive_state(snapshot), "scope_or_authorization_blocked")
        self.assertEqual(
            derive_exclusive_state({"acceptance_ready": True, "ordinary_action_ready": True, "expected_next_action": "continue"}),
            "acceptance_ready",
        )

    def test_high_and_exact_threshold_are_accepted(self):
        for confidence in (0.91, CONFIDENCE_THRESHOLD):
            decision = parse_point_state_response({"state": "ordinary_action_ready", "confidence": confidence})
            self.assertTrue(decision.accepted)
            self.assertEqual(decision.confidence, confidence)

    def test_low_confidence_abstains(self):
        decision = parse_point_state_response({"state": "ordinary_action_ready", "confidence": 0.599})
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.issue, "low_confidence")

    def test_missing_nan_out_of_range_and_boolean_confidence_abstain(self):
        for confidence in (None, math.nan, -0.01, 1.01, True):
            decision = parse_point_state_response({"state": "ordinary_action_ready", "confidence": confidence})
            self.assertFalse(decision.accepted)

    def test_conflicting_and_unknown_states_abstain(self):
        conflicting = parse_point_state_response({"answers": [
            {"state": "ordinary_action_ready", "confidence": 0.9},
            {"state": "evidence_missing", "confidence": 0.9},
        ]})
        unknown = parse_point_state_response({"state": "unknown", "confidence": 0.9})
        self.assertEqual(conflicting.issue, "conflicting_state")
        self.assertEqual(unknown.issue, "unknown_state")
        self.assertFalse(conflicting.accepted)
        self.assertFalse(unknown.accepted)

    def test_malformed_provider_result_never_passes(self):
        decision = parse_point_state_response("not-json")
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.issue, "malformed_response")

    def test_normal_continue_is_bounded_and_guidance_only(self):
        snapshot = {"expected_next_action": "read the named manifest"}
        policy = PointStatePolicy()
        identity = PointIdentity.from_snapshot("task", "run", "1", snapshot)
        decision = parse_point_state_response({"state": "ordinary_action_ready", "confidence": 0.8})
        effect = policy.evaluate(decision, identity, issue="i", milestone="m", snapshot=snapshot)
        self.assertEqual(effect.action, "continue_scoped_action")
        self.assertTrue(effect.guidance_only)
        self.assertFalse(effect.completion_allowed)
        self.assertEqual(effect.instructions, ("read the named manifest",))

    def test_named_verification_and_acceptance_review(self):
        snapshot = {"next_named_verification": "pytest tests/test_contract.py"}
        policy = PointStatePolicy()
        verification = policy.evaluate(
            parse_point_state_response({"state": "named_verification_ready", "confidence": 0.8}),
            PointIdentity.from_snapshot("task", "run", "1", snapshot),
            issue="i", milestone="m1", snapshot=snapshot,
        )
        self.assertEqual(verification.action, "run_named_verification")
        self.assertEqual(verification.verification_name, "pytest tests/test_contract.py")

        acceptance_snapshot = {"expected_next_action": "review"}
        acceptance = policy.evaluate(
            parse_point_state_response({"state": "acceptance_ready", "confidence": 0.8}),
            PointIdentity.from_snapshot("task", "run", "2", acceptance_snapshot),
            issue="i", milestone="m2", snapshot=acceptance_snapshot,
        )
        self.assertEqual(acceptance.action, "review")
        self.assertEqual(acceptance.reason, "default_acceptance_review")
        self.assertFalse(acceptance.completion_allowed)

    def test_first_unrepaired_error_then_issue_budget_across_events(self):
        snapshot = {"correction_instruction": "revalidate actual arguments"}
        store = {}
        policy = PointStatePolicy(store=store)
        decision = parse_point_state_response({"state": "input_error_unresolved", "confidence": 0.8})
        first = policy.evaluate(decision, PointIdentity.from_snapshot("task", "run", "1", snapshot), issue="i", milestone="m", snapshot=snapshot)
        second_snapshot = dict(snapshot, latest_bounded_result="still failing")
        second = policy.evaluate(decision, PointIdentity.from_snapshot("task", "run", "2", second_snapshot), issue="i", milestone="m", snapshot=second_snapshot)
        self.assertEqual(first.action, "correct_input")
        self.assertEqual(second.action, "review")
        self.assertEqual(second.reason, "correction_budget_exhausted")

    def test_refresh_budget_is_not_reset_by_new_event(self):
        store = {}
        policy = PointStatePolicy(store=store)
        decision = parse_point_state_response({"state": "evidence_missing", "confidence": 0.8})
        first_snapshot = {"refs": ["a"]}
        first = policy.evaluate(decision, PointIdentity.from_snapshot("task", "run", "1", first_snapshot), issue="i", milestone="m", snapshot=first_snapshot)
        second_snapshot = {"refs": ["b"]}
        second = policy.evaluate(decision, PointIdentity.from_snapshot("task", "run", "2", second_snapshot), issue="i", milestone="m", snapshot=second_snapshot)
        self.assertEqual(first.action, "refresh_evidence_then_review")
        self.assertEqual(second.action, "review")
        self.assertEqual(second.reason, "refresh_budget_exhausted")

    def test_duplicate_action_dedup_is_snapshot_bound(self):
        snapshot = {"expected_next_action": "read"}
        policy = PointStatePolicy(store={})
        decision = parse_point_state_response({"state": "ordinary_action_ready", "confidence": 0.8})
        identity1 = PointIdentity.from_snapshot("task", "run", "1", snapshot)
        identity2 = PointIdentity.from_snapshot("task", "run", "2", snapshot)
        first = policy.evaluate(decision, identity1, issue="i", milestone="m", snapshot=snapshot)
        duplicate = policy.evaluate(decision, identity2, issue="i", milestone="m", snapshot=snapshot)
        self.assertEqual(first.action, "continue_scoped_action")
        self.assertEqual(duplicate.action, "deduplicated")

    def test_stale_event_and_contract_are_rejected_without_old_action(self):
        snapshot = {"expected_next_action": "read"}
        policy = PointStatePolicy(store={})
        decision = parse_point_state_response({"state": "ordinary_action_ready", "confidence": 0.8})
        newest = policy.evaluate(decision, PointIdentity.from_snapshot("task", "run", "3", snapshot), issue="i", milestone="m", snapshot=snapshot)
        stale = policy.evaluate(decision, PointIdentity.from_snapshot("task", "run", "2", {"expected_next_action": "write"}), issue="i", milestone="m", snapshot={"expected_next_action": "write"})
        stale_contract = policy.evaluate(decision, PointIdentity("task", "run", "4", "old-contract", "hash"), issue="i", milestone="m", snapshot=snapshot)
        self.assertEqual(newest.action, "continue_scoped_action")
        self.assertEqual(stale.action, "reject_stale")
        self.assertEqual(stale_contract.reason, "stale_contract")

    def test_guards_always_prevail(self):
        snapshot = {"expected_next_action": "read"}
        policy = PointStatePolicy(store={})
        effect = policy.evaluate(
            parse_point_state_response({"state": "ordinary_action_ready", "confidence": 0.99}),
            PointIdentity.from_snapshot("task", "run", "1", snapshot),
            issue="i", milestone="m", snapshot=snapshot,
            guards=("scope_guard",),
        )
        self.assertEqual(effect.action, "review")
        self.assertEqual(effect.reason, "guard_precedence")

    def test_untrusted_state_gets_one_bounded_refresh_then_review(self):
        policy = PointStatePolicy(store={})
        snapshot = {"refs": ["named-evidence"]}
        identity = PointIdentity.from_snapshot("task", "run", "1", snapshot)
        first = policy.evaluate(parse_point_state_response({"state": "ordinary_action_ready", "confidence": 0.2}), identity, issue="i", milestone="m", snapshot=snapshot)
        second = policy.evaluate(parse_point_state_response({"state": "ordinary_action_ready", "confidence": 0.2}), PointIdentity.from_snapshot("task", "run", "2", dict(snapshot, latest_bounded_result="x")), issue="i", milestone="m", snapshot=dict(snapshot, latest_bounded_result="x"))
        self.assertEqual(first.action, "refresh_evidence_then_review")
        self.assertEqual(second.action, "review")

    def test_store_path_round_trip_preserves_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            snapshot = {"expected_next_action": "read"}
            decision = parse_point_state_response({"state": "ordinary_action_ready", "confidence": 0.8})
            first = PointStatePolicy(store_path=path).evaluate(decision, PointIdentity.from_snapshot("task", "run", "1", snapshot), issue="i", milestone="m", snapshot=snapshot)
            second = PointStatePolicy(store_path=path).evaluate(decision, PointIdentity.from_snapshot("task", "run", "2", snapshot), issue="i", milestone="m", snapshot=snapshot)
            self.assertEqual(first.action, "continue_scoped_action")
            self.assertEqual(second.action, "deduplicated")
            self.assertIn("actions", json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
