from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "jev_route_screening_targeted_reading_test"

spec = importlib.util.spec_from_file_location(
    PACKAGE_NAME,
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
assert spec is not None and spec.loader is not None
package = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE_NAME] = package
spec.loader.exec_module(package)
router = __import__(f"{PACKAGE_NAME}.skill_router", fromlist=["*"])
bridge = __import__(f"{PACKAGE_NAME}.bridge", fromlist=["*"])


class FakeContext:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get_config(self, key, default=None):
        return self.values.get(key, default)


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


class TargetedReadingRouterTests(unittest.TestCase):
    def make_advisor(self, sender, values=None):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        store = bridge.BridgeStore(root.name)
        advisor = router.TargetedReadingAdvisor(
            FakeContext(values),
            store,
            post_fn=sender,
            clock=lambda: 100.0,
        )
        return advisor, store

    def test_fixture_response_is_typed_and_mandatory_reads_are_preserved(self):
        parsed = router._parse_reference_response(
            router.make_fixture_response("configuration_reference")
        )
        self.assertEqual(parsed["route"], "configuration_reference")
        self.assertEqual(set(parsed["probabilities"]), set(router.ROUTE_BUNDLES))
        refs, reason = router._required_reads(
            parsed["route"], "Please update config and handle a secret token safely"
        )
        self.assertEqual(refs[0], "SKILL.md")
        self.assertIn("references/configuration.md", refs)
        self.assertIn("references/security-privacy.md", refs)
        self.assertIn("mandatory_security_or_authority", reason)
        self.assertIn("mandatory_configuration", reason)

    def test_primary_advisory_deduplicates_and_logs_only_redacted_metadata(self):
        sender = Sender(router.make_fixture_response("configuration_reference"))
        advisor, store = self.make_advisor(sender)
        secret_text = "configure token SUPER_SECRET_VALUE and preserve authority"
        result = advisor.pre_llm_call(
            session_id="session-1",
            task_id="task-1",
            turn_id="turn-1",
            user_message=secret_text,
            is_first_turn=True,
            platform="discord",
        )
        self.assertIsNotNone(result)
        self.assertIn("configuration_reference", result["context"])
        self.assertEqual(len(sender.calls), 1)

        duplicate = advisor.pre_llm_call(
            session_id="session-1",
            task_id="task-1",
            turn_id="turn-2",
            user_message="same session second turn",
            is_first_turn=True,
            platform="discord",
        )
        self.assertIsNone(duplicate)
        self.assertEqual(len(sender.calls), 1)

        rows = [json.loads(line) for line in advisor.reading_path.read_text().splitlines()]
        advisory = next(row for row in rows if row["kind"] == "skill_reading_advisory")
        self.assertEqual(advisory["status"], "advisory")
        self.assertFalse(advisory["request_text_logged"])
        self.assertNotIn("SUPER_SECRET_VALUE", advisor.reading_path.read_text())
        self.assertEqual(advisory["cost"], "not_available")
        self.assertEqual(store.root.stat().st_mode & 0o777, 0o700)
        self.assertEqual(advisor.reading_path.stat().st_mode & 0o777, 0o600)

    def test_malformed_primary_fails_open_without_fallback(self):
        sender = Sender({"answers": {"reference_route": {"choice": "bad"}}})
        advisor, _ = self.make_advisor(sender)
        result = advisor.pre_llm_call(
            session_id="session-malformed",
            task_id="task-2",
            turn_id="turn-1",
            user_message="show the CLI reference",
            is_first_turn=True,
        )
        self.assertIsNone(result)
        self.assertEqual(len(sender.calls), 1)
        rows = [json.loads(line) for line in advisor.reading_path.read_text().splitlines()]
        self.assertEqual(rows[-1]["status"], "malformed")
        self.assertEqual(rows[-1]["reason_code"], "malformed_response")

    def test_timeout_uses_existing_bounded_fallback(self):
        sender = Sender(
            bridge.JevRequestError("timeout"),
            router.make_fixture_response("cli_reference"),
        )
        advisor, _ = self.make_advisor(sender)
        result = advisor.pre_llm_call(
            session_id="session-fallback",
            task_id="task-3",
            turn_id="turn-1",
            user_message="which CLI command should I run",
            is_first_turn=True,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(sender.calls), 2)
        rows = [json.loads(line) for line in advisor.reading_path.read_text().splitlines()]
        advisory = next(row for row in rows if row["kind"] == "skill_reading_advisory")
        self.assertEqual(advisory["status"], "fallback")
        self.assertTrue(advisory["fallback_used"])
        self.assertEqual(advisory["provider_attempts"], 2)
        self.assertEqual(advisory["candidate_references"], ["references/cli-reference.md"])

    def test_reservation_accounting_allows_more_than_ten_calls(self):
        sender = Sender(*(router.make_fixture_response("configuration_reference") for _ in range(11)))
        advisor, _ = self.make_advisor(sender)

        results = [
            advisor.pre_llm_call(
                session_id=f"session-{index}",
                task_id="task-over-ten",
                turn_id=f"turn-{index}",
                user_message="read configuration settings",
                is_first_turn=True,
            )
            for index in range(11)
        ]

        self.assertTrue(all(result and "configuration_reference" in result["context"] for result in results))
        self.assertEqual(len(sender.calls), 11)
        rows = [json.loads(line) for line in advisor.reading_path.read_text().splitlines()]
        reservations = [row for row in rows if row["kind"] == "skill_reading_reservation"]
        advisories = [row for row in rows if row["kind"] == "skill_reading_advisory"]
        self.assertEqual(len(reservations), 11)
        self.assertEqual(sum(row["reserved_calls"] for row in reservations), 22)
        self.assertEqual(len(advisories), 11)

    def test_skill_view_observation_compares_candidate_and_reread(self):
        sender = Sender(router.make_fixture_response("configuration_reference"))
        advisor, _ = self.make_advisor(sender)
        advisor.pre_llm_call(
            session_id="session-observe",
            task_id="task-4",
            turn_id="turn-1",
            user_message="read configuration settings",
            is_first_turn=True,
        )
        args = {"name": "hermes-agent", "file_path": "references/configuration.md"}
        advisor.post_tool_call(
            tool_name="skill_view", args=args, session_id="session-observe", turn_id="turn-1", status="ok"
        )
        advisor.post_tool_call(
            tool_name="skill_view", args=args, session_id="session-observe", turn_id="turn-1", status="ok"
        )
        rows = [json.loads(line) for line in advisor.reading_path.read_text().splitlines()]
        observations = [row for row in rows if row["kind"] == "skill_read_observation"]
        self.assertEqual(len(observations), 2)
        self.assertTrue(observations[0]["candidate_hit"])
        self.assertFalse(observations[0]["re_read"])
        self.assertTrue(observations[1]["re_read"])
        self.assertFalse(observations[0]["content_logged"])

    def test_skill_view_refreshes_later_turn_and_transforms_result(self):
        sender = Sender(
            router.make_fixture_response("configuration_reference"),
            router.make_fixture_response("cli_reference"),
        )
        advisor, _ = self.make_advisor(sender)
        first = advisor.pre_llm_call(
            session_id="session-refresh", task_id="task-refresh", turn_id="turn-1",
            user_message="read configuration settings", is_first_turn=True,
        )
        self.assertIsNotNone(first)
        self.assertEqual(len(sender.calls), 1)

        initial_raw = json.dumps({"success": True, "content": "hub", "metadata": {"n": 1}})
        advisor.post_tool_call(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "SKILL.md"},
            session_id="session-refresh", turn_id="turn-1", status="ok",
        )
        initial_transformed = advisor.transform_tool_result(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "SKILL.md"},
            result=initial_raw, session_id="session-refresh", task_id="task-refresh", turn_id="turn-1",
        )
        self.assertIsNotNone(initial_transformed)
        initial_obj = json.loads(initial_transformed)
        self.assertEqual(initial_obj["content"], "hub")
        self.assertEqual(initial_obj["jev_advisory"]["route"], "configuration_reference")

        # Every turn is captured, but an ordinary later turn does not spend a
        # provider call until the exact hermes-agent skill read is requested.
        self.assertIsNone(advisor.pre_llm_call(
            session_id="session-refresh", task_id="task-refresh", turn_id="turn-2",
            user_message="show the CLI reference", is_first_turn=False,
        ))
        self.assertEqual(len(sender.calls), 1)
        later_raw = json.dumps({"success": True, "content": "hub", "metadata": {"n": 2}})
        advisor.post_tool_call(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "SKILL.md"},
            session_id="session-refresh", turn_id="turn-2", status="ok",
        )
        delivered = advisor.transform_tool_result(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "SKILL.md"},
            result=later_raw, session_id="session-refresh", task_id="task-refresh", turn_id="turn-2",
        )
        self.assertIsNotNone(delivered)
        delivered_obj = json.loads(delivered)
        self.assertEqual(delivered_obj["content"], "hub")
        self.assertEqual(delivered_obj["metadata"], {"n": 2})
        self.assertEqual(delivered_obj["jev_advisory"]["route"], "cli_reference")
        self.assertEqual(len(sender.calls), 2)

        # A same-turn reference read reuses the new advisory without another
        # provider call and still keeps its own original content untouched.
        reference_raw = json.dumps({"success": True, "content": "reference", "metadata": {"n": 3}})
        advisor.post_tool_call(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "references/cli-reference.md"},
            session_id="session-refresh", turn_id="turn-2", status="ok",
        )
        reference_transformed = advisor.transform_tool_result(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "references/cli-reference.md"},
            result=reference_raw, session_id="session-refresh", task_id="task-refresh", turn_id="turn-2",
        )
        self.assertIsNotNone(reference_transformed)
        reference_obj = json.loads(reference_transformed)
        self.assertEqual(reference_obj["content"], "reference")
        self.assertEqual(reference_obj["jev_advisory"]["route"], "cli_reference")
        self.assertEqual(len(sender.calls), 2)

        rows = [json.loads(line) for line in advisor.reading_path.read_text().splitlines()]
        advisory_ids = [row["request_id"] for row in rows if row["kind"] == "skill_reading_advisory"]
        observations = [row for row in rows if row["kind"] == "skill_read_observation"]
        self.assertEqual(len(advisory_ids), 2)
        self.assertEqual({row["request_id"] for row in observations}, {advisory_ids[0], advisory_ids[-1]})

    def test_parallel_skill_reads_coalesce_and_other_skills_do_not_call(self):
        sender = Sender(router.make_fixture_response("main_only"))
        advisor, _ = self.make_advisor(sender)
        self.assertIsNone(advisor.pre_llm_call(
            session_id="session-parallel", task_id="task-parallel", turn_id="turn-2",
            user_message="ordinary later request", is_first_turn=False,
        ))
        self.assertIsNone(advisor.transform_tool_result(
            tool_name="skill_view", args={"name": "other-skill", "file_path": "SKILL.md"},
            result='{"success": true}', session_id="session-parallel", turn_id="turn-2",
        ))
        self.assertEqual(len(sender.calls), 0)

        raw = '{"success": true, "content": "hub"}'
        advisor.post_tool_call(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "SKILL.md"},
            session_id="session-parallel", turn_id="turn-2", status="ok",
        )
        first = advisor.transform_tool_result(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "SKILL.md"},
            result=raw, session_id="session-parallel", task_id="task-parallel", turn_id="turn-2",
        )
        advisor.post_tool_call(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "references/configuration.md"},
            session_id="session-parallel", turn_id="turn-2", status="ok",
        )
        second = advisor.transform_tool_result(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "references/configuration.md"},
            result=raw, session_id="session-parallel", task_id="task-parallel", turn_id="turn-2",
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(json.loads(first)["jev_advisory"]["route"], "main_only")
        self.assertEqual(json.loads(second)["jev_advisory"]["route"], "main_only")
        self.assertEqual(len(sender.calls), 1)

    def test_transform_timeout_and_malformed_fail_open_without_rewrite(self):
        sender = Sender(bridge.JevRequestError("timeout"), bridge.JevRequestError("timeout"))
        advisor, _ = self.make_advisor(sender)
        self.assertIsNone(advisor.pre_llm_call(
            session_id="session-fail-open", task_id="task-fail-open", turn_id="turn-2",
            user_message="later request", is_first_turn=False,
        ))
        original = '{"success": true, "content": "hub"}'
        advisor.post_tool_call(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "SKILL.md"},
            session_id="session-fail-open", turn_id="turn-2",
        )
        transformed = advisor.transform_tool_result(
            tool_name="skill_view", args={"name": "hermes-agent", "file_path": "SKILL.md"},
            result=original, session_id="session-fail-open", task_id="task-fail-open", turn_id="turn-2",
        )
        self.assertIsNone(transformed)
        self.assertEqual(len(sender.calls), 2)

    def test_request_body_is_bounded_and_contains_no_loader_authority(self):
        request = "x" * router.MAX_REQUEST_CHARS
        body = router._request_body(
            request,
            request_id="jread-test",
            platform="discord",
            model="jev-latest",
        )
        self.assertLessEqual(len(body["state"].encode("utf-8")), router.MAX_STATE_BYTES)
        state = json.loads(body["state"])
        self.assertTrue(state["advisory_only"])
        self.assertEqual(state["hub"], "hermes-agent")
        self.assertIn("Do not execute tools", state["selection_scope"])
        self.assertIn("worker/skill loader", " ".join(state["hard_invariants"]))


if __name__ == "__main__":
    unittest.main()
