from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from request_goals import GoalError, RequestGoalStore


class RequestGoalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RequestGoalStore(Path(self.tmp.name) / "goals.jsonl")
        self.source = self.store.record_host_source(
            user_message="new request: Deliver the full news briefing", session_id="s1",
            turn_id="t1", sender_id="u1", platform="discord")
        self.root = self.store.bind(source_id=self.source,
                                    completion_conditions=["deliver all requested sections"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_initial_confirmation_requires_exact_safe_affirmative(self):
        positive = ("OK", "Okay!", "YES.", "はい！", "それでお願いします。")
        negative = (
            "はい、ただしその案では進めないでください。",
            "yes, but do not proceed with that plan",
            "OK, but only if you change it",
            "yes and also send it to everyone",
            "not okay",
            "maybe yes",
        )
        for message in positive:
            self.assertTrue(RequestGoalStore._positive_confirmation(message), message)
        for message in negative:
            self.assertFalse(RequestGoalStore._positive_confirmation(message), message)

    def test_local_repair_and_unfamiliar_wording_do_not_replace_root(self):
        self.store.add_local_milestone(root_id=self.root["root_id"],
                                       milestone="repair validator error")
        projected = RequestGoalStore(self.store.path).get(self.root["root_id"])
        self.assertEqual(projected["outcome"], "Deliver the full news briefing")
        self.assertEqual(projected["version"], 1)
        self.assertEqual(projected["local_milestones"], ["repair validator error"])

    def test_only_existing_host_source_can_bind_independent_root(self):
        with self.assertRaises(GoalError):
            self.store.bind(source_id="model-invented", root_id="new")
        source2 = self.store.record_host_source(
            user_message="new request: A separate request", session_id="s1", turn_id="t2",
            sender_id="u1", platform="discord")
        root2 = self.store.bind(source_id=source2)
        self.assertNotEqual(root2["root_id"], self.root["root_id"])
        self.assertEqual(self.store.get(self.root["root_id"])["outcome"],
                         "Deliver the full news briefing")

    def test_proposal_waits_for_exact_later_host_confirmation(self):
        proposed_text = "Deliver the full news briefing in Japanese"
        proposal_source = self.store.record_host_source(user_message="request proposal",
            session_id="s1", turn_id="t2", sender_id="u1", platform="discord")
        info = self.store.propose(root_id=self.root["root_id"], version=1,
                                  proposed=proposed_text, reason="language clarification",
                                  impact="changes output language", source_id=proposal_source)
        self.assertEqual(info["current"], "Deliver the full news briefing")
        self.assertEqual(info["proposed"], proposed_text)
        ok_source = self.store.record_host_source(user_message="OK",
            session_id="s1", turn_id="t3", sender_id="u1", platform="discord")
        with self.assertRaises(GoalError):
            self.store.approve_from_host(proposal_id=info["proposal_id"],
                source_id=ok_source, user_message="OK", session_id="s1", turn_id="t3",
                sender_id="u1", platform="discord")
        confirmation = f"approve amendment {info['proposal_id']}"
        confirmation_source = self.store.record_host_source(user_message=confirmation,
            session_id="s1", turn_id="t3", sender_id="u1", platform="discord")
        with self.assertRaises(GoalError):
            self.store.approve_from_host(proposal_id=info["proposal_id"],
                source_id=confirmation_source, user_message=confirmation,
                session_id="s1", turn_id="t3", sender_id="other", platform="discord")
        changed = self.store.approve_from_host(proposal_id=info["proposal_id"],
            source_id=confirmation_source, user_message=confirmation,
            session_id="s1", turn_id="t3", sender_id="u1", platform="discord")
        self.assertEqual(changed["version"], 2)
        self.assertEqual(self.store.get(self.root["root_id"])["outcome"], proposed_text)
        self.assertEqual(self.store.pending_proposals(self.root["root_id"]), [])
        with self.assertRaises(GoalError):
            self.store.approve_from_host(proposal_id=info["proposal_id"],
                source_id=confirmation_source, user_message=confirmation,
                session_id="s1", turn_id="t3", sender_id="u1", platform="discord")

    def test_source_over_limit_is_unknown_not_truncated(self):
        with self.assertRaisesRegex(GoalError, "unknown, not truncated"):
            self.store.record_host_source(user_message="日" * 2731,
                session_id="s1", turn_id="t3", sender_id="u1", platform="discord")

    def test_stale_proposal_cannot_amend(self):
        source2 = self.store.record_host_source(user_message="proposal 1",
            session_id="s1", turn_id="t2", sender_id="u1", platform="discord")
        source3 = self.store.record_host_source(user_message="proposal 2",
            session_id="s1", turn_id="t3", sender_id="u1", platform="discord")
        first = self.store.propose(root_id=self.root["root_id"], version=1,
            proposed="first", reason="r", impact="i", source_id=source2)
        stale = self.store.propose(root_id=self.root["root_id"], version=1,
            proposed="stale", reason="r", impact="i", source_id=source3)
        confirmation = f"approve amendment {first['proposal_id']}"
        confirm_source = self.store.record_host_source(user_message=confirmation,
            session_id="s1", turn_id="t4", sender_id="u1", platform="discord")
        self.store.approve_from_host(proposal_id=first["proposal_id"],
            source_id=confirm_source, user_message=confirmation,
            session_id="s1", turn_id="t4", sender_id="u1", platform="discord")
        stale_confirmation = f"approve amendment {stale['proposal_id']}"
        stale_source = self.store.record_host_source(user_message=stale_confirmation,
            session_id="s1", turn_id="t5", sender_id="u1", platform="discord")
        with self.assertRaises(GoalError):
            self.store.approve_from_host(proposal_id=stale["proposal_id"],
                source_id=stale_source, user_message=stale_confirmation,
                session_id="s1", turn_id="t5", sender_id="u1", platform="discord")


if __name__ == "__main__":
    unittest.main()
