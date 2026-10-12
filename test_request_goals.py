from __future__ import annotations

import json
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

    def test_live_goal_store_copy_resolves_like_shared_reader(self):
        workspace_copy = Path(__file__).resolve().parent.parent / "live-request-goals.jsonl"
        self.assertTrue(workspace_copy.is_file())
        store = RequestGoalStore(workspace_copy)
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "legacy-goals.jsonl"
            copied.write_bytes(workspace_copy.read_bytes())
            projected = RequestGoalStore(copied)._rows()
        self.assertEqual(store._rows(), projected)
        self.assertTrue(all(row.get("kind") == "source" for row in projected))

    def test_confirmation_is_one_append_and_projects_after_append_crash(self):
        source = self.store.record_host_source(
            user_message="Can you help me think through a project?", session_id="confirm-session",
            turn_id="offer-turn", sender_id="owner", platform="discord")
        offer = self.store.offer_initial(source_id=source, outcome="staged outcome",
            question="Confirm staged outcome?", completion_conditions=["complete it"])
        self.store.mark_offer_delivered(offer_id=offer["offer_id"],
            assistant_response="Confirm staged outcome?", session_id="confirm-session", turn_id="offer-turn")
        confirmation_source = self.store.record_host_source(
            user_message="OK", session_id="confirm-session", turn_id="confirm-turn",
            sender_id="owner", platform="discord")
        original_append = self.store._append
        appended = []
        def append_then_crash(row):
            original_append(row)
            appended.append(row)
            if row.get("kind") == "initial_goal_confirmation":
                raise RuntimeError("simulated crash after confirmation append")
        self.store._append = append_then_crash
        with self.assertRaisesRegex(RuntimeError, "simulated crash"):
            self.store.confirm_initial_from_host(source_id=confirmation_source, user_message="OK",
                session_id="confirm-session", turn_id="confirm-turn", sender_id="owner", platform="discord")
        self.assertEqual(len(appended), 1)
        self.assertEqual(appended[0].get("kind"), "initial_goal_confirmation")
        reloaded = RequestGoalStore(self.store.path)
        self.assertEqual(reloaded.select_for_session("confirm-session")["outcome"], "staged outcome")
        self.assertEqual(reloaded.get(appended[0]["root"]["root_id"])["outcome"], "staged outcome")

    def test_atomic_confirmation_reader_expands_single_record_and_keeps_legacy_events(self):
        event = {
            "kind": "initial_goal_confirmation",
            "offer_state": {"kind": "initial_goal_offer_state", "offer_id": "offer-1", "status": "confirmed"},
            "root": {"kind": "root", "root_id": "root-1", "version": 1, "outcome": "briefing",
                     "identity": {"session_id": "session-1"}},
            "selection": {"kind": "selection", "root_id": "root-1", "version": 1,
                          "identity": {"session_id": "session-1"}},
        }
        legacy_root = {"kind": "root", "root_id": "legacy-root", "version": 1, "outcome": "legacy briefing",
                       "identity": {"session_id": "session-2"}}
        legacy_selection = {"kind": "selection", "root_id": "legacy-root", "version": 1,
                            "identity": {"session_id": "session-2"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "goals.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in (event, legacy_root, legacy_selection,
                {"kind": "source", "source_id": "legacy"})) + "\n")
            store = RequestGoalStore(path)
            rows = store._rows()
            self.assertEqual([row["kind"] for row in rows], ["initial_goal_offer_state", "root", "selection", "root", "selection", "source"])
            self.assertEqual(store.get("root-1")["outcome"], "briefing")
            self.assertEqual(store.select_for_session("session-1")["root_id"], "root-1")
            self.assertEqual(store.get("legacy-root")["outcome"], "legacy briefing")
            self.assertEqual(store.select_for_session("session-2")["root_id"], "legacy-root")

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
