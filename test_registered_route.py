from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from request_goals import RequestGoalStore, register_request_goal_route


class FakeContext:
    def __init__(self):
        self.tools = {}
        self.hooks = {}
    def register_tool(self, *, name, schema, handler, **kwargs):
        self.tools[name] = (schema, handler)
    def register_hook(self, name, handler):
        self.hooks.setdefault(name, []).append(handler)


class RegisteredRouteTests(unittest.TestCase):
    def test_registered_route_uses_host_callback_and_has_no_model_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext()
            store = RequestGoalStore(Path(tmp) / "goals.jsonl")
            route = register_request_goal_route(ctx, store)
            self.assertEqual(set(ctx.tools), {"request_goal_offer_initial", "request_goal_bind", "request_goal_propose_amendment"})
            self.assertEqual(len(ctx.hooks["pre_llm_call"]), 1)
            self.assertEqual(len(ctx.hooks["post_llm_call"]), 1)
            host = ctx.hooks["pre_llm_call"][0]
            host(user_message="Help me plan an operations dashboard", session_id="plan", turn_id="p1",
                 sender_id="u", platform="discord")
            first_source = next(row for row in reversed(store._rows()) if row.get("kind") == "source")
            rejected = json.loads(ctx.tools["request_goal_offer_initial"][1]({
                "source_id": first_source["source_id"], "outcome": "Build a dashboard",
                "question": "Before I implement, should I build a dashboard?"}, session_id="plan", turn_id="p1"))
            self.assertFalse(rejected["ok"])
            host(user_message="What functions should it have?", session_id="plan", turn_id="p2",
                 sender_id="u", platform="discord")
            second_source = next(row for row in reversed(store._rows()) if row.get("kind") == "source")
            accepted = json.loads(ctx.tools["request_goal_offer_initial"][1]({
                "source_id": second_source["source_id"], "outcome": "Build an operations dashboard",
                "question": "Before I implement, should I build an operations dashboard?",
                "completion_conditions": ["Dashboard runs locally"]}, session_id="plan", turn_id="p2"))
            self.assertTrue(accepted["ok"], accepted)
            self.assertIsNone(store.select_for_session("plan"))
            route.post_llm_call(session_id="plan", turn_id="p2",
                assistant_response="Before I implement, should I build an operations dashboard?")
            self.assertIsNone(store.select_for_session("plan"))
            route.host_message(user_message="I will think about it", session_id="plan", turn_id="p3",
                sender_id="u", platform="discord")
            self.assertIsNone(store.select_for_session("plan"))
            confirmation_context = route.host_message(user_message="Yes", session_id="plan", turn_id="p4",
                sender_id="u", platform="discord")
            proposed_root = store.select_for_session("plan")
            self.assertIsNotNone(proposed_root)
            self.assertEqual(proposed_root["outcome"], "Build an operations dashboard")
            self.assertEqual(json.loads(confirmation_context["context"])["request_goal"]["root_id"], proposed_root["root_id"])
            host(user_message="new request: Deliver daily news", session_id="s", turn_id="t1",
                 sender_id="u", platform="discord")
            bound = json.loads(host(user_message="new request: Deliver daily news", session_id="s", turn_id="t1",
                sender_id="u", platform="discord")["context"])["request_goal"]
            bind_schema = ctx.tools["request_goal_bind"][0]["parameters"]
            self.assertEqual(bind_schema["properties"], {})
            self.assertFalse(any("approve" in name for name in ctx.tools))
            selected_root = store.select_for_session("s")
            self.assertEqual(bound["root_id"], selected_root["root_id"])
            self.assertEqual(bound["root_id"], json.loads(host(user_message="Fix a validator error",
                session_id="s", turn_id="t1b", sender_id="u", platform="discord")["context"])["request_goal"]["root_id"])
            host(user_message="Please propose a Japanese version",
                session_id="s", turn_id="t2", sender_id="u", platform="discord")
            proposal_source = next(row for row in reversed(store._rows()) if row.get("kind") == "source")
            proposal = json.loads(ctx.tools["request_goal_propose_amendment"][1]({
                "root_id": bound["root_id"], "version": 1,
                "source_id": proposal_source["source_id"],
                "proposed": "Deliver daily news in Japanese", "reason": "language",
                "impact": "changes output language"}))
            self.assertEqual(proposal["current"], "Deliver daily news")
            self.assertEqual(proposal["reason"], "language")
            self.assertEqual(proposal["impact"], "changes output language")
            self.assertEqual(store.get(bound["root_id"])["version"], 1)
            ok_context = host(user_message="OK", session_id="s", turn_id="t3",
                              sender_id="u", platform="discord")
            self.assertEqual(json.loads(ok_context["context"])["request_goal"]["outcome"], "Deliver daily news")
            self.assertEqual(store.get(bound["root_id"])["version"], 1)
            approval_context = host(user_message=proposal["confirmation"], session_id="s", turn_id="t4",
                 sender_id="u", platform="discord")
            self.assertIn("confirmed", approval_context["context"])
            self.assertEqual(store.get(bound["root_id"])["version"], 2)


if __name__ == "__main__":
    unittest.main()
