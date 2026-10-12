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
            self.assertEqual(set(ctx.tools), {
                "request_goal_offer_initial", "request_goal_bind", "request_goal_propose_amendment"})
            self.assertEqual(len(ctx.hooks["pre_llm_call"]), 1)
            self.assertEqual(len(ctx.hooks["post_llm_call"]), 1)
            host = ctx.hooks["pre_llm_call"][0]
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

    def test_registered_initial_offer_requires_delivery_and_later_same_host_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext()
            store = RequestGoalStore(Path(tmp) / "goals.jsonl")
            route = register_request_goal_route(ctx, store)
            host = ctx.hooks["pre_llm_call"][0]
            post = ctx.hooks["post_llm_call"][0]
            offer_tool = ctx.tools["request_goal_offer_initial"][1]
            bind_tool = ctx.tools["request_goal_bind"][1]

            first = host(user_message="Can you help me think through a project?",
                session_id="s", turn_id="t1", sender_id="u", platform="discord")
            self.assertIsNotNone(first)
            self.assertIsNone(store.select_for_session("s"))
            self.assertFalse(json.loads(bind_tool({}, session_id="s"))["ok"])

            host(user_message="Let's compare a couple of approaches first.",
                session_id="s", turn_id="t2", sender_id="u", platform="discord")
            source_id = store._rows()[-1]["source_id"]
            args = {"source_id": source_id, "outcome": "Build a small project tracker",
                    "question": "Before I implement, shall I build a small project tracker?",
                    "completion_conditions": ["Create and verify a working tracker"]}
            stale = json.loads(offer_tool(args, session_id="s", turn_id="t1"))
            self.assertFalse(stale["ok"])
            not_transition = json.loads(offer_tool({**args, "question": "Would that be useful?"},
                session_id="s", turn_id="t2"))
            self.assertFalse(not_transition["ok"])

            undelivered = json.loads(offer_tool(args, session_id="s", turn_id="t2"))
            self.assertTrue(undelivered["ok"])
            self.assertIsNone(store.select_for_session("s"), "an offer alone must not bind a root")
            post(session_id="s", turn_id="t2", assistant_response="I will proceed with the plan.")
            host(user_message="OK", session_id="s", turn_id="t3", sender_id="u", platform="discord")
            self.assertIsNone(store.select_for_session("s"), "an undelivered question cannot be confirmed")

            # Keep the successful route independent of the deliberately failed offer above.
            # The failed-delivery record remains an outstanding offer in the current store.
            store = RequestGoalStore(Path(tmp) / "successful-goals.jsonl")
            ctx = FakeContext()
            route = register_request_goal_route(ctx, store)
            host = ctx.hooks["pre_llm_call"][0]
            post = ctx.hooks["post_llm_call"][0]
            offer_tool = ctx.tools["request_goal_offer_initial"][1]
            host(user_message="Can you help me think through a project?",
                session_id="s", turn_id="t4", sender_id="u", platform="discord")
            host(user_message="Let's refine the design before implementation.",
                session_id="s", turn_id="t5", sender_id="u", platform="discord")
            second_source_id = store._rows()[-1]["source_id"]
            question = "Before implementation, shall I create a lightweight project tracker?"
            staged = json.loads(offer_tool({"source_id": second_source_id,
                "outcome": "Create a lightweight project tracker",
                "question": question,
                "completion_conditions": ["Track tasks", "Verify task updates"]},
                session_id="s", turn_id="t5"))
            self.assertTrue(staged["ok"])
            self.assertIsNone(store.select_for_session("s"))
            post(session_id="s", turn_id="t5",
                assistant_response=f"I can start now. {question}")

            host(user_message="OK", session_id="s", turn_id="t5", sender_id="u", platform="discord")
            self.assertIsNone(store.select_for_session("s"), "same-turn OK is not a later confirmation")
            host(user_message="OK", session_id="s", turn_id="t6", sender_id="other",
                platform="discord")
            self.assertIsNone(store.select_for_session("s"), "a different host cannot bind the offer")
            confirmed = host(user_message="OK", session_id="s", turn_id="t7",
                sender_id="u", platform="discord")
            root = store.select_for_session("s")
            self.assertIsNotNone(root)
            self.assertEqual(root["outcome"], "Create a lightweight project tracker")
            self.assertEqual(root["completion_conditions"], ["Track tasks", "Verify task updates"])
            self.assertEqual(json.loads(confirmed["context"])["request_goal"]["root_id"], root["root_id"])


if __name__ == "__main__":
    unittest.main()
