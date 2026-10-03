from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import importlib.util
import sys

PACKAGE_NAME = "purpose_background_test_plugin"
spec = importlib.util.spec_from_file_location(
    PACKAGE_NAME, Path(__file__).with_name("__init__.py"),
    submodule_search_locations=[str(Path(__file__).parent)])
package = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE_NAME] = package
spec.loader.exec_module(package)
purpose_background = sys.modules[PACKAGE_NAME + ".purpose_background"]
bridge = sys.modules[PACKAGE_NAME + ".bridge"]
from request_goals import RequestGoalStore, register_request_goal_route


class Context:
    def __init__(self, path: Path):
        self._config = {"bridge_dir": str(path), "default_scope.enabled": True}
    def get_config(self, key, default=None):
        return self._config.get(key, default)


class PurposeBackgroundTests(unittest.TestCase):
    def test_registered_plugin_hooks_roundtrip_with_confirmed_root(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)

        class Host(Context):
            profile_name = "default"
            def __init__(self, path):
                super().__init__(path)
                self._config.update({"role": "consumer", "consumer_profiles": ["default"],
                                     "producer_profiles": ["ops"], "targeted_reading.enabled": False})
                self.tools = {}
                self.hooks = {}
            def register_tool(self, **kwargs): self.tools[kwargs["name"]] = kwargs["handler"]
            def register_hook(self, name, callback): self.hooks.setdefault(name, []).append(callback)

        ctx = Host(root / "bridge")
        package.register(ctx)
        advisor = next(cb.__self__ for cb in ctx.hooks["pre_llm_call"]
                       if getattr(getattr(cb, "__self__", None), "__class__", type(None)).__name__ == "PurposeBackgroundAdvisor")
        route = next(cb.__self__ for cb in ctx.hooks["pre_llm_call"]
                     if getattr(getattr(cb, "__self__", None), "__class__", type(None)).__name__ == "RequestGoalRoute")
        self.assertEqual(len([cb for cb in ctx.hooks.get("post_llm_call", [])
                              if getattr(getattr(cb, "__self__", None), "__class__", type(None)).__name__ == "PurposeBackgroundAdvisor"]), 1)
        os.environ["HERMES_PROFILE"] = "default"
        self.addCleanup(os.environ.pop, "HERMES_PROFILE", None)
        advisor.scope_guard.decision_fn = lambda request, request_id=None: {
            "label": "insufficient_information", "confidence": 0.2,
            "purpose_assessment": {"category": "purpose_advancing_action", "confidence": 0.82}}
        route.host_message(user_message="Plan a reliable data service", session_id="s", turn_id="t1",
                           sender_id="u", platform="discord")
        route.host_message(user_message="It must recover from transient failures", session_id="s", turn_id="t2",
                           sender_id="u", platform="discord")
        source = next(r for r in reversed(route.store._rows()) if r.get("kind") == "source")
        offered = route.offer_initial_tool({"source_id": source["source_id"],
            "outcome": "Deliver reliable data", "question": "Before I implement, should I deliver reliable data?",
            "completion_conditions": ["A validation query passes"]}, session_id="s", turn_id="t2")
        self.assertTrue(json.loads(offered)["ok"])
        for cb in ctx.hooks.get("post_llm_call", []):
            cb(session_id="s", turn_id="t2", assistant_response="Before I implement, should I deliver reliable data?")
        route.host_message(user_message="Yes", session_id="s", turn_id="t3", sender_id="u", platform="discord")
        root_goal = route.store.select_for_session("s")
        self.assertIsNotNone(root_goal)
        for cb in ctx.hooks.get("pre_llm_call", []):
            cb(session_id="s", turn_id="t4", user_message="Fix a failing query",
               conversation_history=[{"role": "user", "content": "Plan a reliable data service"},
                                     {"role": "user", "content": "Fix a failing query"}],
               completion_conditions=["Validation passes"])
        for cb in ctx.hooks.get("post_llm_call", []):
            cb(session_id="s", turn_id="t4", user_message="Fix a failing query",
               assistant_response="The query runs now.")
        for _ in range(100):
            if any(r.get("kind") == "purpose_background_assessment"
                   for r in advisor.scope_guard.store.default_scopes()):
                break
            time.sleep(.01)
        outputs = [cb(session_id="s", turn_id="t5", user_message="Continue")
                   for cb in ctx.hooks.get("pre_llm_call", [])]
        contexts = [value["context"] for value in outputs if isinstance(value, dict) and "context" in value]
        self.assertTrue(any("category=purpose_advancing_action" in value and "root_version=1" in value
                            for value in contexts), contexts)

    def _setup(self, provider):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        ctx = Context(root / "bridge")
        store = bridge.BridgeStore(root / "bridge")
        guard = bridge.DefaultScopeGuard(ctx, store, decision_fn=provider)
        goals = RequestGoalStore(root / "goals.jsonl")
        route = register_request_goal_route(_Tools(), goals)
        return root, ctx, store, guard, goals, route

    def _bind_goal(self, goals, route):
        for tid, text in (("t1", "Plan a reliable data service"), ("t2", "It must recover from transient failures")):
            route.host_message(user_message=text, session_id="s", turn_id=tid, sender_id="u", platform="discord")
        source = next(r for r in reversed(goals._rows()) if r.get("kind") == "source")
        offered = goals.offer_initial(source_id=source["source_id"], outcome="Deliver reliable data",
            question="Before I implement, should I deliver reliable data?", completion_conditions=["A validation query passes"])
        goals.mark_offer_delivered(offer_id=offered["offer_id"],
            assistant_response=offered["question"], session_id="s", turn_id="t2")
        rows = goals._rows()
        confirm_source = goals.record_host_source(user_message="Yes", session_id="s", turn_id="t3", sender_id="u", platform="discord")
        goals.confirm_initial_from_host(source_id=confirm_source, user_message="Yes", session_id="s", turn_id="t3", sender_id="u", platform="discord")
        return goals.select_for_session("s")

    def test_post_turn_returns_immediately_then_delivers_single_version_bound_advisory(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []
        def provider(request, request_id=None):
            self.assertIn("request_goal_ref_json", request["state"])
            self.assertIn("confirmed_request_goal", request["state"])
            calls.append((request, request_id))
            entered.set()
            release.wait(2)
            return {"label": "insufficient_information", "confidence": 0.2,
                    "purpose_assessment": {"category": "purpose_advancing_action", "confidence": 0.84}}
        root, ctx, store, guard, goals, route = self._setup(provider)
        selected = self._bind_goal(goals, route)
        session_identity, turn_identity = bridge._scope_identity("s", "t4")
        guard._contracts[f"{session_identity}:{turn_identity}"] = {
            "session_identity": session_identity, "turn_identity": turn_identity,
            "session_hash": bridge._identity_hash(session_identity),
            "turn_hash": bridge._identity_hash(turn_identity),
            "original_request": "Fix a failing query", "completion_conditions": ["Validation passes"],
            "current_unknown": "query result", "evidence_refs": [],
            "purpose_projection": {"observed_first_user_goal": "Plan a reliable data service",
                "current_user_instruction": "Fix a failing query",
                "recent_user_attributed_history": [{"role": "user", "content": "Fix a failing query"}],
                "user_constraints": [], "current_local_problem": "query failure", "observed_outcome": "unknown"}}
        advisor = purpose_background.PurposeBackgroundAdvisor(ctx, guard, goals)
        started = time.monotonic()
        advisor.post_llm_call(session_id="s", turn_id="t4", user_message="Fix a failing query",
                              assistant_response="The query runs now.", conversation_history=[])
        self.assertLess(time.monotonic() - started, .2)
        self.assertTrue(entered.wait(1))
        self.assertEqual(len(calls), 1)
        release.set()
        for _ in range(100):
            if any(r.get("kind") == "purpose_background_assessment" for r in store.default_scopes()):
                break
            time.sleep(.01)
        context = advisor.pre_llm_call(session_id="s", turn_id="t5")
        self.assertIn("category=purpose_advancing_action", context["context"])
        self.assertIn("assessed_turn=turn:t4", context["context"])
        self.assertIn("root_version=1", context["context"])
        self.assertIn("outcome=assistant-reported, not independently verified: The query runs now.", context["context"])
        self.assertEqual(advisor.pre_llm_call(session_id="s", turn_id="t6"), None)
        advisor.post_llm_call(session_id="s", turn_id="t4")
        self.assertEqual(len(calls), 1)
        rows = store.default_scopes()
        self.assertEqual(len([r for r in rows if r.get("kind") == "purpose_background_delivery"]), 1)
        assessment = next(r for r in rows if r.get("kind") == "purpose_background_assessment")
        self.assertEqual(assessment["root_id"], selected["root_id"])
        self.assertEqual(assessment["root_version"], 1)

    def test_late_result_is_discarded_and_inflight_slot_prevents_overlap(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []
        active = 0
        max_active = 0
        def provider(request, request_id=None):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            calls.append(request)
            try:
                if len(calls) == 1:
                    entered.set()
                    release.wait(2)
                return {"label": "insufficient_information", "confidence": 0.2,
                        "purpose_assessment": {"category": "procedure_success_but_outcome_missing", "confidence": 0.91}}
            finally:
                active -= 1
        root, ctx, store, guard, goals, route = self._setup(provider)
        self._bind_goal(goals, route)
        advisor = purpose_background.PurposeBackgroundAdvisor(ctx, guard, goals)
        for turn in ("t4", "t6"):
            session_identity, turn_identity = bridge._scope_identity("s", turn)
            guard._contracts[f"{session_identity}:{turn_identity}"] = {
                "session_identity": session_identity, "turn_identity": turn_identity,
                "session_hash": bridge._identity_hash(session_identity),
                "turn_hash": bridge._identity_hash(turn_identity),
                "original_request": f"request {turn}", "completion_conditions": ["Validation passes"],
                "current_unknown": "query result", "evidence_refs": ["case://bounded-evidence"],
                "purpose_projection": {"observed_first_user_goal": "Plan a reliable data service",
                    "current_user_instruction": f"request {turn}", "recent_user_attributed_history": [],
                    "user_constraints": [], "current_local_problem": "query failure", "observed_outcome": "unknown"}}
        advisor.post_llm_call(session_id="s", turn_id="t4", user_message="request t4", assistant_response="old result")
        self.assertTrue(entered.wait(1), "provider must be reached before boundary test")
        self.assertIsNone(advisor.pre_llm_call(session_id="s", turn_id="t5"))
        generation = advisor._turn_state[bridge._identity_hash(bridge._scope_identity("s", "t5")[0])][1]
        self.assertIsNone(advisor.pre_llm_call(session_id="s", turn_id="t5"))
        self.assertEqual(advisor._turn_state[bridge._identity_hash(bridge._scope_identity("s", "t5")[0])][1], generation)
        advisor.post_llm_call(session_id="s", turn_id="t6", user_message="request t6", assistant_response="new result")
        self.assertEqual(len(calls), 1, "do not overlap a provider call while the prior transport is still running")
        release.set()
        for _ in range(100):
            if not advisor._inflight:
                break
            time.sleep(.01)
        self.assertFalse(any(r.get("kind") == "purpose_background_assessment" for r in store.default_scopes()))
        self.assertIsNone(advisor.pre_llm_call(session_id="s", turn_id="t7"))
        session_identity, turn_identity = bridge._scope_identity("s", "t7")
        guard._contracts[f"{session_identity}:{turn_identity}"] = {
            "session_identity": session_identity, "turn_identity": turn_identity,
            "session_hash": bridge._identity_hash(session_identity),
            "turn_hash": bridge._identity_hash(turn_identity),
            "original_request": "request t7", "completion_conditions": ["Validation passes"],
            "current_unknown": "query result", "evidence_refs": ["case://bounded-evidence"],
            "purpose_projection": {"observed_first_user_goal": "Plan a reliable data service",
                "current_user_instruction": "request t7", "recent_user_attributed_history": [],
                "user_constraints": [], "current_local_problem": "query failure", "observed_outcome": "unknown"}}
        advisor.post_llm_call(session_id="s", turn_id="t7", user_message="request t7", assistant_response="new result")
        for _ in range(100):
            if any(r.get("kind") == "purpose_background_assessment" for r in store.default_scopes()):
                break
            time.sleep(.01)
        self.assertEqual(len(calls), 2)
        self.assertEqual(max_active, 1)
        assessment = next(r for r in store.default_scopes() if r.get("kind") == "purpose_background_assessment")
        self.assertEqual(assessment["assessed_turn_id"], "turn:t7")
        delivered = advisor.pre_llm_call(session_id="s", turn_id="t8")
        self.assertIn("assessed_turn=turn:t7", delivered["context"])
        self.assertEqual(len([r for r in store.default_scopes() if r.get("kind") == "purpose_background_delivery"]), 1)

    def test_fail_open_when_no_root_or_provider_timeout(self):
        timeout_entered = threading.Event()
        def timeout(*args, **kwargs):
            timeout_entered.set()
            raise bridge.JevRequestError("timeout")
        root, ctx, store, guard, goals, route = self._setup(timeout)
        advisor = purpose_background.PurposeBackgroundAdvisor(ctx, guard, goals)
        advisor.post_llm_call(session_id="s", turn_id="t1")
        self.assertIsNone(advisor.pre_llm_call(session_id="s", turn_id="t2"))
        self._bind_goal(goals, route)
        session_identity, turn_identity = bridge._scope_identity("s", "t4")
        guard._contracts[f"{session_identity}:{turn_identity}"] = {
            "session_identity": session_identity, "turn_identity": turn_identity,
            "session_hash": bridge._identity_hash(session_identity),
            "turn_hash": bridge._identity_hash(turn_identity),
            "original_request": "Fix a failing query", "completion_conditions": ["Validation passes"],
            "current_unknown": "query result", "evidence_refs": [],
            "purpose_projection": {"observed_first_user_goal": "Plan a reliable data service",
                "current_user_instruction": "Fix a failing query",
                "recent_user_attributed_history": [{"role": "user", "content": "Fix a failing query"}],
                "user_constraints": [], "current_local_problem": "query failure", "observed_outcome": "unknown"}}
        advisor.pre_llm_call(session_id="s", turn_id="t4")
        advisor.post_llm_call(session_id="s", turn_id="t4", user_message="Fix a failing query",
                              assistant_response="Still checking")
        for _ in range(100):
            if not advisor._inflight:
                break
            time.sleep(.01)
        self.assertTrue(timeout_entered.is_set(), "timeout fixture must exercise the provider")
        self.assertIsNone(advisor.pre_llm_call(session_id="s", turn_id="t5"))

    def test_stale_version_is_discarded_and_registered_hooks_are_isolated(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []
        def provider(request, request_id=None):
            calls.append(request)
            entered.set()
            release.wait(2)
            return {"label": "insufficient_information", "confidence": .2,
                    "purpose_assessment": {"category": "procedure_success_but_outcome_missing", "confidence": .91}}
        root, ctx, store, guard, goals, route = self._setup(provider)
        selected = self._bind_goal(goals, route)
        session_identity, turn_identity = bridge._scope_identity("s", "t4")
        guard._contracts[f"{session_identity}:{turn_identity}"] = {
            "session_identity": session_identity, "turn_identity": turn_identity,
            "session_hash": bridge._identity_hash(session_identity),
            "turn_hash": bridge._identity_hash(turn_identity),
            "original_request": "Fix a failing query", "completion_conditions": ["Validation passes"],
            "current_unknown": "query result", "evidence_refs": [],
            "purpose_projection": {"observed_first_user_goal": "Plan a reliable data service",
                "current_user_instruction": "Fix a failing query", "recent_user_attributed_history": [],
                "user_constraints": [], "current_local_problem": "query failure", "observed_outcome": "unknown"}}
        advisor = purpose_background.PurposeBackgroundAdvisor(ctx, guard, goals)
        advisor.post_llm_call(session_id="s", turn_id="t4", user_message="Fix a failing query", assistant_response="Done")
        self.assertTrue(entered.wait(1))
        goals._append({**selected, "version": 2, "outcome": "Changed explicitly"})
        release.set()
        for _ in range(100):
            if not advisor._inflight:
                break
            time.sleep(.01)
        self.assertFalse(any(r.get("kind") == "purpose_background_assessment" for r in store.default_scopes()))


class _Tools:
    def register_tool(self, **kwargs): pass
    def register_hook(self, *args): pass


if __name__ == "__main__":
    unittest.main()
