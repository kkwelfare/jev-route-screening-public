"""Offline structured-input contract tests; no Jev, provider, web, or Gateway calls."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
FIXTURE_TASK_ID = "t_94cd2597"
FIXTURE_NOW = datetime.fromisoformat("2026-09-25T04:00:00+09:00")
PROFILE_PATHS = {profile: PACKAGE_ROOT / "bridge.py" for profile in ("ops", "intl", "docs", "research", "archivist")}


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bridge fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_shared_package() -> Any:
    name = "jev_structured_input_contract_shared"
    spec = importlib.util.spec_from_file_location(
        name,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the shared Jev package")
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return sys.modules[f"{name}.bridge"]


SHARED = _load_shared_package()
# All producer variants intentionally resolve to this unified bridge candidate.
PROFILE_BRIDGES = {profile: SHARED for profile in PROFILE_PATHS}


def _candidate(candidate_id: str, title: str, url: str) -> dict[str, Any]:
    content = f"Offline fixture excerpt for {candidate_id}; not retrieved source text."
    return {
        "candidate_id": candidate_id,
        "title": title,
        "url": url,
        "source": "fixture-only",
        "published_at": "2026-09-25T02:00:00+09:00",
        "fetched_at": "2026-09-25T03:00:00+09:00",
        "relevant_content": content,
        "freshness_context": {"status": "fresh", "rule": "t_94cd2597 fixture timestamp"},
        "deduplication_context": {"status": "checked", "basis": "t_94cd2597 fixture prior-output comparison"},
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "source_references": [url],
    }


def _daily_interest_input() -> dict[str, Any]:
    records = [
        _candidate(
            "fixture-chrome",
            "t_94cd2597 offline fixture candidate A",
            "https://www.theverge.com/tech/999901/google-chrome-tab-memory-gemini-updates",
        ),
        _candidate(
            "fixture-pixel-watch",
            "t_94cd2597 offline fixture candidate B",
            "https://on.theverge.com/tech/999904/google-pixel-watch-5-health-features-rollout",
        ),
    ]
    references = [record["url"] for record in records]
    return {
        "schema_version": 1,
        "scope": "2026-09-25 daily-interest news curation fixture",
        "objective": "preserve the bounded candidate and decision contract",
        "target_candidates": copy.deepcopy(records),
        "requested_outcome": "one-to-eight eligible items or exactly [SILENT]",
        "action_mode": "read_only",
        "allowed_actions": ["compare prior outputs", "verify candidate sources when authorized"],
        "forbidden_actions": ["post to Discord", "change cron or Gateway configuration"],
        "decision_points": [],
        "freshness": {
            "status": "fresh",
            "observed_at": "2026-09-25T03:30:00+09:00",
            "valid_until": "2026-09-25T05:00:00+09:00",
            "rule": "fixture cutoff and publication-time rule",
        },
        "deduplication": {
            "status": "checked",
            "checked_at": "2026-09-25T03:40:00+09:00",
            "basis": "fixture prior seven-day output and topic comparison",
        },
        "baseline": {
            "prior_outputs": ["2026-09-24_19-07-29.md", "2026-09-24_10-27-31.md"],
            "freshness_cutoff": "2026-09-24T16:07:29+09:00",
        },
        "completion_conditions": [
            "adopt only one-to-eight candidates meeting freshness and source quality",
            "produce the specified splitter-ready Japanese manuscript or exact [SILENT]",
        ],
        "prior_context": {
            "previous_curation_run": "2026-09-24T19:07:29Z",
            "known_candidate_freshness": "two fixture candidates require verification",
            "prior_topic_examples": ["Meta Muse", "OpenClaw"],
        },
        "signals": [{"kind": "weak_news_link_signal", "source": "t_94cd2597 fixture"}],
        "feedback": [{"window_days": 21, "like": 1, "neutral": 138, "dislike": 0}],
        "output_requirements": {
            "language": "Japanese",
            "format": "splitter-ready manuscript or exact [SILENT]",
            "sentences_per_item": "two-to-three factual sentences",
        },
        "uncertainty": [],
        "provenance": {
            "task_id": FIXTURE_TASK_ID,
            "source": "offline Kanban task fixture; no retrieval performed",
            "captured_at": "2026-09-25T03:45:00+09:00",
            "source_references": references,
        },
        "domain_extension": {
            "type": "daily_interest_news_curation_v1",
            "candidate_ids": [record["candidate_id"] for record in records],
            "candidate_records": copy.deepcopy(records),
            "unresolved_candidate_ids": [],
            "freshness_dedup_context": {
                "cutoff": "2026-09-24T16:07:29+09:00",
                "prior_outputs": ["2026-09-24_19-07-29.md", "2026-09-24_10-27-31.md"],
                "status": "checked",
            },
            "user_interest_inputs": {"interest_notes": "fixture summary only"},
            "feedback_inputs": {"window_days": 21, "like": 1, "neutral": 138, "dislike": 0},
            "allowed_operations": ["bounded candidate source verification"],
            "forbidden_operations": ["Discord posting", "cron or Gateway changes"],
            "completion_conditions": ["candidate identity and content are integrity checked"],
            "output_requirements": {"format": "splitter-ready Japanese manuscript or exact [SILENT]"},
            "source_references": references,
            "handoff_status": "complete",
            "coverage": {"seen": 2, "expected": 2},
            "truncation_cause": None,
        },
    }


def _generic_input() -> dict[str, Any]:
    value = _daily_interest_input()
    value.update(
        {
            "scope": "unrelated generic read-only task fixture",
            "objective": "preserve generic decision fields without news-specific data",
            "target_candidates": [],
            "domain_extension": None,
            "provenance": {
                "task_id": FIXTURE_TASK_ID,
                "source": "offline generic fixture",
                "captured_at": "2026-09-25T03:45:00+09:00",
                "source_references": [f"kanban://task/{FIXTURE_TASK_ID}"],
            },
        }
    )
    return value


def _task_contract(bridge: Any, profile: str, decision_input: Any) -> Any:
    values = {
        "task_id": FIXTURE_TASK_ID,
        "run_id": "9001",
        "profile": profile,
        "started_at": 1000.0,
        "original_request": "t_94cd2597 structured fixture only",
        "criteria": ("preserve all contract fields", "block invalid input before Jev"),
        "current_unknown": "fixture only",
        "next_action": "return to the worker tool guard",
        "evidence_refs": (f"kanban://task/{FIXTURE_TASK_ID}",),
        "decision_input": copy.deepcopy(decision_input),
    }
    if profile == "research":
        values.update(search_authorized=False, extract_authorized=False)
    return bridge.TaskContract(**values)


def _produce_event(profile: str, decision_input: Any) -> dict[str, Any]:
    bridge = PROFILE_BRIDGES[profile]
    contract = _task_contract(bridge, profile, decision_input)
    with tempfile.TemporaryDirectory(prefix=f"jev-contract-{profile}-") as directory:
        root = Path(directory)
        store = bridge.BridgeStore(root)
        producer = bridge.Producer(
            store,
            profile,
            contract_loader=lambda: contract,
            clock=lambda: 1300.0,
            cadence_seconds=0,
        )
        with patch.object(bridge, "_decision_input_now", return_value=FIXTURE_NOW):
            with patch.dict(
                os.environ,
                {"HERMES_PROFILE": profile, "HERMES_KANBAN_TASK": FIXTURE_TASK_ID, "HERMES_KANBAN_RUN_ID": "9001"},
            ):
                for _ in range(3):
                    producer.post_tool_call(
                        tool_name="read_file",
                        args={"path": "offline-fixture.txt"},
                        result={"ok": True, "changed": True},
                        status="ok",
                    )
        events = store.events()
        if len(events) != 1:
            raise AssertionError(f"{profile} producer expected one event, got {len(events)}")
        return copy.deepcopy(events[0])


class OfflineContext:
    profile_name = "default"

    def __init__(self, point_state_enabled: bool = False) -> None:
        self.point_state_enabled = point_state_enabled

    def get_config(self, key: str, default: Any = None) -> Any:
        if key == SHARED.POINT_STATE_ENABLED_CONFIG:
            return self.point_state_enabled
        return default


class StructuredDecisionInputContractTests(unittest.TestCase):
    def test_generic_contract_does_not_require_daily_news_extension(self) -> None:
        value = _generic_input()
        self.assertIsNone(value["domain_extension"])
        self.assertEqual(value["target_candidates"], [])
        for bridge in [*PROFILE_BRIDGES.values(), SHARED]:
            with self.subTest(bridge=bridge.__name__), patch.object(
                bridge, "_decision_input_now", return_value=FIXTURE_NOW
            ):
                self.assertEqual(bridge._validate_decision_input(value), value)

    def test_all_five_profile_producers_preserve_every_field_to_shared_jev_input(self) -> None:
        value = _daily_interest_input()
        for profile in PROFILE_PATHS:
            with self.subTest(profile=profile):
                event = _produce_event(profile, value)
                self.assertEqual(event["decision_input"], value)
                self.assertEqual(
                    event["decision_input_sha256"], SHARED._decision_input_digest(value)
                )
                for point_state_enabled in (False, True):
                    with tempfile.TemporaryDirectory(prefix="jev-contract-consumer-") as directory:
                        with patch.object(SHARED, "_decision_input_now", return_value=FIXTURE_NOW):
                            request = SHARED.build_jev_request(
                                event,
                                model="offline-contract-test",
                                point_state_enabled=point_state_enabled,
                            )
                            self.assertEqual(
                                json.loads(request["state"])["decision_input"], value
                            )
                            captured: list[dict[str, Any]] = []

                            def offline_post(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
                                captured.append(copy.deepcopy(dict(body)))
                                return {"offline": True}

                            consumer = SHARED.Consumer(
                                OfflineContext(point_state_enabled),
                                SHARED.BridgeStore(Path(directory)),
                            )
                            with patch.object(
                                SHARED,
                                "request_decision_with_fallback",
                                side_effect=offline_post,
                            ):
                                consumer._decision(event)
                            self.assertEqual(len(captured), 1)
                            self.assertEqual(
                                json.loads(captured[0]["state"])["decision_input"], value
                            )

    def test_strict_body_marker_accepts_only_complete_json_not_free_text_or_truncation(self) -> None:
        value = _daily_interest_input()
        body = "jev_decision_input_json: " + json.dumps(value, ensure_ascii=False)
        body += "\nscope_admission_json: {}"
        for bridge in [*PROFILE_BRIDGES.values(), SHARED]:
            with self.subTest(bridge=bridge.__name__), patch.object(
                bridge, "_decision_input_now", return_value=FIXTURE_NOW
            ):
                self.assertEqual(bridge._extract_decision_input(body), value)
                with self.assertRaises(bridge.TaskContractError):
                    bridge._extract_decision_input("scope: prose cannot substitute for the JSON contract")
                with self.assertRaises(bridge.TaskContractError):
                    bridge._extract_decision_input("jev_decision_input_json: {\"schema_version\":")
                too_large = "jev_decision_input_json: " + ("x" * (bridge.MAX_TASK_BODY_BYTES + 1))
                with self.assertRaises(bridge.TaskContractError):
                    bridge._extract_decision_input(too_large)

    def test_invalid_contracts_block_before_any_next_tool_for_each_profile(self) -> None:
        base = _daily_interest_input()
        invalid: dict[str, Any] = {
            "missing": None,
            "malformed_version": {**copy.deepcopy(base), "schema_version": 99},
        }
        incomplete = copy.deepcopy(base)
        incomplete.pop("objective")
        invalid["missing_required_value"] = incomplete
        stale = copy.deepcopy(base)
        stale["freshness"]["valid_until"] = "2026-09-25T03:59:59+09:00"
        invalid["stale"] = stale
        unresolved = copy.deepcopy(base)
        unresolved["domain_extension"]["unresolved_candidate_ids"] = ["fixture-chrome"]
        invalid["unresolved_reference"] = unresolved
        broken_hash = copy.deepcopy(base)
        for record in (broken_hash["target_candidates"][0], broken_hash["domain_extension"]["candidate_records"][0]):
            record["relevant_content"] += " tampered"
        invalid["content_integrity_failure"] = broken_hash
        oversized = copy.deepcopy(base)
        oversized["prior_context"]["oversized"] = "x" * 70_000
        invalid["oversized"] = oversized
        explicit_unknown = copy.deepcopy(base)
        explicit_unknown["uncertainty"] = ["unresolved: source identity"]
        invalid["unresolved_value"] = explicit_unknown

        for profile, bridge in PROFILE_BRIDGES.items():
            for name, decision_input in invalid.items():
                with self.subTest(profile=profile, case=name), tempfile.TemporaryDirectory(
                    prefix="jev-pretool-block-"
                ) as directory:
                    contract = _task_contract(bridge, profile, decision_input)
                    producer = bridge.Producer(
                        bridge.BridgeStore(Path(directory)),
                        profile,
                        contract_loader=lambda contract=contract: contract,
                    )
                    next_tool_calls: list[str] = []
                    with patch.object(bridge, "_decision_input_now", return_value=FIXTURE_NOW):
                        with patch.dict(
                            os.environ,
                            {
                                "HERMES_PROFILE": profile,
                                "HERMES_KANBAN_TASK": FIXTURE_TASK_ID,
                                "HERMES_KANBAN_RUN_ID": "9001",
                            },
                        ):
                            guard = producer.pre_tool_call(
                                tool_name="web_search",
                                args={"query": "must not execute"},
                            )
                            if guard is None:
                                next_tool_calls.append("web_search")
                    self.assertEqual(guard["action"], "block")
                    self.assertEqual(next_tool_calls, [])

    def test_shared_consumer_rejects_bad_event_before_reservation_or_provider(self) -> None:
        event = _produce_event("ops", _daily_interest_input())
        valid_input = copy.deepcopy(event["decision_input"])

        def missing_input(row: dict[str, Any]) -> None:
            row.pop("decision_input", None)
            row.pop("decision_input_sha256", None)

        def malformed_input(row: dict[str, Any]) -> None:
            row["decision_input"]["schema_version"] = 99
            row["decision_input_sha256"] = SHARED._decision_input_digest(row["decision_input"])

        def stale_input(row: dict[str, Any]) -> None:
            row["decision_input"]["freshness"]["valid_until"] = "2026-09-25T03:59:59+09:00"
            row["decision_input_sha256"] = SHARED._decision_input_digest(row["decision_input"])

        def unresolved_input(row: dict[str, Any]) -> None:
            row["decision_input"]["domain_extension"]["unresolved_candidate_ids"] = ["fixture-chrome"]
            row["decision_input_sha256"] = SHARED._decision_input_digest(row["decision_input"])

        def changed_referenced_content(row: dict[str, Any]) -> None:
            for record in (
                row["decision_input"]["target_candidates"][0],
                row["decision_input"]["domain_extension"]["candidate_records"][0],
            ):
                record["relevant_content"] += " tampered"
            row["decision_input_sha256"] = SHARED._decision_input_digest(row["decision_input"])

        def oversized_input(row: dict[str, Any]) -> None:
            row["decision_input"]["prior_context"]["oversized"] = "x" * 70_000
            row["decision_input_sha256"] = SHARED._decision_input_digest(row["decision_input"])

        mutations = {
            "missing": missing_input,
            "malformed": malformed_input,
            "stale": stale_input,
            "unresolved_reference": unresolved_input,
            "content_integrity": changed_referenced_content,
            "oversized": oversized_input,
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory(
                prefix="jev-consumer-block-"
            ) as directory:
                row = copy.deepcopy(event)
                row["state"] = "pending"
                mutate(row)
                store = SHARED.BridgeStore(Path(directory))
                store.append_event(row)
                decisions: list[dict[str, Any]] = []
                consumer = SHARED.Consumer(
                    OfflineContext(),
                    store,
                    decision_fn=lambda value: decisions.append(dict(value)) or {"label": "progressing"},
                )
                with patch.object(SHARED, "_decision_input_now", return_value=FIXTURE_NOW):
                    self.assertFalse(consumer.process_pending())
                self.assertEqual(decisions, [])
                self.assertEqual(store.ledger(), [])

        # The production request builder also rejects a bad integrity handle
        # before the provider wrapper can be reached.
        bad_event = copy.deepcopy(event)
        bad_event["decision_input"] = valid_input
        bad_event["decision_input_sha256"] = "0" * 64
        provider_calls: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="jev-provider-not-called-") as directory:
            consumer = SHARED.Consumer(OfflineContext(), SHARED.BridgeStore(Path(directory)))
            with patch.object(SHARED, "_decision_input_now", return_value=FIXTURE_NOW):
                with patch.object(
                    SHARED,
                    "request_decision_with_fallback",
                    side_effect=lambda body, **kwargs: provider_calls.append(dict(body)) or {},
                ):
                    with self.assertRaises(SHARED.BridgeValidationError):
                        consumer._decision(bad_event)
            self.assertEqual(provider_calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
