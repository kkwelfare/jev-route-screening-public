"""Bounded, opt-in Jev point-state policy for the isolated candidate.

The module deliberately separates provider classification from local control.  A
point state is evidence about the current checkpoint, not authority to publish,
delete, authenticate, pay, or complete a task.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple


POINT_STATE_CONTRACT_VERSION = "point-state-v1"
CONFIDENCE_THRESHOLD = 0.60
DEFAULT_POINT_STATE_MODEL = "jev-latest"

POINT_STATES: Tuple[str, ...] = (
    "scope_or_authorization_blocked",
    "input_error_unresolved",
    "evidence_missing",
    "named_verification_ready",
    "ordinary_action_ready",
    "acceptance_ready",
    "unknown",
)

STATE_PRIORITY: Tuple[str, ...] = (
    "scope_or_authorization_blocked",
    "input_error_unresolved",
    "evidence_missing",
    "named_verification_ready",
    "acceptance_ready",
    "ordinary_action_ready",
    "unknown",
)

STATE_DEFINITIONS: Mapping[str, str] = {
    "scope_or_authorization_blocked": "The next required step is outside the approved scope or lacks required authorization.",
    "input_error_unresolved": "The latest bounded input or result contains an unresolved correction needed before progress.",
    "evidence_missing": "A bounded fact, artifact, or readback required by the current milestone is still missing.",
    "named_verification_ready": "The contract names a verification that is now ready and no broader action is implied.",
    "ordinary_action_ready": "The next ordinary action is inside the existing approved scope and can continue.",
    "acceptance_ready": "The implementation appears ready for default-owned review, not for automatic completion.",
    "unknown": "The current point cannot be classified reliably from the bounded snapshot.",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(snapshot)).encode("utf-8")).hexdigest()


def _finite_confidence(value: Any) -> Optional[float]:
    # bool is an int subclass, so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        return None
    return numeric


def _normalise_state(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value in POINT_STATES else None


def derive_exclusive_state(snapshot: Mapping[str, Any]) -> str:
    """Derive a deterministic state for a local fixture/mock.

    The order is intentionally exclusive.  A provider may classify a richer
    snapshot, but this helper makes the policy's precedence testable without a
    live call and never infers readiness from historical preparation alone.
    """

    if not isinstance(snapshot, Mapping):
        return "unknown"
    if snapshot.get("scope_or_authorization_blocked") or snapshot.get("authorization_required"):
        return "scope_or_authorization_blocked"
    if snapshot.get("input_error_unresolved") or snapshot.get("unresolved_input"):
        return "input_error_unresolved"
    if snapshot.get("evidence_missing") or snapshot.get("missing_evidence"):
        return "evidence_missing"
    if snapshot.get("named_verification_ready") or snapshot.get("next_named_verification"):
        return "named_verification_ready"
    if snapshot.get("acceptance_ready"):
        return "acceptance_ready"
    if snapshot.get("ordinary_action_ready") or snapshot.get("expected_next_action"):
        return "ordinary_action_ready"
    return "unknown"


def build_point_state_request(snapshot: Mapping[str, Any], *, model: str = DEFAULT_POINT_STATE_MODEL) -> Dict[str, Any]:
    """Build the frozen provider request shape.

    ``state`` is a JSON string by contract.  Criteria are explicit so the
    provider cannot silently invent a new action axis.
    """

    bounded_snapshot = {
        "goal": snapshot.get("goal"),
        "scope": snapshot.get("scope"),
        "next_named_milestone": snapshot.get("next_named_milestone"),
        "expected_next_action": snapshot.get("expected_next_action"),
        "latest_bounded_input": snapshot.get("latest_bounded_input"),
        "latest_bounded_result": snapshot.get("latest_bounded_result"),
        "refs": snapshot.get("refs", []),
    }
    criteria = {label: STATE_DEFINITIONS[label] for label in POINT_STATES}
    priority = " > ".join(STATE_PRIORITY)
    return {
        "model": model,
        "state": _canonical_json(bounded_snapshot),
        "questions": {
            "next_action_state": {
                "type": "choice",
                "instructions": f"Classify only the current bounded point and choose exactly one label. Apply this exclusive priority order: {priority}. Acceptance precedes ordinary action; do not grant authority or infer readiness from preparation advancing.",
                "criteria": criteria,
            }
        },
    }


@dataclass(frozen=True)
class PointDecision:
    state: Optional[str]
    confidence: Optional[float]
    accepted: bool
    issue: Optional[str] = None
    raw: Any = None
    provider_confidence: Any = None



def _candidate_pairs(response: Any) -> Sequence[Tuple[Any, Any]]:
    if isinstance(response, Mapping):
        answers = response.get("answers")
        if isinstance(answers, Mapping):
            point_answer = answers.get("next_action_state")
            if isinstance(point_answer, Mapping):
                return [(
                    point_answer.get("choice", point_answer.get("state", point_answer.get("label", point_answer.get("value")))),
                    point_answer.get("confidence"),
                )]
        if isinstance(answers, Sequence) and not isinstance(answers, (str, bytes)):
            pairs = []
            for answer in answers:
                if isinstance(answer, Mapping):
                    state = answer.get("state", answer.get("label", answer.get("value")))
                    confidence = answer.get("confidence")
                    pairs.append((state, confidence))
            if pairs:
                return pairs
        state = response.get("next_action_state", response.get("state", response.get("label")))
        confidence = response.get("confidence")
        return [(state, confidence)]
    if isinstance(response, Sequence) and not isinstance(response, (str, bytes)):
        pairs = []
        for answer in response:
            if isinstance(answer, Mapping):
                pairs.append((answer.get("state", answer.get("label", answer.get("value"))), answer.get("confidence")))
        return pairs
    return []


def parse_point_state_response(response: Any) -> PointDecision:
    """Parse a provider result without inventing reasoning or authority."""

    original = response
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except (TypeError, ValueError):
            return PointDecision(None, None, False, "malformed_response", original, None)
    pairs = _candidate_pairs(response)
    if not pairs:
        return PointDecision(None, None, False, "missing_state", original, None)

    states = [_normalise_state(state) for state, _ in pairs]
    if any(state is None for state in states):
        return PointDecision(None, None, False, "malformed_or_unknown_state", original, [confidence for _, confidence in pairs])
    distinct_states = {state for state in states if state is not None}
    if len(distinct_states) != 1:
        return PointDecision(None, None, False, "conflicting_state", original, [state for state, _ in pairs])
    state = next(iter(distinct_states))

    confidences = [_finite_confidence(confidence) for _, confidence in pairs]
    if any(confidence is None for confidence in confidences):
        return PointDecision(state, None, False, "missing_or_invalid_confidence", original, [confidence for _, confidence in pairs])
    confidence = min(confidence for confidence in confidences if confidence is not None)
    if state == "unknown":
        return PointDecision(state, confidence, False, "unknown_state", original, confidences)
    if confidence < CONFIDENCE_THRESHOLD:
        return PointDecision(state, confidence, False, "low_confidence", original, confidences)
    return PointDecision(state, confidence, True, None, original, confidences)


@dataclass(frozen=True)
class PointIdentity:
    task_id: str
    run_id: str
    event_id: str
    contract_version: str
    snapshot_hash: str

    @classmethod
    def from_snapshot(
        cls,
        task_id: str,
        run_id: str,
        event_id: str,
        snapshot: Mapping[str, Any],
        contract_version: str = POINT_STATE_CONTRACT_VERSION,
    ) -> "PointIdentity":
        return cls(str(task_id), str(run_id), str(event_id), str(contract_version), snapshot_hash(snapshot))

    def stream_key(self, issue: str, milestone: str) -> str:
        return "/".join((self.task_id, self.run_id, issue, milestone))

    def action_key(self, issue: str, milestone: str, action: str) -> str:
        return "/".join((self.task_id, self.run_id, issue, milestone, self.snapshot_hash, action))


@dataclass(frozen=True)
class ControlEffect:
    action: str
    applied: bool
    reason: str
    identity: PointIdentity
    instructions: Tuple[str, ...] = ()
    verification_name: Optional[str] = None
    completion_allowed: bool = False
    guidance_only: bool = True


class PointStatePolicy:
    """Stateful bounded policy with stable issue/milestone budgets.

    ``store`` may be a dict-like object supplied by the host's existing isolated
    store.  A JSON file path is supported for tests and candidate replay only;
    this class never executes a command or performs an external send.
    """

    def __init__(self, store: Optional[MutableMapping[str, Any]] = None, store_path: Optional[str | Path] = None, max_corrections: int = 1, max_refreshes: int = 1) -> None:
        self._store: MutableMapping[str, Any] = store if store is not None else {}
        self._store_path = Path(store_path) if store_path else None
        self.max_corrections = max_corrections
        self.max_refreshes = max_refreshes
        self._load()

    def _load(self) -> None:
        if self._store_path is None or not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, Mapping):
            self._store.update(data)

    def _persist(self) -> None:
        if self._store_path is None:
            return
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(_canonical_json(dict(self._store)) + "\n", encoding="utf-8")

    def _event_is_stale(self, stream_key: str, event_id: str) -> bool:
        latest = self._store.get("latest_event", {}).get(stream_key)
        if latest is None:
            return False
        try:
            return int(event_id) < int(latest)
        except (TypeError, ValueError):
            return event_id != latest and str(event_id) < str(latest)

    def _remember_event(self, stream_key: str, event_id: str) -> None:
        latest_events = self._store.setdefault("latest_event", {})
        latest = latest_events.get(stream_key)
        if latest is None:
            latest_events[stream_key] = event_id
            return
        try:
            if int(event_id) > int(latest):
                latest_events[stream_key] = event_id
        except (TypeError, ValueError):
            if str(event_id) > str(latest):
                latest_events[stream_key] = event_id

    def _budget(self, budget_key: str) -> int:
        return int(self._store.setdefault("budgets", {}).get(budget_key, 0))

    def _consume(self, budget_key: str) -> None:
        budgets = self._store.setdefault("budgets", {})
        budgets[budget_key] = self._budget(budget_key) + 1

    def evaluate(
        self,
        decision: PointDecision,
        identity: PointIdentity,
        *,
        issue: str,
        milestone: str,
        snapshot: Mapping[str, Any],
        guards: Iterable[str] = (),
    ) -> ControlEffect:
        stream_key = identity.stream_key(issue, milestone)
        if not identity.task_id or not identity.run_id or not identity.event_id or not identity.contract_version:
            return ControlEffect("review", False, "invalid_identity", identity)
        if identity.contract_version != POINT_STATE_CONTRACT_VERSION:
            return ControlEffect("reject_stale", False, "stale_contract", identity)
        if self._event_is_stale(stream_key, identity.event_id):
            return ControlEffect("reject_stale", False, "stale_event", identity)

        guards = tuple(str(guard) for guard in guards if guard)
        if guards:
            effect = ControlEffect("review", True, "guard_precedence", identity, instructions=guards)
            self._remember_event(stream_key, identity.event_id)
            self._store.setdefault("actions", {})[identity.action_key(issue, milestone, "review")] = effect.reason
            self._persist()
            return effect

        if not decision.accepted:
            action = "refresh_evidence_then_review"
            action_key = identity.action_key(issue, milestone, action)
            if action_key in self._store.setdefault("actions", {}):
                return ControlEffect("deduplicated", False, "duplicate_snapshot_action", identity)
            budget_key = "/".join((identity.task_id, identity.run_id, issue, milestone, "refresh"))
            if self._budget(budget_key) >= self.max_refreshes:
                effect = ControlEffect("review", True, "refresh_budget_exhausted", identity)
            else:
                self._consume(budget_key)
                effect = ControlEffect(action, True, decision.issue or "untrusted_point_state", identity, instructions=("Perform one bounded refresh of the named evidence, then return to default review; do not execute an arbitrary command.",))
            self._remember_event(stream_key, identity.event_id)
            self._store.setdefault("actions", {})[action_key] = effect.reason
            self._persist()
            return effect

        state = decision.state
        if state is None:
            return ControlEffect("review", False, "missing_state", identity)

        if state in {"scope_or_authorization_blocked", "acceptance_ready"}:
            action = "review"
            reason = "scope_or_authorization_blocked" if state == "scope_or_authorization_blocked" else "default_acceptance_review"
            effect = ControlEffect(action, True, reason, identity, completion_allowed=False)
        elif state == "input_error_unresolved":
            action = "correct_input"
            action_key = identity.action_key(issue, milestone, action)
            if action_key in self._store.setdefault("actions", {}):
                return ControlEffect("deduplicated", False, "duplicate_snapshot_action", identity)
            budget_key = "/".join((identity.task_id, identity.run_id, issue, milestone, "correction"))
            if self._budget(budget_key) >= self.max_corrections:
                effect = ControlEffect("review", True, "correction_budget_exhausted", identity)
            else:
                self._consume(budget_key)
                instruction = str(snapshot.get("correction_instruction") or snapshot.get("expected_next_action") or "Revalidate the actual bounded input against the current contract, then request default review.")
                effect = ControlEffect(action, True, "first_unrepaired_error", identity, instructions=(instruction,))
        elif state == "evidence_missing":
            action = "refresh_evidence_then_review"
            action_key = identity.action_key(issue, milestone, action)
            if action_key in self._store.setdefault("actions", {}):
                return ControlEffect("deduplicated", False, "duplicate_snapshot_action", identity)
            budget_key = "/".join((identity.task_id, identity.run_id, issue, milestone, "refresh"))
            if self._budget(budget_key) >= self.max_refreshes:
                effect = ControlEffect("review", True, "refresh_budget_exhausted", identity)
            else:
                self._consume(budget_key)
                effect = ControlEffect(action, True, "evidence_missing", identity, instructions=("Refresh only the contract-named evidence once, then return to default review.",))
        elif state == "named_verification_ready":
            verification_name = snapshot.get("next_named_verification") or snapshot.get("named_verification")
            if not isinstance(verification_name, str) or not verification_name.strip():
                effect = ControlEffect("review", True, "named_verification_missing", identity)
            else:
                effect = ControlEffect("run_named_verification", True, "named_verification_ready", identity, verification_name=verification_name.strip())
        elif state == "ordinary_action_ready":
            next_action = snapshot.get("expected_next_action")
            if not isinstance(next_action, str) or not next_action.strip():
                effect = ControlEffect("review", True, "ordinary_action_missing", identity)
            else:
                effect = ControlEffect("continue_scoped_action", True, "ordinary_action_ready", identity, instructions=(next_action.strip(),))
        else:
            effect = ControlEffect("review", True, "unsupported_state", identity)

        action_key = identity.action_key(issue, milestone, effect.action)
        if action_key in self._store.setdefault("actions", {}):
            return ControlEffect("deduplicated", False, "duplicate_snapshot_action", identity)
        self._remember_event(stream_key, identity.event_id)
        self._store["actions"][action_key] = effect.reason
        self._persist()
        return effect


def control_from_provider(
    response: Any,
    *,
    task_id: str,
    run_id: str,
    event_id: str,
    snapshot: Mapping[str, Any],
    issue: str,
    milestone: str,
    policy: PointStatePolicy,
    guards: Iterable[str] = (),
) -> Tuple[PointDecision, ControlEffect]:
    decision = parse_point_state_response(response)
    identity = PointIdentity.from_snapshot(task_id, run_id, event_id, snapshot)
    return decision, policy.evaluate(decision, identity, issue=issue, milestone=milestone, snapshot=snapshot, guards=guards)
