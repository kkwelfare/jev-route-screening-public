"""Durable request-scoped upper goals, independent of local repair milestones.

This candidate deliberately has no model-authored approval path. Host-originated
messages are retained as source candidates; callers bind by source ID only.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

MAX_GOAL_BYTES = 8192
_LOCK = threading.RLock()


class GoalError(ValueError):
    pass


def _identity(value: Mapping[str, Any]) -> dict[str, str]:
    keys = ("session_id", "turn_id", "sender_id", "platform")
    identity = {key: str(value.get(key) or "").strip() for key in keys}
    if not all(identity.values()):
        raise GoalError("trusted host source identity is incomplete")
    return identity


class RequestGoalStore:
    """Append-only JSONL event store; projections merge later states by ID."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoalError("goal event log contains invalid JSON") from exc
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _append(self, row: dict[str, Any]) -> None:
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with _LOCK:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, encoded.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)

    def record_host_source(self, *, user_message: str, session_id: str,
                           turn_id: str, sender_id: str, platform: str) -> str:
        """Retain an actual host callback source; this does not create a root."""
        if not isinstance(user_message, str) or not user_message.strip():
            raise GoalError("host source message is empty or unavailable")
        if len(user_message.encode("utf-8")) > MAX_GOAL_BYTES:
            raise GoalError("host source exceeds 8KiB; source is unknown, not truncated")
        identity = _identity(locals())
        source_id = "src_" + uuid.uuid4().hex
        self._append({"kind": "source", "source_id": source_id,
                      "message": user_message, "identity": identity})
        return source_id

    def bind(self, *, source_id: str, root_id: str | None = None,
             completion_conditions: list[str] | None = None) -> dict[str, Any]:
        """Create an independent root from an already retained host source."""
        if not isinstance(source_id, str) or not source_id:
            raise GoalError("an existing host source ID is required")
        if completion_conditions is not None and (
            not isinstance(completion_conditions, list)
            or any(not isinstance(x, str) for x in completion_conditions)
        ):
            raise GoalError("completion_conditions must be a list of strings")
        with _LOCK:
            rows = self._rows()
            source = next((r for r in reversed(rows)
                          if r.get("kind") == "source" and r.get("source_id") == source_id), None)
            if not source:
                raise GoalError("host source ID was not retained")
            source_message = source.get("message")
            if (not isinstance(source_message, str) or not source_message.strip()
                    or len(source_message.encode("utf-8")) > MAX_GOAL_BYTES):
                raise GoalError("retained source is not a valid bounded goal (8KiB UTF-8 maximum)")
            if not source_message.startswith("new request:"):
                raise GoalError("a separate root requires an explicit host message beginning 'new request:'")
            message = source_message[len("new request:"):].strip()
            if not message:
                raise GoalError("explicit new request is empty")
            rid = root_id or "root_" + uuid.uuid4().hex
            if any(r.get("kind") == "root" and r.get("root_id") == rid for r in rows):
                raise GoalError("root ID already exists")
            identity = source["identity"]
            root = {"kind": "root", "root_id": rid, "version": 1,
                    "outcome": message, "completion_conditions": completion_conditions or [],
                    "source_message": source_message, "source_id": source_id,
                    "identity": identity, "local_milestones": []}
            self._append(root)
            self._append({"kind": "selection", "root_id": rid,
                          "identity": identity, "version": 1})
            return root

    def offer_initial(self, *, source_id: str, outcome: str, question: str,
                      completion_conditions: list[str] | None = None) -> dict[str, Any]:
        """Stage an initial root proposal; only a later host confirmation binds it."""
        for name, value in (("outcome", outcome), ("question", question)):
            if not isinstance(value, str) or not value.strip():
                raise GoalError(f"{name} is required")
            if len(value.encode("utf-8")) > MAX_GOAL_BYTES:
                raise GoalError(f"{name} exceeds 8KiB")
        if completion_conditions is not None and (
            not isinstance(completion_conditions, list)
            or any(not isinstance(x, str) for x in completion_conditions)
        ):
            raise GoalError("completion_conditions must be a list of strings")
        with _LOCK:
            rows = self._rows()
            source = next((r for r in reversed(rows)
                          if r.get("kind") == "source" and r.get("source_id") == source_id), None)
            if not source:
                raise GoalError("goal offer source was not retained from the host")
            identity = source.get("identity", {})
            if any(not identity.get(key) for key in ("session_id", "turn_id", "sender_id", "platform")):
                raise GoalError("goal offer source identity is incomplete")
            if any(r.get("kind") == "selection" and
                   r.get("identity", {}).get("session_id") == identity["session_id"] and
                   r.get("identity", {}).get("sender_id") == identity["sender_id"] and
                   r.get("identity", {}).get("platform") == identity["platform"] for r in rows):
                raise GoalError("a request-level root is already selected for this host session")
            pending = [r for r in rows if r.get("kind") == "initial_goal_offer"
                       and r.get("status") in {"pending_delivery", "offered"}
                       and r.get("identity", {}).get("session_id") == identity["session_id"]
                       and r.get("identity", {}).get("sender_id") == identity["sender_id"]
                       and r.get("identity", {}).get("platform") == identity["platform"]]
            if pending:
                raise GoalError("an initial goal proposal is already outstanding")
            offer_id = "offer_" + uuid.uuid4().hex
            row = {"kind": "initial_goal_offer", "offer_id": offer_id,
                   "status": "pending_delivery", "outcome": outcome.strip(),
                   "question": question.strip(),
                   "completion_conditions": completion_conditions or [],
                   "source_id": source_id, "identity": identity}
            self._append(row)
            return {"offer_id": offer_id, "question": row["question"],
                    "outcome": row["outcome"], "completion_conditions": row["completion_conditions"]}

    def mark_offer_delivered(self, *, offer_id: str, assistant_response: str,
                             session_id: str, turn_id: str) -> bool:
        """Mark only the exact proposal question found in its assistant turn."""
        rows = self._rows()
        offer = next((r for r in reversed(rows) if r.get("kind") == "initial_goal_offer"
                      and r.get("offer_id") == offer_id), None)
        if not offer or offer.get("status") != "pending_delivery":
            return False
        identity = offer.get("identity", {})
        if identity.get("session_id") != session_id or identity.get("turn_id") != turn_id:
            return False
        question = offer.get("question")
        if not isinstance(assistant_response, str) or not isinstance(question, str) or question not in assistant_response:
            self._append({"kind": "initial_goal_offer_state", "offer_id": offer_id,
                          "status": "not_delivered", "identity": identity})
            return False
        self._append({"kind": "initial_goal_offer_state", "offer_id": offer_id,
                      "status": "offered", "identity": identity,
                      "offered_turn_id": turn_id})
        return True

    @staticmethod
    def _positive_confirmation(message: str) -> bool:
        import re
        text = " ".join(message.strip().casefold().split())
        exact = {"ok", "okay", "yes", "y", "yep", "sure", "sounds good", "looks good",
                 "that works", "go ahead", "了解", "はい", "いいです", "それでお願いします", "その内容でお願いします"}
        return text in exact or bool(re.match(r"^(?:ok|okay|yes|yep|sure|了解|はい)[!,.、。 ]", text))

    def confirm_initial_from_host(self, *, source_id: str, user_message: str,
                                  session_id: str, turn_id: str, sender_id: str,
                                  platform: str) -> dict[str, Any] | None:
        """Bind a single outstanding delivered proposal on a later host turn."""
        identity = _identity(locals())
        if not self._positive_confirmation(user_message):
            return None
        with _LOCK:
            rows = self._rows()
            sources = {r.get("source_id"): r for r in rows if r.get("kind") == "source"}
            source = sources.get(source_id)
            if not source or source.get("message") != user_message:
                return None
            if source.get("identity") != identity:
                return None
            states: dict[str, dict[str, Any]] = {}
            for row in rows:
                if row.get("kind") == "initial_goal_offer":
                    states[row["offer_id"]] = row
                elif row.get("kind") == "initial_goal_offer_state":
                    states[row["offer_id"]] = {**states.get(row["offer_id"], {}), **row}
            candidates = [r for r in states.values() if r.get("status") == "offered"
                          and r.get("identity", {}).get("session_id") == session_id
                          and r.get("identity", {}).get("sender_id") == sender_id
                          and r.get("identity", {}).get("platform") == platform
                          and r.get("identity", {}).get("turn_id") != turn_id]
            if len(candidates) != 1:
                return None
            offer = candidates[0]
            root_id = "root_" + uuid.uuid4().hex
            if any(r.get("kind") == "selection" and
                   r.get("identity", {}).get("session_id") == session_id and
                   r.get("identity", {}).get("sender_id") == sender_id and
                   r.get("identity", {}).get("platform") == platform for r in rows):
                return None
            root_id = "root_" + uuid.uuid4().hex
            root = {"kind": "root", "root_id": root_id, "version": 1,
                    "outcome": offer["outcome"],
                    "completion_conditions": offer.get("completion_conditions", []),
                    "source_message": offer["question"], "source_id": offer["source_id"],
                    "identity": offer["identity"], "confirmed_by_source_id": source_id,
                    "confirmation_identity": identity, "local_milestones": []}
            self._append({"kind": "initial_goal_offer_state", "offer_id": offer["offer_id"],
                          "status": "confirmed", "confirmation_source_id": source_id,
                          "confirmation_identity": identity, "root_id": root_id})
            self._append(root)
            self._append({"kind": "selection", "root_id": root_id,
                          "identity": offer["identity"], "version": 1})
            return root

    def select_for_host(self, *, session_id: str, turn_id: str,
                        sender_id: str, platform: str) -> dict[str, Any] | None:
        """Return the durable root selected for this exact host owner/session."""
        identity = _identity(locals())
        rows = self._rows()
        selections = [row for row in rows if row.get("kind") == "selection"
                      and row.get("identity", {}).get("session_id") == identity["session_id"]
                      and row.get("identity", {}).get("sender_id") == identity["sender_id"]
                      and row.get("identity", {}).get("platform") == identity["platform"]]
        if not selections:
            return None
        selected = selections[-1]
        try:
            root = self.get(selected["root_id"])
        except (KeyError, GoalError):
            return None
        return root

    def select_for_session(self, session_id: str) -> dict[str, Any] | None:
        """Resolve a unique host-bound selection when a tool gets session only."""
        if not isinstance(session_id, str) or not session_id.strip():
            return None
        rows = self._rows()
        selections = [row for row in rows if row.get("kind") == "selection"
                      and row.get("identity", {}).get("session_id") == session_id]
        if not selections:
            return None
        selected = selections[-1]
        try:
            root = self.get(selected["root_id"])
        except (KeyError, GoalError):
            return None
        if root.get("identity", {}).get("session_id") != session_id:
            return None
        return root

    def add_local_milestone(self, *, root_id: str, milestone: str) -> None:
        if not isinstance(milestone, str) or not milestone.strip():
            raise GoalError("local milestone is required")
        with _LOCK:
            root = self.get(root_id)
            self._append({"kind": "local_milestone", "root_id": root_id,
                          "milestone": milestone, "version": root["version"]})

    def propose(self, *, root_id: str, version: int, proposed: str,
                reason: str, impact: str, source_id: str) -> dict[str, Any]:
        for name, value in (("proposed", proposed), ("reason", reason), ("impact", impact)):
            if not isinstance(value, str) or not value.strip():
                raise GoalError(f"{name} is required")
        root = self.get(root_id)
        if root["version"] != version:
            raise GoalError("proposal uses a stale root version")
        if len(proposed.encode("utf-8")) > MAX_GOAL_BYTES:
            raise GoalError("proposed goal exceeds 8KiB")
        rows = self._rows()
        source = next((r for r in reversed(rows)
                       if r.get("kind") == "source" and r.get("source_id") == source_id), None)
        if not source:
            raise GoalError("proposal source ID was not retained from the host")
        owner = root["identity"]
        identity = source["identity"]
        if any(identity[key] != owner.get(key) for key in ("session_id", "sender_id", "platform")):
            raise GoalError("proposal source does not belong to the root owner")
        proposal_id = "prop_" + uuid.uuid4().hex
        row = {"kind": "proposal", "proposal_id": proposal_id, "root_id": root_id,
               "base_version": version, "proposed": proposed, "reason": reason,
               "impact": impact, "status": "pending", "identity": identity,
               "source_id": source_id}
        self._append(row)
        return {"proposal_id": proposal_id, "root_id": root_id,
                "current": root["outcome"], "proposed": proposed,
                "reason": reason, "impact": impact,
                "confirmation": f"approve amendment {proposal_id}"}

    def approve_from_host(self, *, proposal_id: str, source_id: str, user_message: str,
                          session_id: str, turn_id: str, sender_id: str,
                          platform: str) -> dict[str, Any]:
        """Accept only exact proposal-specific confirmation from same host owner,
        in a later turn and same session/platform/sender.
        """
        identity = _identity(locals())
        with _LOCK:
            rows = self._rows()
            proposals: dict[str, dict[str, Any]] = {}
            for row in rows:
                if row.get("kind") == "proposal":
                    proposals[row["proposal_id"]] = row
                elif row.get("kind") == "settled":
                    proposals[row["proposal_id"]] = {**proposals.get(row["proposal_id"], {}), **row}
            proposal = proposals.get(proposal_id)
            if not proposal or proposal.get("status") != "pending":
                raise GoalError("proposal is absent, already settled, or not pending")
            owner = proposal.get("identity", {})
            if any(identity[key] != owner.get(key) for key in ("session_id", "sender_id", "platform")):
                raise GoalError("approval host identity does not match proposal owner")
            if turn_id == proposal.get("identity", {}).get("turn_id"):
                raise GoalError("approval must arrive in a later host turn than the proposal")
            proposal_index = next((i for i, row in enumerate(rows)
                if row.get("kind") == "proposal" and row.get("proposal_id") == proposal_id), -1)
            confirmation_index = next((i for i, row in enumerate(rows)
                if row.get("kind") == "source" and row.get("source_id") == source_id), -1)
            if proposal_index < 0 or confirmation_index <= proposal_index:
                raise GoalError("confirmation host source must be retained after the proposal")
            if user_message != f"approve amendment {proposal_id}":
                raise GoalError("host response is not the exact proposal-specific confirmation")
            confirmation_source = next((r for r in reversed(rows)
                if r.get("kind") == "source" and r.get("source_id") == source_id), None)
            if not confirmation_source or confirmation_source.get("message") != user_message:
                raise GoalError("confirmation source ID does not identify this retained host message")
            confirmation_identity = confirmation_source.get("identity", {})
            if any(identity[key] != confirmation_identity.get(key)
                   for key in ("session_id", "turn_id", "sender_id", "platform")):
                raise GoalError("confirmation source identity does not match host callback")
            root = self.get(proposal["root_id"])
            if root["version"] != proposal["base_version"]:
                raise GoalError("proposal is stale")
            self._append({"kind": "settled", "proposal_id": proposal_id,
                          "root_id": root["root_id"], "status": "approved",
                          "identity": identity, "turn_id": turn_id,
                          "source_id": source_id, "source_message": user_message})
            updated = {**root, "version": root["version"] + 1,
                       "outcome": proposal["proposed"],
                       "amended_from_proposal": proposal_id,
                       "amendment_source_id": source_id,
                       "amendment_source_message": user_message,
                       "amendment_identity": identity}
            self._append(updated)
            return updated

    def get(self, root_id: str) -> dict[str, Any]:
        rows = self._rows()
        roots = [r for r in rows if r.get("kind") == "root" and r.get("root_id") == root_id]
        if not roots:
            raise GoalError("root goal is unknown")
        root = dict(roots[0])
        milestones: list[str] = []
        for row in rows:
            if row.get("root_id") != root_id:
                continue
            if row.get("kind") == "root" and row.get("version", 0) > root.get("version", 0):
                root = dict(row)
            elif row.get("kind") == "local_milestone":
                milestones.append(row["milestone"])
        root["local_milestones"] = milestones
        return root

    def pending_proposals(self, root_id: str) -> list[dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for row in self._rows():
            if row.get("root_id") != root_id:
                continue
            if row.get("kind") == "proposal":
                states[row["proposal_id"]] = row
            elif row.get("kind") == "settled":
                states[row["proposal_id"]] = {**states.get(row["proposal_id"], {}), **row}
        return [r for r in states.values() if r.get("status") == "pending"]


class RequestGoalRoute:
    """Native tool/hook adapter. Approval authority comes only from host hook."""

    def __init__(self, store: RequestGoalStore):
        self.store = store

    def offer_initial_tool(self, args: Mapping[str, Any] | None = None,
                           session_id: str = "", turn_id: str = "", **_: Any) -> str:
        args = args if isinstance(args, Mapping) else {}
        try:
            source_id = args.get("source_id")
            source = next((r for r in reversed(self.store._rows())
                           if r.get("kind") == "source" and r.get("source_id") == source_id), None)
            identity = source.get("identity", {}) if source else {}
            if identity.get("session_id") != session_id or identity.get("turn_id") != turn_id:
                raise GoalError("goal proposal must reference a host message from this exact turn")
            if len([r for r in self.store._rows() if r.get("kind") == "source"
                    and r.get("identity", {}).get("session_id") == session_id]) < 2:
                raise GoalError("offer a root only after a few consultation turns")
            if not self._planning_transition(str(args.get("question", ""))):
                raise GoalError("offer is allowed only at a design/implementation transition")
            offered = self.store.offer_initial(source_id=source_id, outcome=args.get("outcome"),
                question=args.get("question"), completion_conditions=args.get("completion_conditions"))
            return json.dumps({"ok": True, **offered}, ensure_ascii=False)
        except (GoalError, TypeError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

    @staticmethod
    def _planning_transition(question: str) -> bool:
        cues = ("before i implement", "before implementation", "shall i build",
                "ready to implement", "implementation plan", "design proposal",
                "実装に進む", "設計案", "作業に入る", "この方針で進め")
        text = question.casefold()
        return any(cue in text for cue in cues)

    def post_llm_call(self, *, session_id: str = "", turn_id: str = "",
                      assistant_response: Any = None, **_: Any) -> None:
        if not isinstance(assistant_response, str):
            return
        rows = self.store._rows()
        states: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row.get("kind") == "initial_goal_offer":
                states[row["offer_id"]] = row
            elif row.get("kind") == "initial_goal_offer_state":
                states[row["offer_id"]] = {**states.get(row["offer_id"], {}), **row}
        for offer in reversed(list(states.values())):
            if offer.get("kind") != "initial_goal_offer" or offer.get("status") != "pending_delivery":
                continue
            identity = offer.get("identity", {})
            if identity.get("session_id") == session_id and identity.get("turn_id") == turn_id:
                self.store.mark_offer_delivered(offer_id=offer["offer_id"],
                    assistant_response=assistant_response, session_id=session_id, turn_id=turn_id)
                break

    def host_message(self, *, user_message: Any, session_id: str = "",
                     turn_id: str = "", sender_id: str = "", platform: str = "",
                     **_: Any) -> dict[str, str] | None:
        # Retain source candidates only; never infer a root from ordinary chat.
        if not all(isinstance(v, str) and v.strip() for v in
                   (session_id, turn_id, sender_id, platform)):
            return None
        try:
            source_id = self.store.record_host_source(user_message=user_message,
                session_id=session_id, turn_id=turn_id,
                sender_id=sender_id, platform=platform)
        except GoalError:
            return None
        try:
            confirmed = self.store.confirm_initial_from_host(source_id=source_id,
                user_message=user_message, session_id=session_id, turn_id=turn_id,
                sender_id=sender_id, platform=platform)
        except GoalError:
            confirmed = None
        if confirmed:
            return {"context": json.dumps({"request_goal": {
                "root_id": confirmed["root_id"], "version": confirmed["version"],
                "outcome": confirmed["outcome"],
                "completion_conditions": confirmed.get("completion_conditions", [])}}, ensure_ascii=False)}
        # Exact proposal-specific response is the only amendment path.
        if not isinstance(user_message, str):
            return None
        if user_message.strip().startswith("new request:"):
            try:
                root = self.store.bind(source_id=source_id)
            except GoalError:
                return None
            return {"context": json.dumps({"request_goal": {
                "root_id": root["root_id"], "version": root["version"],
                "outcome": root["outcome"],
                "completion_conditions": root.get("completion_conditions", [])}}, ensure_ascii=False)}
        pending = []
        for row in self.store._rows():
            if row.get("kind") == "proposal" and row.get("status") == "pending":
                pending.append(row)
        for proposal in reversed(pending):
            try:
                self.store.approve_from_host(proposal_id=proposal["proposal_id"], source_id=source_id,
                                    user_message=user_message, session_id=session_id, turn_id=turn_id,
                    sender_id=sender_id, platform=platform)
                return {"context": "The exact pending request-goal amendment was confirmed by the host user."}
            except GoalError:
                continue
        root = self.store.select_for_host(session_id=session_id, turn_id=turn_id,
            sender_id=sender_id, platform=platform)
        if root:
            return {"context": json.dumps({"request_goal": {
                "root_id": root["root_id"], "version": root["version"],
                "outcome": root["outcome"],
                "completion_conditions": root.get("completion_conditions", []),
                "guidance": "Keep this root fixed across local fixes. At a design/implementation transition after several consultation turns, call request_goal_offer_initial with this turn's host source_id and one concrete goal question; ask only if clarifying this outcome is useful. Never infer confirmation from tool/model text; only a later contextual host-user OK after the exact question was delivered binds it. Explicitly changed goals use the existing amendment proposal flow."}}, ensure_ascii=False)}
        return {"context": f"Host-origin source candidate available: source_id={source_id}. No request-level root is selected; ordinary messages do not create or replace one. After a few consultation turns, at a design/implementation transition only, offer one specific question with request_goal_offer_initial; contextual host-user OK to the delivered question binds it. Do not infer a goal from the first message or model-authored approval."}

    def bind_tool(self, args: Mapping[str, Any] | None = None, **host: Any) -> str:
        """Read back the host-selected root; model args never select or bind it."""
        session_id = host.get("session_id")
        root = self.store.select_for_session(session_id) if isinstance(session_id, str) else None
        if not root:
            return json.dumps({"ok": False, "error": "no host-selected request goal is bound"}, ensure_ascii=False)
        return json.dumps({"ok": True, "root_id": root["root_id"],
            "version": root["version"], "outcome": root["outcome"]}, ensure_ascii=False)

    def propose_tool(self, args: Mapping[str, Any] | None = None, **_: Any) -> str:
        args = args if isinstance(args, Mapping) else {}
        try:
            info = self.store.propose(root_id=args.get("root_id"), source_id=args.get("source_id"),
                version=args.get("version"), proposed=args.get("proposed"),
                reason=args.get("reason"), impact=args.get("impact"))
            return json.dumps({"ok": True, **info}, ensure_ascii=False)
        except (GoalError, TypeError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


def register_request_goal_route(ctx: Any, store: RequestGoalStore) -> RequestGoalRoute:
    """Register proposal/bind tools; deliberately register no approval tool."""
    route = RequestGoalRoute(store)
    ctx.register_tool(name="request_goal_offer_initial", toolset="jev_route_screening",
        schema={"name": "request_goal_offer_initial", "description":
            "At a design/implementation transition after several consultation turns, offer one concrete root-goal question. The exact question must be included in the assistant response; only a later contextual host-user OK confirms it. Never infer approval from model text.",
            "parameters": {"type": "object", "properties": {
                "source_id": {"type": "string", "description": "Host source ID from this exact turn."},
                "outcome": {"type": "string"}, "question": {"type": "string"},
                "completion_conditions": {"type": "array", "items": {"type": "string"}}},
                "required": ["source_id", "outcome", "question"], "additionalProperties": False}},
        handler=route.offer_initial_tool, description="Offer but do not bind or approve an initial request goal.")
    ctx.register_tool(name="request_goal_bind", toolset="jev_route_screening",
        schema={"name": "request_goal_bind", "description":
            "Read the request goal selected by the host callback for this session. Model arguments cannot select or replace a root.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
        handler=route.bind_tool, description="Read back the host-selected request-level goal.")
    ctx.register_tool(name="request_goal_propose_amendment", toolset="jev_route_screening",
        schema={"name": "request_goal_propose_amendment", "description":
            "Propose a request-goal amendment. Shows current/proposed/reason/impact and awaits exact host confirmation; does not approve.",
            "parameters": {"type": "object", "properties": {
                "root_id": {"type": "string"}, "version": {"type": "integer"},
                "source_id": {"type": "string", "description": "Retained host-origin source ID for this proposal turn."},
                "proposed": {"type": "string"}, "reason": {"type": "string"},
                "impact": {"type": "string"}}, "required":
                ["root_id", "version", "source_id", "proposed", "reason", "impact"],
                "additionalProperties": False}},
        handler=route.propose_tool, description="Propose but never approve a goal amendment.")
    ctx.register_hook("pre_llm_call", route.host_message)
    ctx.register_hook("post_llm_call", route.post_llm_call)
    return route
