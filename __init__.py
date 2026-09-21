"""Common worker-hook Jev bridge with advisory targeted skill reading."""

from __future__ import annotations

import os

from . import bridge as _bridge
from .skill_router import TargetedReadingAdvisor


def _profile_list(value):
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _record_worker_spawned_binding(ctx, **kwargs):
    """Bridge dispatcher lifecycle metadata to the profile-local worker hook."""
    dispatcher_profile = str(kwargs.get("profile_name") or "").strip()
    if dispatcher_profile not in {"", "default"}:
        return
    assignee = str(kwargs.get("assignee") or "").strip()
    producer_profiles = _profile_list(_bridge._config_value(ctx, "producer_profiles", [])) or {"ops"}
    if assignee not in producer_profiles:
        return
    root = _bridge._config_value(ctx, "bridge_dir", "") or os.environ.get("HERMES_JEV_BRIDGE_DIR", "")
    task_id = str(kwargs.get("task_id") or "").strip()
    run_id = kwargs.get("run_id")
    worker_pid = kwargs.get("worker_pid")
    workspace_path = str(kwargs.get("workspace_path") or "").strip()
    if not root or not task_id or run_id is None or isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0 or not workspace_path:
        return
    try:
        _bridge.record_worker_binding(
            _bridge.BridgeStore(root),
            task_id=task_id,
            run_id=run_id,
            worker_pid=worker_pid,
            worker_profile=assignee,
            workspace_path=workspace_path,
            board=str(kwargs.get("board") or "").strip(),
            dispatcher_profile=dispatcher_profile,
            assignee=assignee,
        )
    except Exception:
        _bridge.log.warning("Jev worker binding observer skipped metadata", exc_info=True)


def register(ctx):
    """Register the existing checkpoint bridge and default reading advisory."""
    _bridge.register(ctx)
    profile = str(getattr(ctx, "profile_name", "") or os.environ.get("HERMES_PROFILE", "")).strip()
    role = _bridge._config_value(ctx, "role", "")
    consumers = _profile_list(_bridge._config_value(ctx, "consumer_profiles", []))
    if not role and profile == "default":
        role, consumers = "consumer", {"default"}
    if role != "consumer" or profile != "default" or profile not in consumers:
        return
    root = _bridge._config_value(ctx, "bridge_dir", "") or os.environ.get("HERMES_JEV_BRIDGE_DIR", "")
    if not root:
        return
    ctx.register_hook("on_kanban_worker_spawned", lambda **kwargs: _record_worker_spawned_binding(ctx, **kwargs))
    advisor = TargetedReadingAdvisor(ctx, _bridge.BridgeStore(root), profile_name=profile)
    if not advisor.enabled:
        return
    ctx.register_hook("pre_llm_call", advisor.pre_llm_call)
    ctx.register_hook("post_tool_call", advisor.post_tool_call)
    ctx.register_hook("transform_tool_result", advisor.transform_tool_result)


__all__ = ["TargetedReadingAdvisor", "register"]
