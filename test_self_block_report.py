"""Offline self-report exception tests; no production board or provider calls."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def load_bridge():
    name = 'jev_self_block_report_fixture'
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'bridge.py', submodule_search_locations=[str(ROOT)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Context:
    profile_name = 'ops'

    def __init__(self, root):
        self.config = {'bridge_dir': str(root), 'role': 'producer', 'producer_profiles': ['ops']}
        self.hooks = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)

    def register_tool(self, **kwargs):
        pass


class SelfBlockTests(unittest.TestCase):
    def setUp(self):
        self.bridge = load_bridge()
        tmp = self.enterContext(tempfile.TemporaryDirectory(prefix='jev-self-block-'))
        self.root = Path(tmp) / 'bridge'
        workspace = Path(tmp) / 'workspace'
        workspace.mkdir()
        self.store = self.bridge.BridgeStore(self.root)
        self.enterContext(patch.dict(os.environ, {
            'HERMES_PROFILE': 'ops', 'HERMES_KANBAN_TASK': 't_fixture',
            'HERMES_KANBAN_RUN_ID': '321', 'HERMES_KANBAN_WORKSPACE': str(workspace),
        }))
        self.enterContext(patch.object(self.bridge, '_trusted_current_board', return_value='default'))
        self.live = self.enterContext(patch.object(self.bridge, '_binding_matches_live_assignment', return_value=True))
        self.bridge.record_worker_binding(
            self.store, task_id='t_fixture', run_id='321', worker_pid=os.getpid(),
            worker_profile='ops', workspace_path=str(workspace), board='default',
            dispatcher_profile='default', assignee='ops',
        )
        ctx = Context(self.root)
        self.bridge.register(ctx)
        self.assertEqual(len(ctx.hooks['pre_tool_call']), 1)
        self.hook = ctx.hooks['pre_tool_call'][0]

    def stop(self):
        self.store.append_control({
            'schema_version': 1, 'control_id': 'ctl-fixture', 'dedupe_key': 'fixture',
            'task_id': 't_fixture', 'run_id': '321', 'checkpoint_id': 'cp-1',
            'trigger_id': 'trg-1', 'control': 'provisional_stop', 'state': 'active',
            'source': 'jev', 'reason': 'fixture', 'created_at': 'now',
        })

    def assert_denied(self, tool, args):
        self.assertEqual(self.hook(tool_name=tool, args=args)['action'], 'block')

    def test_own_native_report_preserves_stop(self):
        self.stop()
        before = self.store.controls()
        for args in ({'reason': 'need input'}, {'reason': 'need input', 'task_id': 't_fixture', 'board': 'default', 'kind': 'needs_input'}):
            self.assertIsNone(self.hook(tool_name='kanban_block', args=args))
        self.assertEqual(self.store.controls(), before)

    def test_other_tools_and_malformed_arguments_denied(self):
        self.stop()
        for tool in ('terminal', 'kanban_complete', 'kanban_request_review', 'kanban_comment', 'kanban_heartbeat'):
            with self.subTest(tool=tool):
                self.assert_denied(tool, {'reason': 'fixture'})
        for args in (
            {'reason': 'fixture', 'task_id': 'other'}, {'reason': 'fixture', 'board': 'other'},
            {'reason': 'fixture', 'run_id': '999'}, {'reason': 'fixture', 'kind': []},
            {'reason': 'fixture', 'extra': True}, {'reason': ''}, {'reason': None}, [], None,
        ):
            with self.subTest(args=args):
                self.assert_denied('kanban_block', args)
        self.assertEqual(len(self.store.controls()), 1)

    def test_stale_assignment_and_observer_denied(self):
        self.stop()
        self.live.return_value = False
        self.assert_denied('kanban_block', {'reason': 'fixture'})
        self.live.return_value = True
        with patch.object(self.bridge, '_load_current_worker_binding', return_value=None):
            self.assert_denied('kanban_block', {'reason': 'fixture'})
        with patch.object(self.bridge, '_load_current_worker_binding', return_value={
            'task_id': 't_fixture', 'run_id': '999', 'worker_profile': 'ops',
        }):
            self.assert_denied('kanban_block', {'reason': 'fixture'})

    def test_no_stop_does_not_read_strict_identity(self):
        with patch.object(self.bridge, '_load_current_worker_binding', side_effect=OSError('unavailable')):
            self.assertIsNone(self.hook(tool_name='terminal', args={'command': 'true'}))
        self.live.assert_not_called()

    def test_control_read_failure_denied(self):
        self.stop()
        with patch.object(self.bridge, '_active_provisional_stop', side_effect=OSError('fixture')):
            self.assert_denied('kanban_block', {'reason': 'fixture'})
        self.assertEqual(len(self.store.controls()), 1)


class AssignmentTests(unittest.TestCase):
    def test_live_row_discriminators(self):
        from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
        bridge = load_bridge()
        task = SimpleNamespace(status='running', current_run_id=321, assignee='ops')
        run = SimpleNamespace(task_id='t_fixture', status='running', profile='ops')
        latest = SimpleNamespace(id=321)
        binding = {'task_id': 't_fixture', 'run_id': '321', 'board': 'default'}

        @contextmanager
        def connection(*, board):
            self.assertEqual(board, 'default')
            yield object()

        with patch.object(kbc, 'connect_closing', connection), patch.object(kb, 'get_task', return_value=task), patch.object(kb, 'get_run', return_value=run), patch.object(kb, 'latest_run', return_value=latest):
            self.assertTrue(bridge._binding_matches_live_assignment(binding, 'ops'))
            for row, field, value in (
                (task, 'status', 'done'), (task, 'current_run_id', 999), (task, 'assignee', 'other'),
                (run, 'task_id', 'other'), (run, 'status', 'done'), (run, 'profile', 'other'),
                (latest, 'id', 999),
            ):
                with self.subTest(field=field, value=value):
                    original = getattr(row, field)
                    setattr(row, field, value)
                    self.assertFalse(bridge._binding_matches_live_assignment(binding, 'ops'))
                    setattr(row, field, original)


if __name__ == '__main__':
    unittest.main()
