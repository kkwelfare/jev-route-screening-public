"""Offline producer→consumer contract tests; never call Jev providers or Gateway."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

DEFAULT = Path(__file__).resolve().parent
RESEARCH = DEFAULT

def load(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

consumer = load('jev_discovery_test_consumer', DEFAULT / '__init__.py', True)
cb = sys.modules['jev_discovery_test_consumer.bridge']
ps = sys.modules['jev_discovery_test_consumer.point_state']
rb = cb
CONTEXT = {'phase': 'before_first_search', 'next_action': 'web_search', 'completion_requested': False, 'provenance': 'dispatcher_contract_and_worker_hook'}
POST_CONTEXT = {'phase': 'after_search_source_check', 'next_action': 'web_extract', 'completion_requested': False, 'provenance': 'dispatcher_contract_and_worker_hook'}

def generic_decision_input(task_id):
    return {
        'schema_version': rb.DECISION_INPUT_SCHEMA_VERSION,
        'scope': 'offline research discovery fixture',
        'objective': 'preserve the worker decision contract',
        'target_candidates': [],
        'requested_outcome': 'bounded advisory route only',
        'action_mode': 'read_only',
        'allowed_actions': ['read bounded task evidence'],
        'forbidden_actions': ['external posting', 'authority changes'],
        'decision_points': [],
        'freshness': {'status': 'fresh', 'observed_at': '2026-01-01T00:00:00Z',
                      'valid_until': '2099-01-01T00:00:00Z', 'rule': 'static offline fixture window'},
        'deduplication': {'status': 'not_applicable', 'basis': 'unique integration fixture'},
        'baseline': {'status': 'defined', 'value': 'offline'},
        'completion_conditions': ['the worker event preserves the contract'],
        'prior_context': {'summary': 'fixture only'},
        'signals': [],
        'feedback': [],
        'output_requirements': {'format': 'machine-readable advisory'},
        'uncertainty': [],
        'provenance': {'task_id': task_id, 'source': 'offline_test_fixture',
                       'captured_at': '2026-01-01T00:00:00Z',
                       'source_references': [f'kanban://task/{task_id}']},
        'domain_extension': None,
    }

class ResearchDiscoveryTests(unittest.TestCase):
    def make_contract(self, authorized=True, extract_authorized=True):
        return rb.TaskContract(
            task_id='research-task', run_id='901', profile='research', started_at=1000.0,
            original_request='Collect candidates',
            criteria=('native VLR succeeds and worker invokes direct web_search', 'return candidate refs'),
            current_unknown='candidates unknown', next_action='collect sources',
            evidence_refs=('kanban://task/research-task',),
            search_authorized=authorized, extract_authorized=extract_authorized,
            decision_input=generic_decision_input('research-task'),
        )

    def produce(self, authorized=True, first_search=False, completion=False, extract_authorized=True):
        with tempfile.TemporaryDirectory(prefix='jev-discovery-unit-') as directory:
            store = rb.BridgeStore(Path(directory))
            contract = self.make_contract(authorized, extract_authorized)
            producer = rb.Producer(store, 'research', contract_loader=lambda: contract, clock=lambda: 1300.0, cadence_seconds=0)
            with patch.dict(os.environ, {'HERMES_KANBAN_TASK': contract.task_id, 'HERMES_KANBAN_RUN_ID': contract.run_id, 'HERMES_PROFILE': 'research'}):
                for tool in (('web_search' if first_search else 'read_file'),
                             ('kanban_complete' if completion else 'read_file'), 'write_file'):
                    producer.post_tool_call(tool_name=tool, args={'path': 'candidate.json'}, result={'ok': True, 'changed': True}, status='ok')
            self.assertEqual(len(store.events()), 1)
            return store.events()[0]

    def run_integrated(self, directory, response, *, first_search=False, contract=None):
        class Context:
            def get_config(self, key, default=None):
                return True if key == cb.POINT_STATE_ENABLED_CONFIG else default

        contract = contract or self.make_contract()
        root = Path(directory)
        producer = rb.Producer(
            rb.BridgeStore(root),
            'research',
            contract_loader=lambda: contract,
            clock=lambda: 1300.0,
            cadence_seconds=0,
            bridge_root=root,
        )
        with patch.dict(os.environ, {'HERMES_KANBAN_TASK': contract.task_id, 'HERMES_KANBAN_RUN_ID': contract.run_id, 'HERMES_PROFILE': 'research'}):
            for tool in (('web_search' if first_search else 'read_file'), 'read_file', 'write_file'):
                producer.post_tool_call(tool_name=tool, args={'path': 'candidate.json'}, result={'ok': True, 'changed': True}, status='ok')
            store = cb.BridgeStore(root)
            consumer = cb.Consumer(Context(), store, decision_fn=(response if callable(response) else lambda _: dict(response)))
            self.assertTrue(consumer.process_pending())
            pre_tool_result = producer.pre_tool_call('read_file')
        return producer, consumer, store, pre_tool_result

    def test_authorized_pre_search_is_forwarded_and_does_not_stop_for_final_evidence(self):
        event = self.produce()
        self.assertEqual(event['discovery_context'], CONTEXT)
        self.assertTrue(cb._event_projection_is_eligible(event))
        snapshot = cb._point_snapshot_from_event(event)
        self.assertEqual(snapshot['discovery_context'], CONTEXT)
        request = ps.build_point_state_request(snapshot)
        self.assertEqual(json.loads(request['state'])['discovery_context'], CONTEXT)
        self.assertIn('missing final candidate evidence is not yet', request['questions']['next_action_state']['instructions'])
        identity = ps.PointIdentity.from_snapshot(event['task_id'], event['run_id'], event['event_key'], snapshot)
        policy = ps.PointStatePolicy()
        missing = ps.PointDecision('evidence_missing', 0.95, True, None, {}, [0.95])
        effect = policy.evaluate(missing, identity, issue='worker_checkpoint', milestone='collect', snapshot=snapshot)
        self.assertEqual(effect.action, 'continue_scoped_action')
        self.assertFalse(effect.completion_allowed)
        self.assertIsNone(cb.Consumer._point_state_trigger_category('evidence_missing', effect))
        scope_signal = ps.PointDecision('scope_or_authorization_blocked', 0.95, True, None, {}, [0.95])
        scope_effect = ps.PointStatePolicy().evaluate(scope_signal, identity, issue='worker_checkpoint', milestone='collect', snapshot=snapshot)
        self.assertEqual(scope_effect.action, 'continue_scoped_action')
        self.assertIsNone(cb.Consumer._point_state_trigger_category(scope_signal.state, scope_effect))
        guarded = ps.PointStatePolicy().evaluate(missing, identity, issue='worker_checkpoint', milestone='collect', snapshot=snapshot, guards=('no_search_permission',))
        self.assertNotEqual(guarded.action, 'continue_scoped_action')

    def test_post_search_source_check_survives_low_confidence_without_weakening_other_stops(self):
        event = self.produce(first_search=True)
        self.assertEqual(event['discovery_context'], POST_CONTEXT)
        self.assertTrue(cb._event_projection_is_eligible(event))
        snapshot = cb._point_snapshot_from_event(event)
        self.assertEqual(snapshot['expected_next_action'], 'web_extract')
        self.assertEqual(json.loads(ps.build_point_state_request(snapshot)['state'])['discovery_context'], POST_CONTEXT)
        identity = ps.PointIdentity.from_snapshot(event['task_id'], event['run_id'], event['event_key'], snapshot)
        for decision in (ps.PointDecision('evidence_missing', 0.95, True),):
            with self.subTest(decision=decision):
                effect = ps.PointStatePolicy().evaluate(decision, identity, issue='worker_checkpoint', milestone='collect', snapshot=snapshot)
                self.assertEqual(effect.action, 'continue_scoped_action')
                self.assertFalse(effect.completion_allowed)
                self.assertIsNone(cb.Consumer._point_state_trigger_category('evidence_missing', effect))

        # A low-confidence classification is advisory even when its category
        # names an authorization/input issue: the label itself is not an
        # independently verified guard. It never authorizes completion.
        ordinary_snapshot = dict(snapshot)
        ordinary_snapshot.pop('discovery_context')
        ordinary_snapshot['expected_next_action'] = 'authorized in-scope action'
        for state in ('scope_or_authorization_blocked', 'input_error_unresolved', 'evidence_missing',
                      'named_verification_ready', 'ordinary_action_ready', 'acceptance_ready'):
            decision = ps.PointDecision(state, 0.4, False, 'low_confidence')
            with self.subTest(decision=decision):
                policy = ps.PointStatePolicy()
                effect = policy.evaluate(
                    decision,
                    identity,
                    issue='worker_checkpoint',
                    milestone='collect',
                    snapshot=ordinary_snapshot,
                )
                self.assertEqual(effect.action, 'continue_scoped_action')
                self.assertEqual(effect.reason, 'low_confidence_advisory')
                self.assertFalse(effect.completion_allowed)
                self.assertIsNone(cb.Consumer._point_state_trigger_category(state, effect))
                replay = policy.evaluate(
                    decision,
                    identity,
                    issue='worker_checkpoint',
                    milestone='collect',
                    snapshot=ordinary_snapshot,
                )
                self.assertEqual(replay.action, 'continue_scoped_action')
                self.assertFalse(replay.completion_allowed)
                self.assertIsNone(cb.Consumer._point_state_trigger_category(state, replay))

        # Separately supplied guards retain precedence over the classifier.
        low_confidence = ps.PointDecision('scope_or_authorization_blocked', 0.4, False, 'low_confidence')
        for guard in ('authorization_missing', 'current_input_error', 'explicit_hold'):
            with self.subTest(guard=guard):
                effect = ps.PointStatePolicy().evaluate(
                    low_confidence,
                    identity,
                    issue='worker_checkpoint',
                    milestone='collect',
                    snapshot=snapshot,
                    guards=(guard,),
                )
                self.assertEqual(effect.action, 'review')
                self.assertEqual(effect.reason, 'guard_precedence')
                self.assertIsNotNone(cb.Consumer._point_state_trigger_category(low_confidence.state, effect))

        # Classifier states are signals, not independent stop evidence.
        for state in ('scope_or_authorization_blocked', 'input_error_unresolved',
                      'named_verification_ready', 'ordinary_action_ready'):
            decision = ps.PointDecision(state, 0.95, True, None, {}, [0.95])
            with self.subTest(decision=decision):
                effect = ps.PointStatePolicy().evaluate(decision, identity, issue='worker_checkpoint', milestone='collect', snapshot=snapshot)
                self.assertIsNone(cb.Consumer._point_state_trigger_category(state, effect))
        # Missing/invalid confidence and malformed output remain advisory.
        for decision in (ps.PointDecision('evidence_missing', None, False, 'missing_or_invalid_confidence'),
                         ps.PointDecision('evidence_missing', 0.4, False, 'malformed_response')):
            with self.subTest(decision=decision):
                effect = ps.PointStatePolicy().evaluate(decision, identity, issue='worker_checkpoint', milestone='collect', snapshot=snapshot)
                self.assertEqual(effect.action, 'continue_scoped_action')
                self.assertFalse(effect.completion_allowed)
                self.assertIsNone(cb.Consumer._point_state_trigger_category('evidence_missing', effect))

    def test_low_confidence_integrated_path_is_advisory_not_acceptance_or_stop(self):
        with tempfile.TemporaryDirectory(prefix='jev-low-confidence-integrated-') as directory:
            _, _, store, pre_tool_result = self.run_integrated(
                directory,
                {'point_state': 'evidence_missing', 'confidence': 0.4},
            )
            review = store.reviews()[-1]
            self.assertEqual(review['point_state_issue'], 'low_confidence')
            self.assertEqual(review['point_state_action'], 'continue_scoped_action')
            self.assertEqual(review['point_state_reason'], 'low_confidence_advisory')
            self.assertFalse(review['point_state_accepted'])
            self.assertFalse(review['completion_allowed'])
            self.assertIsNone(review['trigger_id'])
            self.assertFalse(store.triggers())
            self.assertFalse(store.controls())
            self.assertIsNone(pre_tool_result)

    def test_low_confidence_authorization_and_input_labels_are_advisory_not_guards(self):
        for state in ('scope_or_authorization_blocked', 'input_error_unresolved'):
            with self.subTest(state=state), tempfile.TemporaryDirectory(prefix='jev-low-confidence-label-') as directory:
                _, _, store, pre_tool_result = self.run_integrated(
                    directory,
                    {'point_state': state, 'confidence': 0.4},
                )
                review = store.reviews()[-1]
                self.assertEqual(review['point_state_issue'], 'low_confidence')
                self.assertEqual(review['point_state_action'], 'continue_scoped_action')
                self.assertEqual(review['point_state_reason'], 'low_confidence_advisory')
                self.assertFalse(review['point_state_accepted'])
                self.assertFalse(review['completion_allowed'])
                self.assertFalse(store.triggers())
                self.assertFalse(store.controls())
                self.assertIsNone(pre_tool_result)

    def test_no_discovery_context_still_allows_evidence_gathering(self):
        for case in ({'authorized': False}, {'first_search': True, 'extract_authorized': False}, {'completion': True}):
            with self.subTest(case=case):
                event = self.produce(**case)
                self.assertNotIn('discovery_context', event)
                snapshot = cb._point_snapshot_from_event(event)
                self.assertNotIn('discovery_context', snapshot)
                identity = ps.PointIdentity.from_snapshot(event['task_id'], event['run_id'], event['event_key'], snapshot)
                missing = ps.PointDecision('evidence_missing', 0.95, True, None, {}, [0.95])
                effect = ps.PointStatePolicy().evaluate(missing, identity, issue='worker_checkpoint', milestone='collect', snapshot=snapshot)
                self.assertEqual(effect.action, 'continue_scoped_action')
                self.assertFalse(effect.completion_allowed)
                self.assertIsNone(cb.Consumer._point_state_trigger_category('evidence_missing', effect))

    def test_manual_or_wrong_profile_cannot_assert_discovery_context(self):
        event = self.produce()
        for field, value in (('source', 'manual'), ('profile', 'ops'), ('synthetic', True)):
            with self.subTest(field=field):
                malformed = dict(event, **{field: value})
                self.assertFalse(cb._event_projection_is_eligible(malformed))
        for context in ({**CONTEXT, 'completion_requested': True}, {**CONTEXT, 'phase': 'after_search_source_check'},
                        {**POST_CONTEXT, 'next_action': 'kanban_complete'}):
            self.assertFalse(cb._event_projection_is_eligible(dict(event, discovery_context=context)))

    def test_post_search_checkpoint_requires_observed_search_or_prior_hook_history(self):
        event = self.produce(first_search=True)
        without_search = dict(event, recent_steps=[{'action': 'write_file', 'result': 'ok', 'evidence_changed': True, 'evidence_refs': []}], completed_tools=1)
        self.assertFalse(cb._event_projection_is_eligible(without_search))
        completion = dict(event, recent_steps=event['recent_steps'] + [{'action': 'kanban_complete', 'result': 'ok', 'evidence_changed': False, 'evidence_refs': []}])
        self.assertFalse(cb._event_projection_is_eligible(completion))
        # A six-step recent window may omit a search observed earlier in the
        # same complete producer-hook history. Reattachment does not emit this.
        earlier_search = dict(without_search, completed_tools=17)
        self.assertTrue(cb._event_projection_is_eligible(earlier_search))

    def test_reattached_producer_does_not_invent_search_history(self):
        with tempfile.TemporaryDirectory(prefix='jev-research-reattach-') as directory:
            store = rb.BridgeStore(Path(directory))
            contract = self.make_contract()
            store.append_event(self.produce(first_search=True))
            # The persisted event belongs to the same trusted run, but a new
            # Producer has not observed all earlier tool callbacks itself.
            producer = rb.Producer(store, 'research', contract_loader=lambda: contract, clock=lambda: 1400.0, cadence_seconds=0)
            with patch.dict(os.environ, {'HERMES_KANBAN_TASK': contract.task_id, 'HERMES_KANBAN_RUN_ID': contract.run_id, 'HERMES_PROFILE': 'research'}):
                state = producer._state(contract)
                self.assertFalse(state['history_complete'])
                state['search_seen'] = True
                state['total_tools'] += 3
                state['tools_since'] = 3
                state['steps'] = [{'action': action, 'result': 'success=true', 'evidence_changed': False, 'evidence_refs': []}
                                  for action in ('web_search', 'read_file', 'write_file')]
                self.assertNotIn('discovery_context', producer._build_event(contract, state, 1450.0))

    def test_fresh_research_registration_exposes_producer_hooks(self):
        class Context:
            profile_name = 'research'
            def __init__(self, root):
                self.root, self.tools, self.hooks = root, {}, {}
            def get_config(self, key, default=None):
                return {'bridge_dir': str(self.root), 'role': 'producer', 'producer_profiles': ['research'],
                        'consumer_profiles': ['default']}.get(key, default)
            def register_tool(self, **kwargs):
                self.tools[kwargs['name']] = kwargs
            def register_hook(self, name, callback):
                self.hooks.setdefault(name, []).append(callback)
        with tempfile.TemporaryDirectory(prefix='jev-research-registration-') as directory:
            context = Context(Path(directory))
            rb.register(context)
            self.assertIn('jev_bridge_checkpoint', context.tools)
            self.assertEqual(len(context.hooks['pre_tool_call']), 1)
            self.assertEqual(len(context.hooks['post_tool_call']), 1)

    def test_offline_consumer_records_continue_without_provisional_stop(self):
        class Context:
            def get_config(self, key, default=None):
                return True if key == cb.POINT_STATE_ENABLED_CONFIG else default

        for state in ('evidence_missing', 'scope_or_authorization_blocked'):
            with self.subTest(state=state), tempfile.TemporaryDirectory(prefix='jev-consumer-discovery-') as directory:
                event = self.produce()
                store = cb.BridgeStore(Path(directory))
                store.append_event(event)
                consumer = cb.Consumer(Context(), store, decision_fn=lambda _: {'point_state': state, 'confidence': 0.95})
                self.assertTrue(consumer.process_pending())
                self.assertFalse(consumer.process_pending())
                self.assertFalse(store.triggers())
                self.assertFalse(store.controls())
                self.assertEqual(store.reviews()[-1]['point_state_action'], 'continue_scoped_action')

    def test_offline_consumer_post_search_low_confidence_records_no_stop(self):
        class Context:
            def get_config(self, key, default=None):
                return True if key == cb.POINT_STATE_ENABLED_CONFIG else default

        with tempfile.TemporaryDirectory(prefix='jev-consumer-source-check-') as directory:
            store = cb.BridgeStore(Path(directory))
            store.append_event(self.produce(first_search=True))
            consumer = cb.Consumer(Context(), store, decision_fn=lambda _: {'point_state': 'evidence_missing', 'confidence': 0.4})
            self.assertTrue(consumer.process_pending())
            self.assertEqual(store.reviews()[-1]['point_state_action'], 'continue_scoped_action')
            self.assertFalse(store.triggers())
            self.assertFalse(store.controls())

    def test_integrated_evidence_missing_without_context_permits_next_scoped_tool(self):
        contract = rb.TaskContract(
            task_id='research-task', run_id='901', profile='research', started_at=1000.0,
            original_request='Collect bounded evidence',
            criteria=('read evidence within the admitted task scope', 'record a later checkpoint'),
            current_unknown='candidate evidence remains incomplete', next_action='read_file',
            evidence_refs=('kanban://task/research-task',), search_authorized=False,
            extract_authorized=False, decision_input=generic_decision_input('research-task'),
        )
        with tempfile.TemporaryDirectory(prefix='jev-evidence-missing-integrated-') as directory:
            producer, _consumer, store, pre_tool_result = self.run_integrated(
                directory,
                {'point_state': 'evidence_missing', 'confidence': 0.95},
                contract=contract,
            )
            event = store.events()[0]
            self.assertNotIn('discovery_context', event)
            review = store.reviews()[-1]
            self.assertEqual(review['point_state'], 'evidence_missing')
            self.assertEqual(review['point_state_action'], 'continue_scoped_action')
            self.assertFalse(review['completion_allowed'])
            self.assertIsNone(review['trigger_id'])
            self.assertFalse(store.triggers())
            self.assertFalse(store.controls())
            self.assertIsNone(pre_tool_result, 'the next in-scope worker tool must not be blocked')


    def test_untrusted_jev_outcomes_do_not_block_or_accept_completion(self):
        outcomes = (
            ('low_confidence', {'point_state': 'acceptance_ready', 'confidence': 0.4}),
            ('missing_confidence', {'point_state': 'evidence_missing'}),
            ('invalid_confidence', {'point_state': 'evidence_missing', 'confidence': '0.8'}),
            ('malformed_output', {'point_state': ['not-a-state'], 'confidence': 0.95}),
            ('unknown_classification', {'point_state': 'not_in_contract', 'confidence': 0.95}),
            ('unknown_state', {'point_state': 'unknown', 'confidence': 0.95}),
        )
        contract = self.make_contract(authorized=False, extract_authorized=False)
        for name, outcome in outcomes:
            with self.subTest(outcome=name), tempfile.TemporaryDirectory(prefix='jev-untrusted-outcome-') as directory:
                producer, _consumer, store, pre_tool_result = self.run_integrated(
                    directory, outcome, contract=contract,
                )
                review = store.reviews()[-1]
                self.assertEqual(review['point_state_action'], 'continue_scoped_action')
                self.assertFalse(review['completion_allowed'])
                self.assertIsNone(review['trigger_id'])
                self.assertFalse(store.triggers())
                self.assertFalse(store.controls())
                self.assertIsNone(pre_tool_result)
                self.assertIsNone(producer.pre_tool_call('read_file'))

        def unavailable(_event):
            raise cb.JevRequestError('timeout')

        with tempfile.TemporaryDirectory(prefix='jev-unavailable-outcome-') as directory:
            producer, _consumer, store, pre_tool_result = self.run_integrated(
                directory, unavailable, contract=contract,
            )
            self.assertEqual(store.reviews()[-1]['state'], 'failed')
            self.assertFalse(store.triggers())
            self.assertFalse(store.controls())
            self.assertIsNone(pre_tool_result)
            self.assertIsNone(producer.pre_tool_call('read_file'))

    def test_acceptance_ready_requires_bound_default_readback_before_next_tool(self):
        contract = self.make_contract(authorized=False, extract_authorized=False)
        with tempfile.TemporaryDirectory(prefix='jev-acceptance-ready-readback-') as directory:
            producer, consumer, store, pre_tool_result = self.run_integrated(
                directory,
                {'point_state': 'acceptance_ready', 'confidence': 0.95},
                contract=contract,
            )
            event = store.events()[0]
            triggers = store.triggers()
            controls = store.controls()
            self.assertEqual(len(triggers), 1)
            self.assertEqual(len(controls), 1)
            trigger = triggers[0]
            self.assertEqual(trigger['task_id'], event['task_id'])
            self.assertEqual(str(trigger['run_id']), str(event['run_id']))
            self.assertEqual(trigger['checkpoint_id'], event['checkpoint_id'])
            self.assertEqual(trigger['state'], 'pending_review')
            self.assertFalse(trigger['completion_allowed'])
            self.assertEqual(controls[0]['trigger_id'], trigger['trigger_id'])
            self.assertEqual(controls[0]['control'], 'provisional_stop')
            self.assertIsNotNone(pre_tool_result)
            self.assertEqual(pre_tool_result['action'], 'block')
            with patch.object(rb, '_worker_binding', return_value=(contract.task_id, contract.run_id)):
                blocked = producer.pre_tool_call('read_file')
            self.assertIsNotNone(blocked)
            self.assertEqual(blocked['action'], 'block')
            self.assertEqual(store.reviews()[-1]['point_state_action'], 'review')
            self.assertFalse(store.reviews()[-1]['completion_allowed'])

            readback = {
                'task_id': event['task_id'],
                'run_id': event['run_id'],
                'original_request': event['original_request'],
                'completion_conditions': event['criteria'],
                'evidence_refs': event['evidence_refs'],
                'next_action': event['next_action'],
                'worker_conclusion': {
                    'status': 'unmeasured',
                    'reason': 'offline acceptance-ready test has no final worker conclusion',
                },
            }
            mismatched = dict(readback, run_id='wrong-run')
            with self.assertRaises(cb.BridgeValidationError):
                consumer.review_tool(args={
                    'trigger_id': trigger['trigger_id'],
                    'decision': 'no_intervention',
                    'readback': mismatched,
                })
            self.assertEqual(store.triggers()[0]['state'], 'pending_review')
            self.assertEqual(store.controls()[0]['control'], 'provisional_stop')
            with patch.object(rb, '_worker_binding', return_value=(contract.task_id, contract.run_id)):
                self.assertIsNotNone(producer.pre_tool_call('read_file'))

            released = json.loads(consumer.review_tool(args={
                'trigger_id': trigger['trigger_id'],
                'decision': 'no_intervention',
                'readback': readback,
            }))
            self.assertEqual(released['state'], 'no_intervention')
            self.assertEqual(released['task_id'], event['task_id'])
            self.assertEqual(str(released['run_id']), str(event['run_id']))
            self.assertEqual(store.triggers()[-1]['state'], 'no_intervention')
            self.assertEqual(store.controls()[-1]['control'], 'release')
            with patch.object(rb, '_worker_binding', return_value=(contract.task_id, contract.run_id)):
                self.assertIsNone(producer.pre_tool_call('read_file'))

if __name__ == '__main__':
    unittest.main()
