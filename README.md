# Jev route screening

This repository contains the public source candidate for the Jev route-screening Hermes plugin. It includes the selected bridge/router implementation, the plugin manifest (version 0.4.1), and bounded offline tests. It is intentionally experimental; this candidate patch creates no tag or release and does not configure an installed Hermes runtime. The repository is licensed under the MIT License; production use remains a separate decision.

## Lifecycle logging

The plugin emits bounded `jev.lifecycle` INFO events to the Gateway component logger for provider attempts, validated results, fallback decisions, and advisory persistence/delivery. Events correlate with a request ID and omit request/response bodies and credentials; logging failures do not change the advisory or fallback behavior.

## What it does

The plugin adds a Jev checkpoint bridge for two bounded paths:

- Worker checkpoints: the producer hook records a projection of worker progress and evidence references without persisting raw tool content.
- Targeted Hermes reading: the default consumer can request an advisory route for the smallest `hermes-agent` reference bundle and attach that advisory to an exact `skill_view` result.
- Optional task-input staging: when explicitly enabled for the default consumer, a distinct tool creates a Kanban task blocked, binds `provenance.task_id` to the generated ID, persists and reads back the versioned marker, then unblocks it. This transports input only; Jev remains advisory and the tool does not change normal task creation.

The route labels and reference manifest are local allowlisted values. A Jev response cannot execute a tool, select an arbitrary file, change permissions, mutate skills/configuration, or replace the host's worker/skill authority.

## Authority and privacy boundary

Jev is advisory only. Hermes, the dispatcher, the normal skill loader, the task contract, and the default profile retain authority. The local manifest maps route labels to repository-relative reference paths; an untrusted provider response is never used as a file path.

Local runtime records are bounded and metadata-redacted: the bridge avoids persisting raw tool arguments, provider credentials, or provider response content in its normal ledger records, while some bounded task-contract text and evidence references may be stored locally. Provider request envelopes still carry bounded request text and selected structured context; this candidate does not provide general PII or secret scrubbing before provider submission. Provider API keys are read only from the named environment variables at runtime and are never printed by this repository.

The default scope guard is a separate safety path. A high-confidence `scope_drift` result at or above the configured `0.8` threshold can create a provisional default stop and require a matching default readback. Low-confidence or unknown-confidence results, malformed responses, timeouts, and other advisory failures remain fail-open so the original host flow is preserved. The required `pre_tool_call` scope-guard hook is therefore intentional and must not be removed to make the transform seam test pass.

## Default purpose-necessity advisory

The Default guard now supplies a bounded, role-attributed conversation projection alongside the current action. The earlier user goal and the current instruction are separate fields: a later error-repair request must not silently replace the intended outcome. The earlier goal is evidence, not immutable authority; explicit user changes or cancellation take precedence. When the host does not expose a structured problem or observed outcome, those fields remain explicitly unknown.

A second typed `purpose` question asks whether the action advances the requested outcome, merely satisfies a procedure while missing the outcome, narrows discovery to fix an error, adds an unsupported requirement, represents a goal-critical prerequisite, or uses legitimately partial evidence. Insufficient context is `unknown`. Host-supplied completion conditions are requirement hypotheses, not proof that the user imposed or needs them. This assessment shares the existing bounded provider invocation; it adds no separate classifier call.

The classification and confidence are persisted and returned through the next `pre_llm_call` context seam. Purpose confidence never creates a stop, changes permissions, or decides completion: Default must inspect the actual facts and decide whether to continue, change, or drop the requirement. Existing scope-stop/release controls remain unchanged. Delivery is through the supported hook path, not a guarantee for hook-bypassing callers.

Offline purpose regression:

```bash
python3 -m pytest -q --import-mode=importlib test_default_scope_guard.py test_plugin_manager_smoke.py
```

Offline fixtures and a native registered-hook probe verify request projection, typed response handling, persistence, and next-model context delivery. They do not establish real-provider classification accuracy or prove that an already-running Gateway loaded changed source; process activation is a separate operator step.

## Opt-in point-state route

The worker `Consumer` can locally classify a bounded point when `settings.point_state_enabled` is explicitly enabled; the default is `false`. `evidence_missing`, a classifier-only `scope_or_authorization_blocked` label, and malformed or low-confidence output do not create a hold; any continuation remains limited to the existing authorized task scope and never accepts completion. Only accepted `acceptance_ready` creates a task/run-bound hold for default-owned readback, while separately supplied trusted guards retain precedence. Other classifications provide bounded guidance or a named verification without granting authority. The established default scope classifier and its legacy `0.8` threshold remain separate and unchanged. The new point-state acceptance threshold is `0.60` and remains provisional/uncalibrated. This repository adds no automatic command executor or completion authority, and it does not claim live worker-to-default end-to-end verification.

## Stopped-worker self-reporting

An active worker provisional stop still blocks ordinary tools, shell/terminal calls, completion, review requests, and other task mutations. The only reporting exception is the native `kanban_block` tool for the current trusted worker task/run/profile and board. Omitting `task_id` uses the verified current task, matching the native handler's default. Wrong-task/board requests, stale bindings, malformed arguments, and control-read failures remain denied.

This exception does not release the stop, permit work to continue, or bypass the native tool's authorization checks. Default retains the existing evidence-review and stop-release responsibility. The narrow exception requires the compatible Hermes Kanban APIs; unavailable identity verification fails closed for the exception, without changing the ordinary no-stop path.

Offline regression coverage for the registered hook and live-assignment discriminators is available via `python3 test_self_block_report.py`; it uses temporary state and mocked board rows, not a production board mutation.

## Installation and configuration

Install this directory under the active Hermes home plugin root as `plugins/jev-route-screening/`, then enable it through the host plugin configuration. Do not install a second gateway or copy credentials into this repository. A consumer configuration can be represented as:

```yaml
plugins:
  enabled:
    - jev-route-screening
  entries:
    jev-route-screening:
      settings:
        bridge_dir: ~/.hermes/jev-bridge
        role: consumer
        consumer_profiles:
          - default
        producer_profiles:
          - ops
        targeted_reading:
          enabled: true
          once_per_session: true
          timeout_seconds: 5.0
          max_request_chars: 4000
          max_context_chars: 6000
        task_input_staging_enabled: false
```

Task-input staging is disabled unless explicitly enabled under the default consumer's plugin settings. Set `task_input_staging_enabled: true` to register `jev_stage_task_input`; it is never registered for worker profiles. The tool requires the standard `kanban_create`, `kanban_show`, and `kanban_unblock` tools plus `terminal` for the supported `hermes kanban edit` body update. Failures before verified persistence leave the task blocked; retrying identical arguments reuses Kanban's idempotency key.

For a producer worker profile, use `role: producer` and include that profile in `producer_profiles`; the production code intentionally does not expose the producer bridge as a default consumer tool. `bridge_dir` is required. The path is created with restrictive permissions and should be a local directory, not a shared public location.

The manifest declares the host-facing tools and hooks:

- Tools: `jev_bridge_checkpoint`, `jev_bridge_review`, and the opt-in default-only `jev_stage_task_input`.
- Hooks: `pre_llm_call`, `pre_tool_call`, `post_tool_call`, `transform_tool_result`, and `on_kanban_worker_spawned`.

## Provider and data flow

The targeted-reading and decision paths use TypeSafe as the primary provider and OpenRouter as the bounded fallback. The provider names, endpoints, models, and environment-variable names are defined in `bridge.py`:

1. The local hook captures a bounded request and constructs a typed, advisory-only state envelope.
2. The primary TypeSafe request is attempted first.
3. Only the implementation's permitted failure classes use the OpenRouter fallback.
4. The typed response is checked against the local route manifest.
5. The existing Hermes loader reads the hub and any selected local references; the provider does not authorize those reads.
6. `transform_tool_result` adds a separate `jev_advisory` field to a JSON `skill_view` result while preserving the original result fields and content.

The tests use fixtures and mock callbacks only. Running this repository's tests does not make provider or network calls.

## Local persistence

`bridge_dir` contains local JSONL state such as worker events, reservations/ledger rows, reviews, delivery records, triggers, controls, diagnostics, worker bindings, default-scope outcomes, and targeted-reading advisory/observation rows. The implementation creates the directory with mode `0700` and JSONL files with mode `0600` where the host permits it. These files are runtime state, not release artifacts; do not commit them or include them in an external share. This candidate does not define a general retention or pruning policy, so operators must choose the directory and retention period deliberately.

## Host compatibility

The code targets the Hermes Agent plugin API that supports the declared hooks, including the `transform_tool_result` seam. The transform callback runs after the normal `post_tool_call` callback and before the result is delivered to model context. Provider failure and hook timeout must preserve the original tool result. The integration probe uses a fresh child process and a temporary Hermes home so plugin discovery, registration, and the real `skill_view` dispatch are exercised without changing a live installation.

The repository is not a standalone Python package and does not vendor Hermes Agent. Run its tests in an environment where the compatible Hermes runtime is installed or otherwise importable. Test paths derive from the checked-out repository and the host's temporary-directory facilities; they do not require this machine's absolute profile paths. `JEV_TEST_SCRATCH` can optionally select a temporary parent directory, and `JEV_OPS_PLUGIN_ROOT` can optionally point the worker-binding test at a separate compatible plugin copy.

## Verification

From the repository root:

```bash
(cd .. && python3 -m pytest --import-mode=importlib -q jev-route-screening-public/test_jev_point_state_integration.py jev-route-screening-public/test_point_state.py)
python3 test_transform_tool_result_delivery.py
python3 test_plugin_manager_smoke.py
python3 test_task_input_staging.py
python3 -m py_compile bridge.py skill_router.py __init__.py task_input_staging.py test_*.py
hermes plugins validate --json "$PWD"
hermes plugins doctor --ci "$PWD"
```

The point-state command uses pytest's importlib mode from the parent directory because the repository root contains hyphens. It covers missing evidence, classifier-only scope labels, untrusted results, guard precedence, and acceptance-ready readback/release. Full unittest discovery is not listed: two legacy dynamic-loader tests import `bridge.py` outside its package and fail its relative `point_state` import in this checkout. The transform probe covers success, provider exception, timeout/fail-open, fresh plugin discovery, later-turn refresh, same-turn deduplication, and the required default scope-guard registration. Other regression modules cover the default scope guard, provider fallback, targeted-reading router, worker binding, worker-to-consumer proof, and isolated plugin registration.

## Experimental status and sharing boundary

This remains an experimental source candidate. Version 0.4.1 is the plugin manifest version; this patch creates no tag or release. The repository license is MIT; credential provisioning, provider policy, and production enablement remain separate decisions and are not implied by this repository.
