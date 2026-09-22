# Jev route screening

This repository is a clean, local sharing candidate for the Jev route-screening Hermes plugin. It contains the selected bridge/router implementation, the plugin manifest, and bounded offline tests. It is intentionally experimental: it is not a release and does not configure a remote. The repository is licensed under the MIT License; public publication and production use remain separate decisions.

## What it does

The plugin adds a Jev checkpoint bridge for two bounded paths:

- Worker checkpoints: the producer hook records a projection of worker progress and evidence references without persisting raw tool content.
- Targeted Hermes reading: the default consumer can request an advisory route for the smallest `hermes-agent` reference bundle and attach that advisory to an exact `skill_view` result.

The route labels and reference manifest are local allowlisted values. A Jev response cannot execute a tool, select an arbitrary file, change permissions, mutate skills/configuration, or replace the host's worker/skill authority.

## Authority and privacy boundary

Jev is advisory only. Hermes, the dispatcher, the normal skill loader, the task contract, and the default profile retain authority. The local manifest maps route labels to repository-relative reference paths; an untrusted provider response is never used as a file path.

Local runtime records are bounded and metadata-redacted: the bridge avoids persisting raw tool arguments, provider credentials, or provider response content in its normal ledger records, while some bounded task-contract text and evidence references may be stored locally. Provider request envelopes still carry bounded request text and selected structured context; this candidate does not provide general PII or secret scrubbing before provider submission. Provider API keys are read only from the named environment variables at runtime and are never printed by this repository.

The default scope guard is a separate safety path. A high-confidence `scope_drift` result at or above the configured `0.8` threshold can create a provisional default stop and require a matching default readback. Low-confidence or unknown-confidence results, malformed responses, timeouts, and other advisory failures remain fail-open so the original host flow is preserved. The required `pre_tool_call` scope-guard hook is therefore intentional and must not be removed to make the transform seam test pass.

## Opt-in point-state route

The worker `Consumer` can locally classify a bounded point when `settings.point_state_enabled` is explicitly enabled; the default is `false`. `ordinary_action_ready` permits only the existing scoped continuation path, while other classifications become bounded hold/guidance and return to default review. The established default scope classifier and its legacy `0.8` threshold remain separate and unchanged. The new point-state acceptance threshold is `0.60` and remains provisional/uncalibrated. This repository adds no automatic command executor or completion authority, and it does not claim live worker-to-default end-to-end verification.

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
```

For a producer worker profile, use `role: producer` and include that profile in `producer_profiles`; the production code intentionally does not expose the producer bridge as a default consumer tool. `bridge_dir` is required. The path is created with restrictive permissions and should be a local directory, not a shared public location.

The manifest declares the host-facing tools and hooks:

- Tools: `jev_bridge_checkpoint`, `jev_bridge_review`.
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
python3 -m unittest discover -v -p 'test_*.py'
python3 test_transform_tool_result_delivery.py
python3 test_plugin_manager_smoke.py
python3 -m py_compile bridge.py skill_router.py __init__.py test_*.py
```

The transform probe covers success, provider exception, timeout/fail-open, fresh plugin discovery, later-turn refresh, same-turn deduplication, and the required default scope-guard registration. The remaining tests cover the default scope guard, provider fallback, targeted-reading router, worker binding, worker-to-consumer proof, and isolated plugin registration.

## Experimental status and sharing boundary

This is an experimental local candidate for review. It has no remote and no publication metadata. The repository license is MIT; public release, credential provisioning, provider policy, and production enablement remain separate decisions and are not implied by this repository.
