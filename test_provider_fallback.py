from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODULE_NAME = "jev_route_screening_bridge_provider_test"
spec = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "bridge.py")
assert spec is not None and spec.loader is not None
bridge = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = bridge
spec.loader.exec_module(bridge)


LABELS = bridge.LABELS
GOOD_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "route": {
            "type": "choice",
            "choice": "progressing",
            "probabilities": {label: (1.0 if label == "progressing" else 0.0) for label in LABELS},
            "confidence": 1.0,
        }
    },
    "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.01},
}
REQUEST = {
    "model": "ignored-by-router",
    "state": "bounded state",
    "questions": {"route": {"type": "choice", "criteria": {label: None for label in LABELS}}},
}


class Sender:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, spec, body, timeout):
        self.calls.append((spec, dict(body), timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return json.dumps(self.payload).encode("utf-8")


class ProviderFallbackTests(unittest.TestCase):
    def test_primary_wire_and_order(self):
        sender = Sender(GOOD_RESPONSE)
        decision = bridge.request_decision_with_fallback(REQUEST, post_fn=sender, timeout=3.5)
        self.assertEqual(decision["transport"], "typesafe-system-one")
        self.assertEqual(len(sender.calls), 1)
        spec, body, timeout = sender.calls[0]
        self.assertEqual(spec.endpoint, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(spec.model, "jev-latest")
        self.assertEqual(body["model"], "jev-latest")
        self.assertEqual(timeout, 3.5)

    def test_allowed_primary_failures_use_openrouter_once(self):
        failures = [
            bridge.JevRequestError("credential_unavailable"),
            bridge.JevRequestError("network_error"),
            bridge.JevRequestError("timeout"),
            *[bridge.JevRequestError("http_error", status) for status in (401, 403, 408, 429, 500, 503, 529)],
        ]
        for failure in failures:
            with self.subTest(reason=failure.reason, status=failure.status):
                sender = Sender(failure, GOOD_RESPONSE)
                decision = bridge.request_decision_with_fallback(REQUEST, post_fn=sender)
                self.assertEqual(decision["transport"], "openrouter-decisions")
                self.assertEqual([call[0].name for call in sender.calls], ["typesafe", "openrouter"])
                self.assertEqual(sender.calls[1][0].endpoint, "https://openrouter.ai/api/alpha/decisions")
                self.assertEqual(sender.calls[1][1]["model"], "typesafe/jev-1.13")

    def test_malformed_response_and_request_never_fallback(self):
        for failure in (bridge.JevRequestError("malformed_response"), bridge.JevRequestError("http_error", 400), bridge.JevRequestError("http_error", 422)):
            with self.subTest(reason=failure.reason, status=failure.status):
                sender = Sender(failure, GOOD_RESPONSE)
                with self.assertRaises(bridge.JevRequestError):
                    bridge.request_decision_with_fallback(REQUEST, post_fn=sender)
                self.assertEqual(len(sender.calls), 1)

        sender = Sender(bridge.BridgeValidationError("malformed request"), GOOD_RESPONSE)
        with self.assertRaises(bridge.BridgeValidationError):
            bridge.request_decision_with_fallback(REQUEST, post_fn=sender)
        self.assertEqual(len(sender.calls), 1)

    def test_direct_post_wire_uses_named_env_without_output(self):
        env_name = "JEV_PROVIDER_TEST_KEY"
        previous = os.environ.get(env_name)
        os.environ[env_name] = "test-secret-only-in-memory"
        captured = {}
        original_urlopen = bridge.urlopen

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(GOOD_RESPONSE)

        bridge.urlopen = fake_urlopen
        try:
            spec = bridge.ProviderSpec("test", "https://provider.example/v1/systemone", env_name, "jev-test", "test-transport")
            value = bridge._post_provider(spec, REQUEST, timeout=4.0)
        finally:
            bridge.urlopen = original_urlopen
            if previous is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous

        self.assertEqual(value["model"], "jev-1.13.0")
        self.assertEqual(captured["url"], "https://provider.example/v1/systemone")
        self.assertEqual(captured["authorization"], "Bearer test-secret-only-in-memory")
        self.assertEqual(captured["body"]["model"], "ignored-by-router")
        self.assertEqual(captured["timeout"], 4.0)

    def test_missing_credential_error_is_secret_safe(self):
        env_name = "JEV_PROVIDER_TEST_MISSING_7F31"
        previous = os.environ.pop(env_name, None)
        try:
            with self.assertRaises(bridge.JevRequestError) as caught:
                bridge._runtime_api_key(env_name)
        finally:
            if previous is not None:
                os.environ[env_name] = previous
        self.assertEqual(caught.exception.reason, "credential_unavailable")
        self.assertNotIn(env_name, str(caught.exception))
        self.assertNotIn("test-secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
