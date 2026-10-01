"""Offline lifecycle logging contract: ordering, fallback, privacy, fail-open."""
import importlib.util, logging, sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("jev_lifecycle_test_bridge",ROOT/"bridge.py",submodule_search_locations=[str(ROOT)])
bridge=importlib.util.module_from_spec(spec); sys.modules[spec.name]=bridge; spec.loader.exec_module(bridge)
class Capture(logging.Handler):
 def __init__(self): super().__init__(); self.messages=[]
 def emit(self,record): self.messages.append(record.getMessage())
def good():
 return {"answers":{"route":{"choice":"progressing","probabilities":{label:float(label=="progressing") for label in bridge.LABELS},"confidence":1.0}}}
def test_ordered_success_and_private_body_exclusion():
 capture=Capture(); root=logging.getLogger("hermes_plugins"); root.addHandler(capture)
 try:
  result=bridge.request_decision_with_fallback({"state":"synthetic_request_text"},post_fn=lambda *args:good(),route="test.offline",request_id="test-123")
 finally: root.removeHandler(capture)
 assert result["label"]=="progressing"
 assert [m.split("event=")[1].split()[0] for m in capture.messages]==["provider_start","transport_response","validated_response","result_returned"]
 assert all("test-123" in m for m in capture.messages)
 assert "synthetic_request_text" not in "\n".join(capture.messages)
def test_fallback_order_and_logging_failure_is_non_interfering():
 capture=Capture(); root=logging.getLogger("hermes_plugins"); root.addHandler(capture); calls=[]
 def sender(spec,*args):
  calls.append(spec.name)
  if len(calls)==1: raise bridge.JevRequestError("timeout")
  return good()
 try: result=bridge.request_decision_with_fallback({},post_fn=sender,request_id="fallback-123")
 finally: root.removeHandler(capture)
 assert result["label"]=="progressing" and calls==[bridge.PRIMARY_PROVIDER.name,bridge.FALLBACK_PROVIDER.name]
 events=[m.split("event=")[1].split()[0] for m in capture.messages]
 assert events==["provider_start","provider_failure","fallback_start","provider_start","transport_response","validated_response","result_returned"]
 with patch.object(bridge.log,"info",side_effect=RuntimeError("sink unavailable")):
  assert bridge.request_decision_with_fallback({},post_fn=lambda *args:good())["label"]=="progressing"
