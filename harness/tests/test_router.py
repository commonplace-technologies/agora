"""Tests for router.py. Network tests run against in-process fake servers that
speak the subset of the OpenAI API llama-server exposes; nothing leaves the host.

Run: python3 -m pytest harness/tests -q
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import router  # noqa: E402
from router import FormatError, HybridRouter, OutputFormat, Provider, RouterConfig  # noqa: E402

GOOD_DIFF = """--- a/app.py
+++ b/app.py
@@ -1,3 +1,4 @@
 import os
+import sys

 def main():
"""


# --------------------------------------------------------------------------- #
# Fake OpenAI-compatible server
# --------------------------------------------------------------------------- #

class FakeServer:
    """behaviour: ok | slow_first | stall | oom | down-models.
    reply: text the model "generates" (streamed in 8-char chunks for local)."""

    def __init__(self, behaviour="ok", reply="hello", models=None, stream=True):
        self.behaviour, self.reply, self.stream = behaviour, reply, stream
        self.models = models or [{"id": "Qwen-2.5-Coder-32B-Q8", "object": "model",
                                  "status": {"value": "loaded"}}]
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._json(200, {"object": "list", "data": outer.models})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(req)
                if outer.behaviour == "oom":
                    return self._json(500, {"error": {"message": "ggml_vulkan: ErrorOutOfDeviceMemory",
                                                      "type": "server_error"}})
                if not req.get("stream"):
                    return self._json(200, {
                        "id": "x", "object": "chat.completion", "created": 0, "model": "cloud-model",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": outer.reply}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 5, "total_tokens": 6}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if outer.behaviour == "slow_first":
                    time.sleep(3)
                chunks = [outer.reply[i:i + 8] for i in range(0, len(outer.reply), 8)]
                try:
                    for n, piece in enumerate(chunks):
                        if outer.behaviour == "stall" and n == 1:
                            time.sleep(3)
                        data = {"id": "x", "object": "chat.completion.chunk", "created": 0,
                                "model": req["model"],
                                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def servers():
    made = []

    def make(**kw):
        s = FakeServer(**kw)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def make_router(local, cloud=None, **overrides):
    cfg = RouterConfig(local_base=local.base if local else "http://127.0.0.1:9/v1",
                       cloud_provider="custom", cloud_base=cloud.base if cloud else "http://127.0.0.1:9/v1",
                       cloud_api_key="k" if cloud else None, cloud_model="cloud-model",
                       local_first_token_timeout=1.0, local_idle_timeout=1.0, local_load_timeout=2.0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return HybridRouter(cfg)


# --------------------------------------------------------------------------- #
# Heuristics and config
# --------------------------------------------------------------------------- #

def test_benchmark_cases_route_as_expected():
    cfg = RouterConfig()
    for prompt, context, expected in router.BENCHMARK_CASES:
        assert router.classify(prompt, context, cfg).provider is expected, prompt


def test_phrase_matching_is_whole_word_and_whitespace_tolerant():
    cfg = RouterConfig()
    assert router.classify("Find the ROOT\n  cause here", config=cfg).provider is Provider.CLOUD
    assert router.classify("rootcause", config=cfg).provider is Provider.LOCAL


def test_force_overrides_heuristics():
    d = router.classify("refactor architecture", force="local")
    assert d.provider is Provider.LOCAL and d.forced


@pytest.mark.parametrize("env,expected", [
    ({}, "http://localhost:11434/v1"),
    ({"OLLAMA_HOST": "192.168.8.153:11434"}, "http://192.168.8.153:11434/v1"),
    ({"OLLAMA_HOST": "0.0.0.0", "OPENAI_BASE_URL": "http://lebowski:11434/v1"}, "http://lebowski:11434/v1"),
    ({"LOCAL_API_BASE": "http://a:1/v1/", "OLLAMA_HOST": "b:2"}, "http://a:1/v1"),
])
def test_local_base_resolution(monkeypatch, env, expected):
    for k in ("LOCAL_API_BASE", "OLLAMA_HOST", "OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert router.resolve_local_base() == expected


# --------------------------------------------------------------------------- #
# Output validation
# --------------------------------------------------------------------------- #

def test_extract_diff_strips_fences_and_prose():
    raw = f"Here is the patch:\n```diff\n{GOOD_DIFF}```\nLet me know!"
    assert router.extract_diff(raw) == GOOD_DIFF


def test_extract_diff_rejects_bad_hunk_counts():
    bad = GOOD_DIFF.replace("@@ -1,3 +1,4 @@", "@@ -1,3 +1,9 @@")
    with pytest.raises(FormatError, match="declares"):
        router.extract_diff(bad)


def test_extract_diff_rejects_non_diff():
    with pytest.raises(FormatError):
        router.extract_diff("def f():\n    return 1\n")


def test_extract_tool_call_validates_arguments():
    call = router.extract_tool_call('```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```')
    assert call == {"name": "read_file", "arguments": {"path": "a.py"}}
    with pytest.raises(FormatError):
        router.extract_tool_call('{"name": "read_file", "arguments": {}}')
    with pytest.raises(FormatError):
        router.extract_tool_call('{"name": "rm_rf", "arguments": {}}')


# --------------------------------------------------------------------------- #
# Execution, model resolution and failover
# --------------------------------------------------------------------------- #

def test_local_success_maps_ollama_tag_to_llama_server_alias(servers):
    local = servers(reply="local answer")
    resp = make_router(local).route_and_execute("fix a typo")
    assert resp.provider_used == "local" and not resp.failover
    assert resp.model_used == "Qwen-2.5-Coder-32B-Q8"
    assert local.requests[0]["model"] == "Qwen-2.5-Coder-32B-Q8"
    assert resp.content == "local answer\n"


@pytest.mark.parametrize("behaviour,reason", [
    ("slow_first", "no first token"),
    ("stall", "stalled"),
    ("oom", "driver/OOM"),
])
def test_local_failures_fail_over_to_cloud(servers, caplog, behaviour, reason):
    local = servers(behaviour=behaviour, reply="local answer that is long enough to chunk")
    cloud = servers(reply="cloud answer")
    resp = make_router(local, cloud).route_and_execute("fix a typo")
    assert resp.provider_used == "cloud" and resp.failover
    assert reason in resp.failover_reason
    assert "failing over to cloud" in caplog.text


def test_unreachable_local_fails_over(servers):
    cloud = servers(reply="cloud answer")
    resp = make_router(None, cloud).route_and_execute("fix a typo")
    assert resp.provider_used == "cloud" and "unreachable" in resp.failover_reason


def test_unloaded_model_gets_load_timeout(servers):
    # slow_first sleeps 3s: beyond first-token (1s) but within load timeout (5s).
    local = servers(behaviour="slow_first", reply="ok",
                    models=[{"id": "Qwen-2.5-Coder-32B-Q8", "status": {"value": "unloaded"}}])
    resp = make_router(local, None, local_load_timeout=5.0).route_and_execute("fix a typo")
    assert resp.provider_used == "local"


def test_missing_local_model_fails_over(servers):
    local = servers(models=[{"id": "Phi-4-14B"}])
    cloud = servers(reply="cloud answer")
    resp = make_router(local, cloud).route_and_execute("fix a typo")
    assert resp.provider_used == "cloud" and "not served" in resp.failover_reason


def test_forced_local_does_not_fail_over(servers):
    local = servers(behaviour="oom")
    cloud = servers(reply="cloud answer")
    with pytest.raises(RuntimeError, match="all providers failed"):
        make_router(local, cloud).route_and_execute("fix a typo", force="local")
    assert not cloud.requests


def test_invalid_local_diff_is_repaired_then_fails_over(servers):
    local = servers(reply="not a diff at all")
    cloud = servers(reply=f"```diff\n{GOOD_DIFF}```")
    resp = make_router(local, cloud).route_and_execute("add import", output_format=OutputFormat.DIFF)
    assert len(local.requests) == 2  # original + one repair
    assert resp.provider_used == "cloud" and resp.content == GOOD_DIFF
    assert "invalid diff" in resp.failover_reason


def test_json_tool_mode_sends_schema_locally_and_json_object_to_cloud(servers):
    reply = '{"name": "final_answer", "arguments": {"text": "done"}}'
    local = servers(reply=reply)
    resp = make_router(local).route_and_execute("say done", output_format="json_tool")
    assert resp.tool_call == {"name": "final_answer", "arguments": {"text": "done"}}
    assert local.requests[0]["response_format"]["type"] == "json_schema"

    cloud = servers(reply=reply)
    resp = make_router(local, cloud).route_and_execute("say done", output_format="json_tool", force="cloud")
    assert cloud.requests[0]["response_format"] == {"type": "json_object"}


def test_cloud_without_key_reports_local_failure(servers):
    local = servers(behaviour="oom")
    r = make_router(local, None)
    with pytest.raises(RuntimeError, match="after local failure"):
        r.route_and_execute("fix a typo")
