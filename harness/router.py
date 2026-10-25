#!/usr/bin/env python3
"""Hybrid local/cloud router for coding tasks.

Routes a prompt to a local OpenAI-compatible server (llama.cpp ``llama-server``
or Ollama on Lebowski) or to a cloud OpenAI-compatible API (OpenRouter or
DeepSeek), based on cheap heuristics. Local calls are streamed and guarded by
first-token and idle watchdogs; timeouts, connection failures and driver/OOM
errors fail over to the cloud with a warning. Output can be constrained to a
unified diff or to a single JSON tool call, validated on either provider.

Run ``python3 router.py --help`` for the CLI. Import ``HybridRouter`` to use it
as a library (``mcp_server.py`` does).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import openai
    from openai import OpenAI
except ImportError:  # pragma: no cover - surfaced at runtime with a clear message
    openai = None
    OpenAI = None

log = logging.getLogger("hybrid_router")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_LOCAL_BASE = "http://localhost:11434/v1"

# Ollama tag -> llama-server preset alias. Resolution also falls back to a
# normalised fuzzy match, so this only needs entries that fuzzy matching misses.
MODEL_ALIASES = {
    "qwen2.5-coder:32b-instruct-q8_0": "Qwen-2.5-Coder-32B-Q8",
    "llama3.3:70b-instruct-q4_K_M": "Llama-3.3-70B",
}

CLOUD_PRESETS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "deepseek/deepseek-chat"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat"),
}


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _normalise_base(url: str) -> str:
    """Accept ``host``, ``host:port``, ``http://host:port`` or a full ``/v1`` URL."""
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def resolve_local_base() -> str:
    """LOCAL_API_BASE > OLLAMA_HOST > OPENAI_BASE_URL > localhost.

    OLLAMA_HOST and OPENAI_BASE_URL are honoured because they are already
    exported on mini to point at Lebowski.
    """
    for name in ("LOCAL_API_BASE", "OLLAMA_HOST", "OPENAI_BASE_URL"):
        value = _env(name)
        if value:
            # OLLAMA_HOST=0.0.0.0 is a bind address, not something to connect to.
            if name == "OLLAMA_HOST" and value.split(":")[0] in ("0.0.0.0", "http://0.0.0.0"):
                continue
            return _normalise_base(value)
    return DEFAULT_LOCAL_BASE


@dataclasses.dataclass
class RouterConfig:
    local_base: str = dataclasses.field(default_factory=resolve_local_base)
    local_api_key: str = dataclasses.field(default_factory=lambda: _env("LOCAL_API_KEY", "sk-local"))
    local_model: str = dataclasses.field(
        default_factory=lambda: _env("LOCAL_PRIMARY_MODEL", "qwen2.5-coder:32b-instruct-q8_0")
    )
    # Seconds allowed before the first streamed token (covers prompt processing).
    local_first_token_timeout: float = dataclasses.field(
        default_factory=lambda: float(_env("LOCAL_TIMEOUT_SECONDS", "45"))
    )
    # Used instead of the above when the model is not resident and must be loaded.
    local_load_timeout: float = dataclasses.field(
        default_factory=lambda: float(_env("LOCAL_LOAD_TIMEOUT_SECONDS", "240"))
    )
    # Max silence between streamed chunks once generation has started.
    local_idle_timeout: float = dataclasses.field(
        default_factory=lambda: float(_env("LOCAL_IDLE_TIMEOUT_SECONDS", "20"))
    )
    # Hard ceiling on a local call; 0 disables it.
    local_total_timeout: float = dataclasses.field(
        default_factory=lambda: float(_env("LOCAL_TOTAL_TIMEOUT_SECONDS", "900"))
    )

    cloud_provider: str = dataclasses.field(default_factory=lambda: _env("CLOUD_PROVIDER", "openrouter"))
    cloud_base: Optional[str] = dataclasses.field(default_factory=lambda: _env("CLOUD_API_BASE"))
    cloud_api_key: Optional[str] = dataclasses.field(default_factory=lambda: _env("CLOUD_API_KEY"))
    cloud_model: Optional[str] = dataclasses.field(default_factory=lambda: _env("CLOUD_MODEL"))
    cloud_timeout: float = dataclasses.field(default_factory=lambda: float(_env("CLOUD_TIMEOUT_SECONDS", "300")))

    token_threshold: int = dataclasses.field(default_factory=lambda: int(_env("ROUTER_TOKEN_THRESHOLD", "8000")))
    file_threshold: int = dataclasses.field(default_factory=lambda: int(_env("ROUTER_FILE_THRESHOLD", "4")))
    max_tokens: int = dataclasses.field(default_factory=lambda: int(_env("ROUTER_MAX_TOKENS", "4096")))
    temperature: float = dataclasses.field(default_factory=lambda: float(_env("ROUTER_TEMPERATURE", "0.2")))
    # One same-provider repair attempt when output fails format validation.
    repair_attempts: int = 1

    def __post_init__(self) -> None:
        preset = CLOUD_PRESETS.get(self.cloud_provider)
        if preset is None and not self.cloud_base:
            raise ValueError(
                f"CLOUD_PROVIDER={self.cloud_provider!r} is unknown; use one of "
                f"{sorted(CLOUD_PRESETS)} or set CLOUD_API_BASE"
            )
        if preset:
            base, key_var, model = preset
            self.cloud_base = self.cloud_base or base
            self.cloud_api_key = self.cloud_api_key or _env(key_var)
            self.cloud_model = self.cloud_model or model


# --------------------------------------------------------------------------- #
# Routing heuristics
# --------------------------------------------------------------------------- #

class Provider(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


# Phrases that indicate cross-cutting design or diagnosis work. Matched as
# whole phrases (word boundaries, flexible whitespace), case-insensitive.
COMPLEXITY_PHRASES = [
    "refactor architecture",
    "architecture refactor",
    "re-architect",
    "rearchitect",
    "system redesign",
    "redesign the system",
    "multi-module migration",
    "cross-module",
    "monorepo migration",
    "root cause",
    "race condition",
    "deadlock",
    "memory leak",
    "distributed consensus",
    "event sourcing",
    "threat model",
]
_PHRASE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in COMPLEXITY_PHRASES) + r")\b",
    re.IGNORECASE,
)
# File markers in pasted context: diff headers or "### path/to/file.ext" headings
# (the format build_context() uses for --file).
_FILE_MARKER_RE = re.compile(
    r"^(?:diff --git a/(\S+) b/\S+|\+\+\+ b/(\S+)|#{2,4} (?:File: )?`?([\w./-]+\.\w+)`?)\s*$",
    re.MULTILINE,
)


def estimate_tokens(text: str) -> int:
    """~4 characters per token. Code tokenises denser than prose, so this
    under-counts by 10-25%; good enough for a routing threshold."""
    return (len(text) + 3) // 4


@dataclasses.dataclass
class RoutingDecision:
    provider: Provider
    reason: str
    estimated_tokens: int
    triggers: list[str]
    files: int
    forced: bool = False


def classify(prompt: str, context: str = "", config: Optional[RouterConfig] = None,
             force: Optional[str] = None) -> RoutingDecision:
    config = config or RouterConfig()
    tokens = estimate_tokens(prompt) + estimate_tokens(context)
    triggers = sorted({m.group(1).lower() for m in _PHRASE_RE.finditer(prompt)})
    files = len({next(g for g in m.groups() if g) for m in _FILE_MARKER_RE.finditer(context)})

    if force:
        provider = Provider(force)
        return RoutingDecision(provider, f"forced {provider.value}", tokens, triggers, files, forced=True)
    if tokens > config.token_threshold:
        return RoutingDecision(Provider.CLOUD, f"~{tokens} tokens > threshold {config.token_threshold}",
                               tokens, triggers, files)
    if triggers:
        return RoutingDecision(Provider.CLOUD, f"complexity phrases: {', '.join(triggers)}",
                               tokens, triggers, files)
    if files >= config.file_threshold:
        return RoutingDecision(Provider.CLOUD, f"{files} files in context >= {config.file_threshold}",
                               tokens, triggers, files)
    return RoutingDecision(Provider.LOCAL, "routine task", tokens, triggers, files)


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #

class OutputFormat(str, Enum):
    TEXT = "text"
    DIFF = "diff"
    JSON_TOOL = "json_tool"


DIFF_SYSTEM = (
    "You are a code-editing engine. Respond with a single unified diff and nothing else: "
    "no prose, no explanation, no markdown fences. Use git-style headers "
    "(`--- a/<path>` and `+++ b/<path>`, `/dev/null` for created or deleted files) and "
    "`@@ -l,s +l,s @@` hunk headers with correct line counts. Include three lines of "
    "unchanged context around each change. Paths are relative to the repository root."
)

TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "arguments"],
    "properties": {
        "name": {"type": "string", "enum": ["apply_patch", "read_file", "execute_command", "final_answer"]},
        "arguments": {"type": "object"},
    },
}
TOOL_ARGUMENTS = {
    "apply_patch": {"patch": str},
    "read_file": {"path": str},
    "execute_command": {"command": str},
    "final_answer": {"text": str},
}
JSON_TOOL_SYSTEM = (
    "You are a coding agent that acts only through tools. Respond with exactly one JSON "
    'object of the form {"name": <tool>, "arguments": {...}} and nothing else. Tools:\n'
    '- apply_patch {"patch": "<unified diff>"}\n'
    '- read_file {"path": "<repo-relative path>"}\n'
    '- execute_command {"command": "<shell command>"}\n'
    '- final_answer {"text": "<answer for the user>"}'
)

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class FormatError(ValueError):
    pass


def _strip_fences(text: str, prefer: tuple[str, ...]) -> str:
    """Return the body of the first fenced block that looks like the wanted
    content, else the text itself."""
    blocks = _FENCE_RE.findall(text)
    for block in blocks:
        if block.lstrip().startswith(prefer):
            return block
    return blocks[0] if len(blocks) == 1 else text


def extract_diff(text: str) -> str:
    """Normalise model output to a bare unified diff and check its structure.

    Checks headers and hunk line counts; it does not check that context lines
    match the target file. Use ``--check-apply`` (git apply --check) for that.
    """
    body = _strip_fences(text, ("diff --git", "--- ", "Index:"))
    lines = body.strip("\n").splitlines()
    # Drop leading prose before the first header.
    start = next((i for i, l in enumerate(lines) if l.startswith(("diff --git", "--- "))), None)
    if start is None:
        raise FormatError("no '--- a/...' header found")
    lines = lines[start:]

    files = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                raise FormatError(f"'---' header without '+++' on line {start + i + 1}")
            files += 1
            i += 2
            continue
        m = _HUNK_RE.match(line)
        if m:
            old_n = int(m.group(2) if m.group(2) is not None else 1)
            new_n = int(m.group(4) if m.group(4) is not None else 1)
            i += 1
            seen_old = seen_new = 0
            while i < len(lines) and (seen_old < old_n or seen_new < new_n):
                l = lines[i]
                if l.startswith("\\"):  # "\ No newline at end of file"
                    i += 1
                    continue
                tag = l[:1]
                if tag in (" ", ""):
                    seen_old += 1
                    seen_new += 1
                elif tag == "-":
                    seen_old += 1
                elif tag == "+":
                    seen_new += 1
                else:
                    break
                i += 1
            if (seen_old, seen_new) != (old_n, new_n):
                raise FormatError(
                    f"hunk '{line}' declares -{old_n}/+{new_n} lines but has -{seen_old}/+{seen_new}"
                )
            continue
        if line.startswith(("diff --git", "index ", "new file mode", "deleted file mode",
                            "similarity index", "rename from", "rename to", "\\")) or not line.strip():
            i += 1
            continue
        # Trailing prose after the last hunk: cut it off.
        if files:
            lines = lines[:i]
            break
        raise FormatError(f"unexpected line {start + i + 1}: {line[:60]!r}")
    if not files:
        raise FormatError("no file headers found")
    return "\n".join(lines) + "\n"


def extract_tool_call(text: str) -> dict[str, Any]:
    body = _strip_fences(text, ("{",)).strip()
    if not body.startswith("{"):
        # Tolerate prose around a single object.
        first, last = body.find("{"), body.rfind("}")
        if first == -1 or last <= first:
            raise FormatError("no JSON object found")
        body = body[first:last + 1]
    try:
        call = json.loads(body)
    except json.JSONDecodeError as exc:
        raise FormatError(f"invalid JSON: {exc}") from None
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise FormatError("expected exactly the keys 'name' and 'arguments'")
    spec = TOOL_ARGUMENTS.get(call["name"])
    if spec is None:
        raise FormatError(f"unknown tool {call['name']!r}")
    args = call["arguments"]
    if not isinstance(args, dict):
        raise FormatError("'arguments' must be an object")
    for key, typ in spec.items():
        if not isinstance(args.get(key), typ):
            raise FormatError(f"{call['name']}.{key} must be a {typ.__name__}")
    if call["name"] == "apply_patch":
        args["patch"] = extract_diff(args["patch"])
    return call


def git_apply_check(diff: str, repo: str = ".") -> None:
    proc = subprocess.run(["git", "-C", repo, "apply", "--check", "-"], input=diff,
                          text=True, capture_output=True)
    if proc.returncode != 0:
        raise FormatError(f"git apply --check failed: {proc.stderr.strip()}")


# --------------------------------------------------------------------------- #
# Provider calls
# --------------------------------------------------------------------------- #

class LocalFailure(RuntimeError):
    """A local failure that should trigger cloud failover."""


# Substrings seen in llama.cpp (Vulkan/HIP), Ollama and ROCm error bodies when
# the backend runs out of memory or the driver/worker falls over.
_DRIVER_ERROR_RE = re.compile(
    r"out of memory|\boom\b|ErrorOutOfDeviceMemory|ErrorDeviceLost|vk::|vulkan|hipError|"
    r"\bhip\b|rocm|cuda error|failed to (?:load|allocate)|unable to allocate|ggml_|"
    r"llama runner|runner process|model .* failed|proxy error|could not establish connection|"
    r"failed to read connection|context (?:size|length|window)|exceeds the available",
    re.IGNORECASE,
)


@dataclasses.dataclass
class RouterResponse:
    content: str
    provider_used: str
    model_used: str
    latency_seconds: float
    decision: RoutingDecision
    output_format: str
    failover: bool = False
    failover_reason: Optional[str] = None
    first_token_seconds: Optional[float] = None
    completion_tokens: Optional[int] = None
    tool_call: Optional[dict[str, Any]] = None

    @property
    def tokens_per_second(self) -> Optional[float]:
        if not self.completion_tokens or self.first_token_seconds is None:
            return None
        gen = self.latency_seconds - self.first_token_seconds
        return self.completion_tokens / gen if gen > 0 else None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["decision"]["provider"] = self.decision.provider.value
        d["tokens_per_second"] = self.tokens_per_second
        return d


class HybridRouter:
    def __init__(self, config: Optional[RouterConfig] = None):
        if OpenAI is None:
            raise RuntimeError("the 'openai' package is required: pip install -r requirements.txt")
        self.config = config or RouterConfig()
        c = self.config
        # Watchdogs below enforce local deadlines; the SDK's own timeouts are a backstop.
        self._local = OpenAI(base_url=c.local_base, api_key=c.local_api_key, max_retries=0,
                             timeout=openai.Timeout(c.local_load_timeout + 60, connect=5.0))
        self._cloud = None
        if c.cloud_api_key:
            headers = {"X-Title": "hybrid-router"} if "openrouter.ai" in (c.cloud_base or "") else None
            self._cloud = OpenAI(base_url=c.cloud_base, api_key=c.cloud_api_key, max_retries=2,
                                 timeout=c.cloud_timeout, default_headers=headers)
        self._models_cache: Optional[tuple[float, list[dict[str, Any]]]] = None

    # -- local model discovery ------------------------------------------------

    def local_models(self, refresh: bool = False) -> list[dict[str, Any]]:
        now = time.monotonic()
        if refresh or not self._models_cache or now - self._models_cache[0] > 30:
            page = self._local.with_options(timeout=5.0).models.list()
            self._models_cache = (now, [m.model_dump() for m in page.data])
        return self._models_cache[1]

    @staticmethod
    def _norm(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower().replace("instruct", ""))

    def resolve_local_model(self, requested: Optional[str] = None) -> tuple[str, Optional[bool]]:
        """Map a requested name to what the server actually serves.

        Returns (model_id, loaded) where loaded is None if the server does not
        report load state. Ollama tags are mapped to llama-server aliases via
        MODEL_ALIASES, then by normalised prefix match.
        """
        requested = requested or self.config.local_model
        models = self.local_models()
        ids = {m["id"]: m for m in models}
        candidates = [requested, MODEL_ALIASES.get(requested, "")]
        match = next((c for c in candidates if c in ids), None)
        if match is None:
            want = [self._norm(c) for c in candidates if c]
            for mid in ids:
                n = self._norm(mid.split("/")[-1])
                if any(n == w or n.startswith(w) or w.startswith(n) for w in want):
                    match = mid
                    break
        if match is None:
            raise LocalFailure(f"model {requested!r} not served at {self.config.local_base}; "
                               f"available: {', '.join(sorted(ids)) or 'none'}")
        return match, _loaded_state(ids[match])

    # -- calls ----------------------------------------------------------------

    def _call_local(self, messages: list[dict[str, str]], fmt: OutputFormat) -> tuple[str, str, float, Optional[int]]:
        c = self.config
        try:
            model, loaded = self.resolve_local_model()
        except LocalFailure:
            raise
        except Exception as exc:  # connection refused, DNS, 5xx on /models
            raise LocalFailure(f"local server unreachable at {c.local_base}: {exc}") from exc

        first_deadline = c.local_load_timeout if loaded is False else c.local_first_token_timeout
        kwargs: dict[str, Any] = dict(model=model, messages=messages, stream=True,
                                      max_tokens=c.max_tokens, temperature=c.temperature)
        if fmt is OutputFormat.JSON_TOOL:
            # llama-server compiles the schema to a grammar, so output is
            # constrained at sampling time. Ollama accepts the same field.
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "tool_call", "schema": TOOL_SCHEMA}}
        try:
            stream = self._local.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LocalFailure(_describe(exc)) from exc

        state = _StreamState()
        worker = threading.Thread(target=state.consume, args=(stream,), daemon=True)
        start = time.monotonic()
        worker.start()
        while worker.is_alive():
            worker.join(0.1)
            now = time.monotonic()
            reason = None
            if state.first_at is None and now - start > first_deadline:
                reason = (f"no first token after {first_deadline:.0f}s"
                          + (" (model was not loaded)" if loaded is False else ""))
            elif state.first_at is not None and now - state.last_at > c.local_idle_timeout:
                reason = f"stream stalled for {c.local_idle_timeout:.0f}s after {len(state.text)} chars"
            elif c.local_total_timeout and now - start > c.local_total_timeout:
                reason = f"exceeded total limit of {c.local_total_timeout:.0f}s"
            if reason:
                state.abort(stream)
                raise LocalFailure("timeout: " + reason)
        if state.error is not None:
            raise LocalFailure(_describe(state.error)) from state.error
        text = "".join(state.text)
        if not text.strip():
            raise LocalFailure("empty response")
        ttft = (state.first_at - start) if state.first_at else None
        # Servers that omit usage in streams stream ~one token per chunk.
        return text, model, ttft, state.completion_tokens or state.chunks

    def _call_cloud(self, messages: list[dict[str, str]], fmt: OutputFormat) -> tuple[str, str, Optional[int]]:
        if self._cloud is None:
            raise RuntimeError(
                f"cloud provider {self.config.cloud_provider!r} has no API key "
                f"(set {CLOUD_PRESETS.get(self.config.cloud_provider, ('', 'CLOUD_API_KEY'))[1]})"
            )
        kwargs: dict[str, Any] = dict(model=self.config.cloud_model, messages=messages,
                                      max_tokens=self.config.max_tokens, temperature=self.config.temperature)
        if fmt is OutputFormat.JSON_TOOL:
            # json_object is the most widely supported mode across OpenRouter
            # backends and DeepSeek; the schema is enforced by validation.
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._cloud.chat.completions.create(**kwargs)
        usage = resp.usage.completion_tokens if resp.usage else None
        return resp.choices[0].message.content or "", resp.model or self.config.cloud_model, usage

    # -- public API -----------------------------------------------------------

    def route_and_execute(self, prompt: str, system_context: str = "",
                          output_format: OutputFormat | str = OutputFormat.TEXT,
                          force: Optional[str] = None, check_apply: Optional[str] = None) -> RouterResponse:
        """Route, execute, validate and fail over.

        ``system_context`` is code or other material the prompt refers to.
        ``check_apply`` is a repo path; when set, diffs must pass git apply --check.
        """
        fmt = OutputFormat(output_format)
        decision = classify(prompt, system_context, self.config, force)
        messages = build_messages(prompt, system_context, fmt)
        start = time.monotonic()
        failover_reason = None

        order = [decision.provider]
        if decision.provider is Provider.LOCAL and force != "local":
            order.append(Provider.CLOUD)

        last_error: Optional[Exception] = None
        for provider in order:
            for attempt in range(self.config.repair_attempts + 1):
                ttft = tokens = None
                try:
                    if provider is Provider.LOCAL:
                        raw, model, ttft, tokens = self._call_local(messages, fmt)
                    else:
                        raw, model, tokens = self._call_cloud(messages, fmt)
                except LocalFailure as exc:
                    last_error = exc
                    failover_reason = str(exc)
                    if len(order) > 1:
                        log.warning("LOCAL FAILURE -> failing over to cloud (%s): %s",
                                    self.config.cloud_model, exc)
                    break
                except Exception as exc:
                    if provider is Provider.CLOUD:
                        after = f" (after local failure: {failover_reason})" if failover_reason else ""
                        raise RuntimeError(f"cloud call failed: {exc}{after}") from exc
                    raise
                try:
                    content, tool_call = finalise(raw, fmt, check_apply)
                except FormatError as exc:
                    last_error = exc
                    log.warning("%s output failed %s validation (attempt %d): %s",
                                provider.value, fmt.value, attempt + 1, exc)
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": f"That output is invalid: {exc}. "
                                                    f"Reply again with only the corrected {fmt.value}."},
                    ]
                    continue
                return RouterResponse(
                    content=content, provider_used=provider.value, model_used=model,
                    latency_seconds=time.monotonic() - start, decision=decision,
                    output_format=fmt.value, failover=provider is not decision.provider,
                    failover_reason=failover_reason if provider is not decision.provider else None,
                    first_token_seconds=ttft, completion_tokens=tokens, tool_call=tool_call,
                )
            else:
                # Repair attempts exhausted on this provider.
                if provider is Provider.LOCAL and len(order) > 1:
                    failover_reason = f"invalid {fmt.value} output: {last_error}"
                    log.warning("LOCAL OUTPUT INVALID -> failing over to cloud: %s", last_error)
                    messages = build_messages(prompt, system_context, fmt)
        raise RuntimeError(f"all providers failed; last error: {last_error}") from last_error


class _StreamState:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.first_at: Optional[float] = None
        self.last_at: float = time.monotonic()
        self.error: Optional[BaseException] = None
        self.completion_tokens: Optional[int] = None
        self.chunks = 0
        self.aborted = False

    def consume(self, stream: Any) -> None:
        try:
            for chunk in stream:
                self.last_at = time.monotonic()
                if getattr(chunk, "usage", None):
                    self.completion_tokens = chunk.usage.completion_tokens
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    if self.first_at is None:
                        self.first_at = self.last_at
                    self.text.append(delta)
                    self.chunks += 1
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            if not self.aborted:
                self.error = exc

    def abort(self, stream: Any) -> None:
        self.aborted = True
        try:
            stream.close()
        except Exception:
            pass


def _loaded_state(model: dict[str, Any]) -> Optional[bool]:
    """llama-server in router mode reports per-model status; Ollama does not."""
    status = model.get("status")
    if isinstance(status, dict):
        status = status.get("value")
    if isinstance(status, str):
        return status.lower() == "loaded"
    return None


def _describe(exc: BaseException) -> str:
    if openai is not None:
        if isinstance(exc, openai.APIConnectionError):
            return f"connection error: {exc}"
        if isinstance(exc, openai.APIStatusError):
            body = str(exc.body or exc.message)
            kind = "driver/OOM error" if _DRIVER_ERROR_RE.search(body) else "server error"
            return f"{kind} (HTTP {exc.status_code}): {body[:300]}"
    msg = str(exc)
    if _DRIVER_ERROR_RE.search(msg):
        return f"driver/OOM error: {msg[:300]}"
    return f"{type(exc).__name__}: {msg[:300]}"


def build_messages(prompt: str, context: str, fmt: OutputFormat) -> list[dict[str, str]]:
    system = {OutputFormat.DIFF: DIFF_SYSTEM, OutputFormat.JSON_TOOL: JSON_TOOL_SYSTEM}.get(
        fmt, "You are an expert software engineer. Be precise and concise."
    )
    user = prompt if not context else f"{context.rstrip()}\n\n---\n\n{prompt}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def finalise(raw: str, fmt: OutputFormat, check_apply: Optional[str] = None) -> tuple[str, Optional[dict]]:
    if fmt is OutputFormat.DIFF:
        diff = extract_diff(raw)
        if check_apply:
            git_apply_check(diff, check_apply)
        return diff, None
    if fmt is OutputFormat.JSON_TOOL:
        call = extract_tool_call(raw)
        if check_apply and call["name"] == "apply_patch":
            git_apply_check(call["arguments"]["patch"], check_apply)
        return json.dumps(call, indent=2), call
    return raw.strip() + "\n", None


def build_context(paths: list[str], extra: str = "") -> str:
    parts = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        parts.append(f"### {p}\n```\n{text.rstrip()}\n```")
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

BENCHMARK_CASES = [
    ("Fix off-by-one error in binary search index calculation", "", Provider.LOCAL),
    ("Write a dataclass for UserProfile with validation", "", Provider.LOCAL),
    ("Generate a unified diff to add logging to the requests retry loop", "", Provider.LOCAL),
    ("Refactor architecture across payment, invoicing and billing modules", "", Provider.CLOUD),
    ("Find the root cause of the deadlock between the worker pool and the event loop", "", Provider.CLOUD),
    ("Explain the architecture of this module", "", Provider.LOCAL),
    ("Summarise these changes", "".join(f"diff --git a/m{i}.py b/m{i}.py\n" for i in range(5)), Provider.CLOUD),
    ("Review this file", "x = 1\n" * 6000, Provider.CLOUD),
]


def run_benchmark(config: RouterConfig) -> int:
    failures = 0
    for prompt, context, expected in BENCHMARK_CASES:
        d = classify(prompt, context, config)
        ok = d.provider is expected
        failures += not ok
        label = prompt if not context else f"{prompt} [+{len(context):,} chars context]"
        print(f"{'ok  ' if ok else 'FAIL'} {d.provider.value:<5}  {label}\n       {d.reason}")
    print(f"\n{len(BENCHMARK_CASES) - failures}/{len(BENCHMARK_CASES)} routing decisions as expected")
    return 1 if failures else 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--prompt", help="task prompt ('-' reads stdin)")
    ap.add_argument("-f", "--file", action="append", default=[], help="add a file as context (repeatable)")
    ap.add_argument("-c", "--context", default="", help="extra context text")
    ap.add_argument("-m", "--mode", choices=[f.value for f in OutputFormat], default="text")
    ap.add_argument("--force", choices=[p.value for p in Provider], help="skip heuristics")
    ap.add_argument("--check-apply", metavar="REPO", nargs="?", const=".",
                    help="require diffs to pass git apply --check in REPO (default .)")
    ap.add_argument("--dry-run", action="store_true", help="print the routing decision only")
    ap.add_argument("--json", action="store_true", help="print the full response as JSON")
    ap.add_argument("--benchmark", action="store_true", help="check routing heuristics (no network)")
    ap.add_argument("--models", action="store_true", help="list models on the local server")
    ap.add_argument("--resolve", metavar="MODEL", help="print the local model id MODEL maps to")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="[router] %(levelname)s %(message)s", stream=sys.stderr)
    config = RouterConfig()

    if args.benchmark:
        return run_benchmark(config)
    if args.models:
        router = HybridRouter(config)
        for m in router.local_models(refresh=True):
            state = _loaded_state(m)
            print(f"{m['id']}" + ("" if state is None else f"  ({'loaded' if state else 'unloaded'})"))
        return 0
    if args.resolve:
        try:
            mid, loaded = HybridRouter(config).resolve_local_model(args.resolve)
        except Exception as exc:
            print(exc, file=sys.stderr)
            return 1
        print(mid + ("" if loaded is None else (" (loaded)" if loaded else " (not loaded)")))
        return 0
    if not args.prompt:
        ap.error("--prompt is required")

    prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
    context = build_context(args.file, args.context)

    if args.dry_run:
        d = classify(prompt, context, config, args.force)
        print(json.dumps(dataclasses.asdict(d) | {"provider": d.provider.value}, indent=2))
        return 0

    router = HybridRouter(config)
    try:
        resp = router.route_and_execute(prompt, context, args.mode, args.force, args.check_apply)
    except Exception as exc:
        print(f"[router] ERROR {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(resp.to_dict(), indent=2))
    else:
        sys.stdout.write(resp.content)
        tps = resp.tokens_per_second
        print(f"[router] {resp.provider_used} {resp.model_used} {resp.latency_seconds:.1f}s"
              + (f" {tps:.1f} tok/s" if tps else "")
              + (f" FAILOVER: {resp.failover_reason}" if resp.failover else "")
              + f" | {resp.decision.reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
