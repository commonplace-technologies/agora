# Hybrid coding router (Lebowski + cloud)

This harness sends routine coding tasks to a local model on **Lebowski** (an HP Z2 Mini G1a with a Ryzen AI Max+ 395 and 128 GB of unified memory). It escalates heavy tasks to a cloud model through an OpenAI-compatible API (OpenRouter or DeepSeek). If the local call times out or errors, the router fails over to the cloud and logs a warning. In `diff` and `json_tool` modes, output from either provider is validated.

| File | Purpose |
|---|---|
| `setup_environment.sh` | Writes the env file, pulls or verifies models, and measures tokens/s through `/v1/chat/completions`. |
| `router.py` | Routing engine, failover, output validation and the CLI. It can also be imported as a library. |
| `mcp_server.py` | stdio MCP server that exposes the router to Claude Code, Claude Desktop and Antigravity. |
| `tests/test_router.py` | Tests that run against in-process fake servers. No GPU or network is needed. |

## What Lebowski actually runs

The original spec assumed Ollama on ROCm. Lebowski serves port 11434 with **llama.cpp `llama-server` in router mode on Vulkan** (`--device Vulkan0 --n-gpu-layers 99`), reading models from `/mnt/data/models`. That server has no Ollama `/api/*` endpoints, only the OpenAI-style `/v1/models` and `/v1/chat/completions`, which both work. The harness handles either backend:

- `router.py` maps the Ollama tag `qwen2.5-coder:32b-instruct-q8_0` to the alias that llama-server serves (`Qwen-2.5-Coder-32B-Q8`). It does this through `/v1/models`, so the same config works against either backend.
- `setup_environment.sh` detects the backend. On Ollama it pulls models through `/api/pull`. On llama-server, which can't pull, it checks that the models are served.

Both spec tags exist in the Ollama library: `qwen2.5-coder:32b-instruct-q8_0` (35 GB) and `llama3.3:70b-instruct-q4_K_M` (43 GB).

## Setup

You need Python 3.10 or newer, because the `mcp` package requires it. The `/usr/bin/python3` that ships with macOS is 3.9, so on mini use Homebrew's Python:

```bash
cd harness
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

The endpoint is resolved in this order: `LOCAL_API_BASE`, then `OLLAMA_HOST`, then `OPENAI_BASE_URL`, then `http://localhost:11434/v1`. A bare `host:port` gets `http://` and `/v1` added, and `OLLAMA_HOST=0.0.0.0` is skipped because it's a bind address, not a destination. The exports you already have on mini therefore work unchanged.

```bash
./setup_environment.sh                                   # uses the resolved endpoint
./setup_environment.sh --endpoint 192.168.8.153:11434 --bench-all
./setup_environment.sh --install-systemd                 # Ollama hosts only: writes a systemd drop-in (sudo)
```

The script writes `~/.config/hybrid-router/env`. To persist it, add `. ~/.config/hybrid-router/env` to `~/.bashrc`. The script is plain bash and runs under macOS's bash 3.2.

For cloud escalation, set one key:

```bash
export OPENROUTER_API_KEY=...            # default: CLOUD_PROVIDER=openrouter, CLOUD_MODEL=deepseek/deepseek-chat
# or
export CLOUD_PROVIDER=deepseek DEEPSEEK_API_KEY=...   # CLOUD_MODEL defaults to deepseek-chat
# Claude via OpenRouter: CLOUD_MODEL=anthropic/<model-id from openrouter.ai/models>
```

## CLI

```bash
python3 router.py --benchmark                         # routing heuristics only, no network
python3 router.py --models                            # what the local server can serve, with load state
python3 router.py -p "Add a --verbose flag" -f cli.py --mode diff
python3 router.py -p "Add a --verbose flag" -f cli.py --mode diff --check-apply   # must pass git apply --check
python3 router.py -p "Find the root cause of this deadlock" -f worker.py          # escalates to cloud
python3 router.py -p "..." --force local              # never leaves Lebowski (no failover)
python3 router.py -p "..." --dry-run                  # print the routing decision only
git diff | python3 router.py -p - --mode text         # prompt from stdin
python3 router.py -p "..." --json                     # full response with provider, latency, tok/s
```

The output goes to stdout. A one-line summary goes to stderr, giving the provider, model, latency, tok/s and the failover reason if there was one. That means `--mode diff > fix.patch` gives you a clean patch.

Bash alias for mini (`~/.bashrc`):

```bash
alias lc='python3 "/Users/johnbacus/Projects/Commonplace Technologies/agora/harness/router.py"'
```

## Library

```python
from router import HybridRouter, OutputFormat

r = HybridRouter()                        # reads env
resp = r.route_and_execute("Add type hints", system_context=src, output_format=OutputFormat.DIFF)
resp.content, resp.provider_used, resp.model_used, resp.failover_reason, resp.tokens_per_second
```

## Using it from Claude and Antigravity (MCP)

`mcp_server.py` exposes three tools:

- `ask_hybrid_coder(prompt, context, files, mode, force, check_apply_repo)`
- `preview_route(prompt, context)`
- `list_local_models()`

Every example below uses the venv's interpreter, so replace `PY` with its absolute path, `/Users/johnbacus/Projects/Commonplace Technologies/agora/harness/.venv/bin/python`.

**Claude Code**

```bash
claude mcp add --scope user hybrid-router \
  -e LOCAL_API_BASE=http://192.168.8.153:11434/v1 -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -- "$PY" "/Users/johnbacus/Projects/Commonplace Technologies/agora/harness/mcp_server.py"
```

A local generation of a few hundred tokens takes a minute or more. Raise the MCP tool timeout, for example `export MCP_TOOL_TIMEOUT=900000` (milliseconds), before you start `claude`.

**Claude Desktop.** Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`, then restart the app:

```json
{
  "mcpServers": {
    "hybrid-router": {
      "command": "/Users/johnbacus/Projects/Commonplace Technologies/agora/harness/.venv/bin/python",
      "args": ["/Users/johnbacus/Projects/Commonplace Technologies/agora/harness/mcp_server.py"],
      "env": { "LOCAL_API_BASE": "http://192.168.8.153:11434/v1", "OPENROUTER_API_KEY": "..." }
    }
  }
}
```

**Antigravity.** Open its MCP server settings (raw config), add the same `mcpServers` entry, and refresh. Antigravity can also call llama-server directly as an OpenAI-compatible provider: set the base URL to `http://192.168.8.153:11434/v1`, use any API key, and pick the model `Qwen-2.5-Coder-32B-Q8`. That path skips the router's failover and validation.

The router has no `--mcp` flag. Run `mcp_server.py` instead.

## Tests

```bash
python3 -m pytest harness/tests -q      # from the repo root
```

The tests run fake OpenAI-compatible servers in-process. They cover alias resolution, all three timeout paths, OOM-style 500s, unreachable servers, missing models, forced-local behaviour, diff repair followed by failover, and JSON mode. They do not exercise a real model, so on Lebowski also run `./setup_environment.sh` and `python3 router.py -p "Write is_palindrome" --mode text -v`.

## Design choices where the spec was underspecified

- **The timeout is measured to the first token, not over the whole call.** At about 6.6 tok/s on Lebowski, a 45 s ceiling on the whole call would cap local output near 300 tokens, so most real diffs would fail over. Instead, local calls stream under three watchdogs:
  - `LOCAL_TIMEOUT_SECONDS=45` limits the wait for the first token. That wait covers prompt evaluation, and at the observed ~142 tok/s, 45 s is about 6,400 prompt tokens.
  - `LOCAL_IDLE_TIMEOUT_SECONDS=20` limits the gap between chunks.
  - `LOCAL_TOTAL_TIMEOUT_SECONDS=900` is a hard cap on the call.
- **Model load gets more time.** If `/v1/models` reports the target as not loaded (llama-server's router mode does this), the first-token limit becomes `LOCAL_LOAD_TIMEOUT_SECONDS=240`. A cold 35 GB load would otherwise always fail over.
- **OOM and driver errors are detected from the outside.** Over HTTP there is no structured OOM signal. The router fails over on connection errors and dropped streams (a dead worker), on any 5xx, and on error bodies that match Vulkan, HIP, ROCm, ggml, out-of-memory, failed-to-load or context-overflow patterns. It does not fail over on other 4xx responses, since those are request bugs that the cloud would reject too.
- **Failover discards partial output.** A local stream that dies halfway is thrown away, and the cloud answers from scratch.
- **Heuristics.** Tokens are estimated at 4 characters each, which undercounts code by 10–25%. Complexity phrases are matched as whole phrases, so "architecture" alone stays local and only "refactor architecture" escalates. A context that holds 4 or more files (counted from diff headers or `### path` headings) also escalates. All thresholds can be set through env.
- **Validation and repair.** Diffs are checked structurally: headers, plus hunk line counts that match their `@@` ranges. `--check-apply` adds a real `git apply --check`. If validation fails, the router asks the same provider once to correct it. If local output is still invalid after that, it fails over.
- **JSON tool mode.** Locally, the request sends `response_format: json_schema`, which llama-server compiles into a grammar, so the output is constrained as it's generated. The cloud request uses `json_object`, which is the most widely supported option, and the tool schema is then enforced by validation. The tools are `apply_patch`, `read_file`, `execute_command` and `final_answer`.

## Issues seen on Lebowski (2026-10-25, over the network only)

- Inference was broken. Requests to the loaded `Qwen-2.5-Coder-32B-Q8` returned "proxy error: Could not establish connection", which means the worker had died. A request for `Phi-4-14B` returned HTTP 500 "failed to load". Check `journalctl` for the llama-server unit.
- `Qwen3.5-122B-A10B-Q4_K_M` is a 3-part split GGUF, but the router lists each part as its own model. Only `-00001-of-00003` is loadable. Use a presets file, or move the parts into a subdirectory.
- No per-model flags are set: no `--jinja` (needed for native tool calling), no `-c`, no `-np`. The server reports 32768 context across 4 slots. If the KV cache isn't unified, each slot gets 8192, and prompts near the router's 8,000-token threshold will hit context errors and fail over. Set `-np 1` (the llama-server equivalent of `OLLAMA_NUM_PARALLEL=1`) or lower `ROUTER_TOKEN_THRESHOLD`.
- `max_instances` is 1, so a request for the 70B model unloads the coder model, and the next coder request pays the reload cost.
- `HSA_OVERRIDE_GFX_VERSION=11.5.1` is in the env file because the spec asks for it. It has no effect on Vulkan, and on ROCm it matches the native gfx1151 target, so it only matters if you switch to a ROCm build.
