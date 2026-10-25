#!/usr/bin/env bash
# Configure, provision and benchmark the local model server for the hybrid router.
#
# Works against either backend on Lebowski:
#   - Ollama:      writes env vars (optionally a systemd drop-in), pulls models via /api/pull.
#   - llama-server (llama.cpp, OpenAI API only): cannot pull; verifies the models are served.
# Then measures tokens/s through the OpenAI-compatible /v1/chat/completions endpoint.
#
# Safe to run from mini against a remote endpoint. Portable to macOS bash 3.2.
#
# Usage: ./setup_environment.sh [--endpoint URL] [--models "tag1 tag2"] [--install-systemd]
#                               [--skip-pull] [--skip-bench] [--bench-all] [--env-file PATH]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMARY_MODEL="${LOCAL_PRIMARY_MODEL:-qwen2.5-coder:32b-instruct-q8_0}"
SECONDARY_MODEL="${LOCAL_SECONDARY_MODEL:-llama3.3:70b-instruct-q4_K_M}"
MODELS="$PRIMARY_MODEL $SECONDARY_MODEL"
ENV_FILE="${HOME}/.config/hybrid-router/env"
INSTALL_SYSTEMD=0
SKIP_PULL=0
SKIP_BENCH=0
BENCH_ALL=0
ENDPOINT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --models) MODELS="$2"; PRIMARY_MODEL="${2%% *}"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --install-systemd) INSTALL_SYSTEMD=1; shift ;;
    --skip-pull) SKIP_PULL=1; shift ;;
    --skip-bench) SKIP_BENCH=1; shift ;;
    --bench-all) BENCH_ALL=1; shift ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null || die "curl is required"
command -v python3 >/dev/null || die "python3 is required"

# Resolve the endpoint the same way router.py does.
if [ -z "$ENDPOINT" ]; then
  ENDPOINT="$(cd "$SCRIPT_DIR" && python3 -c 'import router; print(router.resolve_local_base())' 2>/dev/null \
              || echo http://localhost:11434/v1)"
fi
case "$ENDPOINT" in *://*) ;; *) ENDPOINT="http://$ENDPOINT" ;; esac
ENDPOINT="${ENDPOINT%/}"
case "$ENDPOINT" in */v1) ;; *) ENDPOINT="$ENDPOINT/v1" ;; esac
ROOT="${ENDPOINT%/v1}"
info "Endpoint: $ENDPOINT"

# ---------------------------------------------------------------------------
# Step 1: environment
# ---------------------------------------------------------------------------
info "Step 1: environment"
mkdir -p "$(dirname "$ENV_FILE")"
cat > "$ENV_FILE" <<EOF
# Written by setup_environment.sh. Source from ~/.bashrc:  . "$ENV_FILE"
# Ollama server settings (ignored by llama-server; see README for its flags).
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=32768
# ROCm only. gfx1151 is the native Strix Halo ISA, so this is a no-op on ROCm builds
# that ship gfx1151 kernels and has no effect on Vulkan.
export HSA_OVERRIDE_GFX_VERSION=11.5.1
# Router client settings.
export LOCAL_API_BASE="$ENDPOINT"
export LOCAL_PRIMARY_MODEL="$PRIMARY_MODEL"
EOF
info "Wrote $ENV_FILE (source it from ~/.bashrc to persist)"

# Detect backend.
BACKEND=""
if curl -fsS -m 5 "$ROOT/api/version" >/dev/null 2>&1; then
  BACKEND=ollama
elif curl -fsS -m 5 "$ENDPOINT/models" >/dev/null 2>&1; then
  BACKEND=openai
  if curl -fsS -m 5 "$ROOT/props" >/dev/null 2>&1 || curl -fsS -m 5 "$ROOT/health" >/dev/null 2>&1; then
    BACKEND=llama-server
  fi
else
  die "nothing answering at $ROOT (tried /api/version and /v1/models)"
fi
info "Backend: $BACKEND"

if [ "$INSTALL_SYSTEMD" = 1 ]; then
  if [ "$BACKEND" != ollama ]; then
    warn "--install-systemd only applies to Ollama; skipping (backend is $BACKEND)"
  elif ! systemctl cat ollama.service >/dev/null 2>&1; then
    warn "ollama.service not found on this host; skipping systemd drop-in"
  else
    info "Installing /etc/systemd/system/ollama.service.d/hybrid-router.conf (sudo)"
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    sudo tee /etc/systemd/system/ollama.service.d/hybrid-router.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_CONTEXT_LENGTH=32768"
Environment="HSA_OVERRIDE_GFX_VERSION=11.5.1"
EOF
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      curl -fsS -m 2 "$ROOT/api/version" >/dev/null 2>&1 && break
      sleep 1
    done
  fi
fi

# ---------------------------------------------------------------------------
# Step 2: models
# ---------------------------------------------------------------------------
info "Step 2: models"
VERIFIED=""
for model in $MODELS; do
  if [ "$BACKEND" = ollama ]; then
    if [ "$SKIP_PULL" = 0 ]; then
      info "Pulling $model (this can take a while)"
      # /api/pull works for remote servers too; stream progress lines and show the status field.
      curl -fsS -N "$ROOT/api/pull" -d "{\"model\": \"$model\"}" \
        | python3 -c 'import sys, json
last = None
for line in sys.stdin:
    s = json.loads(line)
    if "error" in s: sys.exit("pull failed: " + s["error"])
    if s.get("status") != last:
        last = s.get("status"); print("    " + last, flush=True)' \
        || { warn "pull failed for $model"; continue; }
    fi
    if curl -fsS -m 10 "$ROOT/api/show" -d "{\"model\": \"$model\"}" >/dev/null 2>&1; then
      info "OK  $model"
      VERIFIED="$VERIFIED $model"
    else
      warn "MISSING $model"
    fi
  else
    # OpenAI-only server: map the Ollama tag to a served alias with router.py's logic.
    if served="$(cd "$SCRIPT_DIR" && LOCAL_API_BASE="$ENDPOINT" python3 router.py --resolve "$model")"; then
      info "OK  $model -> $served"
      VERIFIED="$VERIFIED ${served%% (*}"
    else
      warn "MISSING $model on $BACKEND (it cannot pull; place the GGUF in the server's models dir)"
    fi
  fi
done
[ -n "$VERIFIED" ] || die "no requested model is available"

# ---------------------------------------------------------------------------
# Step 3: throughput
# ---------------------------------------------------------------------------
if [ "$SKIP_BENCH" = 1 ]; then
  info "Skipping benchmark"
  exit 0
fi
info "Step 3: throughput via $ENDPOINT/chat/completions"
BENCH_MODELS="${VERIFIED# }"
[ "$BENCH_ALL" = 1 ] || BENCH_MODELS="${BENCH_MODELS%% *}"

for model in $BENCH_MODELS; do
  info "Benchmarking $model (warm-up loads the model if needed)"
  python3 - "$ENDPOINT" "$model" <<'EOF'
import json, sys, time, urllib.request

base, model = sys.argv[1], sys.argv[2]

def chat(prompt, max_tokens, timeout):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base + "/chat/completions", body,
                                 {"Content-Type": "application/json", "Authorization": "Bearer sk-local"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data, time.monotonic() - t0

try:
    _, warm = chat("Say OK.", 4, 600)
    print(f"    warm-up: {warm:.1f}s")
    data, wall = chat("Write a Python function that parses an ISO-8601 duration string into "
                      "seconds, with docstring and tests.", 256, 600)
except Exception as exc:
    sys.exit(f"    benchmark request failed: {exc}")

usage = data.get("usage") or {}
timings = data.get("timings")  # llama-server only
if timings:
    print(f"    prompt eval: {timings['prompt_n']} tok @ {timings['prompt_per_second']:.1f} tok/s")
    print(f"    generation:  {timings['predicted_n']} tok @ {timings['predicted_per_second']:.1f} tok/s")
else:
    n = usage.get("completion_tokens", 0)
    print(f"    generation:  {n} tok in {wall:.1f}s = {n / wall:.1f} tok/s (wall clock, includes prompt eval)")
print(f"    total wall:  {wall:.1f}s")
EOF
done
info "Done. Next: python3 $SCRIPT_DIR/router.py --benchmark"
