#!/usr/bin/env python3
"""MCP server (stdio) exposing the hybrid router as tools.

Register with Claude Code, Claude Desktop or Antigravity; see README.md.
Logs go to stderr because stdout carries the MCP protocol.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal, Optional

import anyio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router import HybridRouter, RouterConfig, build_context, classify  # noqa: E402

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="[hybrid-router] %(levelname)s %(message)s")

server = _Server(
    "hybrid-router",
    instructions=(
        "Offload routine coding work (single-file edits, small diffs, boilerplate, tests) to a "
        "local model on Lebowski; heavy tasks are escalated to a cloud model automatically. "
        "Local generation is slow (single-digit tokens/s), so keep requests focused."
    ),
)
_router: Optional[HybridRouter] = None


def _get_router() -> HybridRouter:
    global _router
    if _router is None:
        _router = HybridRouter(RouterConfig())
    return _router


@server.tool()
async def ask_hybrid_coder(
    prompt: str,
    context: str = "",
    files: Optional[list[str]] = None,
    mode: Literal["text", "diff", "json_tool"] = "text",
    force: Optional[Literal["local", "cloud"]] = None,
    check_apply_repo: Optional[str] = None,
) -> dict:
    """Run a coding task through the hybrid router.

    Args:
        prompt: The task, e.g. "Add a --verbose flag to cli.py".
        context: Code or notes the task refers to, pasted inline.
        files: Absolute paths of files to include as context (read by this server).
        mode: "diff" returns a validated unified diff; "json_tool" returns one tool-call
            object; "text" returns free-form markdown.
        force: Skip heuristics and use "local" (no cloud fallback) or "cloud".
        check_apply_repo: Repo path; when set, a diff must pass `git apply --check` there.

    Returns the output plus which provider/model answered, latency, and any failover reason.
    """
    full_context = build_context(files or [], context)
    resp = await anyio.to_thread.run_sync(
        lambda: _get_router().route_and_execute(prompt, full_context, mode, force, check_apply_repo)
    )
    d = resp.to_dict()
    return {
        "content": d["content"],
        "provider": d["provider_used"],
        "model": d["model_used"],
        "latency_seconds": round(d["latency_seconds"], 2),
        "tokens_per_second": d["tokens_per_second"] and round(d["tokens_per_second"], 1),
        "failover_reason": d["failover_reason"],
        "routing_reason": d["decision"]["reason"],
    }


@server.tool()
def preview_route(prompt: str, context: str = "") -> dict:
    """Show where a task would be routed (local or cloud) and why, without running it."""
    d = classify(prompt, context, _get_router().config)
    return {"provider": d.provider.value, "reason": d.reason, "estimated_tokens": d.estimated_tokens}


@server.tool()
async def list_local_models() -> list[str]:
    """List the models the local server (Lebowski) can serve."""
    models = await anyio.to_thread.run_sync(lambda: _get_router().local_models(refresh=True))
    return [m["id"] for m in models]


if __name__ == "__main__":
    server.run()
