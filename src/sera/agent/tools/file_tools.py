"""File I/O tool handlers per section 29.2.3.

Provides sandboxed read/write access within the SERA workspace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sera.agent.tool_policy import ToolPolicy


async def handle_read_file(
    args: dict[str, Any],
    workspace: Path,
    policy: ToolPolicy,
) -> dict[str, Any]:
    """Read a file within the workspace."""
    relative_path = args["path"]
    path = policy.resolve_safe_path(workspace, relative_path)

    if not path.exists():
        return {"error": f"File not found: {relative_path}", "content": None}
    if not path.is_file():
        return {"error": f"Not a file: {relative_path}", "content": None}

    size = path.stat().st_size
    if size > policy.max_file_read_bytes:
        content = path.read_text(encoding="utf-8", errors="replace")[:policy.max_file_read_bytes]
        return {"content": content, "truncated": True, "size_bytes": size}

    content = path.read_text(encoding="utf-8", errors="replace")
    return {"content": content, "truncated": False, "size_bytes": size}


async def handle_write_file(
    args: dict[str, Any],
    workspace: Path,
    policy: ToolPolicy,
) -> dict[str, Any]:
    """Write a file within the workspace (subject to policy constraints)."""
    relative_path = args["path"]
    content = args["content"]

    # Check write policy
    ok, reason = policy.check_write_path(relative_path)
    if not ok:
        return {"success": False, "error": reason}

    path = policy.resolve_safe_path(workspace, relative_path)
    content_bytes = content.encode("utf-8")

    if len(content_bytes) > policy.max_file_write_bytes:
        return {"success": False, "error": f"Content exceeds max write size ({policy.max_file_write_bytes} bytes)"}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content_bytes)
    return {"success": True, "path": relative_path, "size_bytes": len(content_bytes)}


async def handle_read_metrics(
    args: dict[str, Any],
    workspace: Path,
    policy: ToolPolicy,
) -> dict[str, Any]:
    """Read metrics.json for a specific node."""
    node_id = args["node_id"]
    metrics_path = policy.resolve_safe_path(workspace, f"runs/{node_id}/metrics.json")

    if not metrics_path.exists():
        return {"error": f"No metrics.json for node {node_id}", "metrics": None}

    try:
        metrics = json.loads(metrics_path.read_text())
        return {"metrics": metrics, "node_id": node_id}
    except (json.JSONDecodeError, OSError) as e:
        return {"error": str(e), "metrics": None}


async def handle_read_experiment_log(
    args: dict[str, Any],
    workspace: Path,
    policy: ToolPolicy,
) -> dict[str, Any]:
    """Read stdout.log or stderr.log for a specific node."""
    node_id = args["node_id"]
    log_type = args.get("log_type", "stderr")  # "stdout" or "stderr"

    if log_type not in ("stdout", "stderr"):
        return {"error": f"Invalid log_type: {log_type}"}

    log_path = policy.resolve_safe_path(workspace, f"runs/{node_id}/{log_type}.log")

    if not log_path.exists():
        return {"error": f"No {log_type}.log for node {node_id}", "content": None}

    content = log_path.read_text(encoding="utf-8", errors="replace")
    if len(content) > policy.max_file_read_bytes:
        # Return tail for logs (most useful part)
        content = content[-policy.max_file_read_bytes:]
        return {"content": content, "truncated": True, "log_type": log_type}

    return {"content": content, "truncated": False, "log_type": log_type}


async def handle_list_directory(
    args: dict[str, Any],
    workspace: Path,
    policy: ToolPolicy,
) -> dict[str, Any]:
    """List contents of a directory within the workspace."""
    relative_path = args.get("path", ".")
    path = policy.resolve_safe_path(workspace, relative_path)

    if not path.exists():
        return {"error": f"Directory not found: {relative_path}", "entries": []}
    if not path.is_dir():
        return {"error": f"Not a directory: {relative_path}", "entries": []}

    entries = []
    for item in sorted(path.iterdir()):
        entry: dict[str, Any] = {
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
        }
        if item.is_file():
            entry["size_bytes"] = item.stat().st_size
        entries.append(entry)

    return {"entries": entries[:200], "total": len(entries), "truncated": len(entries) > 200}
