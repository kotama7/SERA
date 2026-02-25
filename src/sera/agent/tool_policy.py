"""Tool execution policy: path constraints, rate limits, and safety controls.

See task/23_tool_execution.md section 29.6 for specification.
"""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolPolicy:
    """Policy governing tool execution safety and resource limits."""

    # Phase-specific tool allow-lists (phase_name -> list of tool names)
    phase_allowed_tools: dict[str, list[str]] = field(default_factory=dict)

    # File I/O limits
    max_file_read_bytes: int = 1_000_000       # 1 MB
    max_file_write_bytes: int = 500_000         # 500 KB
    max_output_tokens: int = 2000               # observation truncation

    # Write access control
    allowed_write_dirs: list[str] = field(
        default_factory=lambda: ["runs/", "paper/", "outputs/"]
    )
    blocked_write_patterns: list[str] = field(
        default_factory=lambda: ["specs/*.yaml", "*.lock", "*.jsonl"]
    )

    # Shell command whitelist
    allowed_shell_commands: list[str] = field(
        default_factory=lambda: ["pip", "python", "ls", "cat", "wc"]
    )

    # Rate limiting for external API calls
    api_rate_limit_per_minute: int = 30
    api_rate_limit_burst: int = 5

    # NetworkConfig integration
    require_network_allowed: bool = True
    require_api_allowed: bool = True

    def __post_init__(self) -> None:
        self._api_call_timestamps: list[float] = []

    def check_tool_allowed(self, tool_name: str, phase: str | None = None) -> tuple[bool, str]:
        """Check whether *tool_name* is allowed in the given *phase*.

        Returns (allowed, reason).
        """
        if phase and phase in self.phase_allowed_tools:
            allowed = self.phase_allowed_tools[phase]
            if tool_name not in allowed:
                return False, f"Tool {tool_name!r} not allowed in phase {phase!r}"
        return True, ""

    def check_write_path(self, relative_path: str) -> tuple[bool, str]:
        """Check whether writing to *relative_path* is permitted."""
        for pattern in self.blocked_write_patterns:
            if fnmatch.fnmatch(relative_path, pattern):
                return False, f"Write blocked by pattern: {pattern}"

        for allowed_dir in self.allowed_write_dirs:
            if relative_path.startswith(allowed_dir):
                return True, ""

        return False, f"Write not allowed: {relative_path} not in {self.allowed_write_dirs}"

    def check_shell_command(self, command: str) -> tuple[bool, str]:
        """Check whether the shell command is in the whitelist."""
        executable = command.strip().split()[0] if command.strip() else ""
        if executable not in self.allowed_shell_commands:
            return False, f"Shell command {executable!r} not in whitelist"
        return True, ""

    def check_api_rate_limit(self) -> tuple[bool, str]:
        """Check whether an external API call is within rate limits."""
        now = time.monotonic()
        # Remove timestamps older than 60 seconds
        self._api_call_timestamps = [
            t for t in self._api_call_timestamps if now - t < 60
        ]
        if len(self._api_call_timestamps) >= self.api_rate_limit_per_minute:
            return False, "API rate limit exceeded"
        return True, ""

    def record_api_call(self) -> None:
        """Record that an API call was made (for rate limiting)."""
        self._api_call_timestamps.append(time.monotonic())

    def resolve_safe_path(self, workspace: Path, relative_path: str) -> Path:
        """Resolve *relative_path* within *workspace*, preventing traversal.

        Raises ``PermissionError`` if the resolved path escapes the workspace.
        """
        resolved = (workspace / relative_path).resolve()
        workspace_resolved = workspace.resolve()
        if not str(resolved).startswith(str(workspace_resolved)):
            raise PermissionError(f"Path traversal attempt: {relative_path}")
        return resolved
