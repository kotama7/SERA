"""Tool execution policy: path constraints, rate limits, and safety controls.

See task/23_tool_execution.md section 29.6 for specification.
"""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolPolicy:
    """Policy governing tool execution safety and resource limits."""

    # Global tool-system enabled flag (mirrors ToolConfig.enabled / PlanSpec.tools.enabled)
    tools_enabled: bool = True

    # Per-tool disabled set (tool names that are individually disabled)
    disabled_tools: set[str] = field(default_factory=set)

    # Phase-specific tool allow-lists (phase_name -> list of tool names)
    phase_allowed_tools: dict[str, list[str]] = field(default_factory=dict)

    # File I/O limits
    max_file_read_bytes: int = 1_000_000  # 1 MB
    max_file_write_bytes: int = 500_000  # 500 KB
    max_output_tokens: int = 2000  # observation truncation

    # Write access control
    allowed_write_dirs: list[str] = field(default_factory=lambda: ["runs/", "paper/", "outputs/"])
    blocked_write_patterns: list[str] = field(default_factory=lambda: ["specs/*.yaml", "*.lock", "*.jsonl"])

    # Shell command whitelist
    allowed_shell_commands: list[str] = field(default_factory=lambda: ["pip", "python", "ls", "cat", "wc"])

    # Build tool commands (§7.3.2.7) — allowed only when compiled=True
    allowed_build_commands: list[str] = field(
        default_factory=lambda: [
            "g++", "gcc", "clang++", "clang",  # C/C++ compilers
            "cargo", "rustc",                    # Rust
            "go",                                # Go
            "make", "cmake",                     # Build systems
        ]
    )
    compiled_language: bool = False  # Set True to enable build commands

    # Rate limiting for external API calls
    api_rate_limit_per_minute: int = 30
    api_rate_limit_burst: int = 5

    # NetworkConfig integration fields
    allow_network: bool = True
    allow_api_calls: bool = True
    allowed_domains: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._api_call_timestamps: list[float] = []

    @classmethod
    def from_specs(cls, plan_spec: Any = None, resource_spec: Any = None) -> "ToolPolicy":
        """Create a ToolPolicy from spec objects.

        Reads ``plan_spec.tools.enabled``, ``plan_spec.tools.api_rate_limit_per_minute``,
        ``plan_spec.agent_commands.tools.phase_tool_map``, and
        ``resource_spec.network`` to populate the policy.
        """
        kwargs: dict[str, Any] = {}

        if plan_spec is not None:
            # ToolConfig level
            tools_cfg = getattr(plan_spec, "tools", None)
            if tools_cfg is not None:
                kwargs["tools_enabled"] = getattr(tools_cfg, "enabled", True)
                kwargs["api_rate_limit_per_minute"] = getattr(tools_cfg, "api_rate_limit_per_minute", 30)

            # AgentCommands level — phase_tool_map
            agent_commands = getattr(plan_spec, "agent_commands", None)
            if agent_commands is not None:
                ac_tools = getattr(agent_commands, "tools", None)
                if ac_tools is not None:
                    ptm = getattr(ac_tools, "phase_tool_map", None)
                    if ptm:
                        kwargs["phase_allowed_tools"] = dict(ptm)

        if resource_spec is not None:
            network_cfg = getattr(resource_spec, "network", None)
            if network_cfg is not None:
                kwargs["allow_network"] = getattr(network_cfg, "allow_internet", True)
                kwargs["allow_api_calls"] = getattr(network_cfg, "allow_api_calls", True)

        return cls(**kwargs)

    @classmethod
    def from_specs_with_problem(
        cls,
        plan_spec: Any = None,
        resource_spec: Any = None,
        problem_spec: Any = None,
    ) -> "ToolPolicy":
        """Create a ToolPolicy from spec objects, including language config."""
        policy = cls.from_specs(plan_spec, resource_spec)
        if problem_spec is not None:
            lang = getattr(problem_spec, "language", None)
            if lang is not None:
                policy.compiled_language = getattr(lang, "compiled", False)
        return policy

    def check_tool_allowed(self, tool_name: str, phase: str | None = None) -> tuple[bool, str]:
        """Check whether *tool_name* is allowed in the given *phase*.

        Returns (allowed, reason).
        """
        # Global tools-enabled gate
        if not self.tools_enabled:
            return False, "Tool execution is globally disabled (tools.enabled=false)"

        # Per-tool disabled check
        if tool_name in self.disabled_tools:
            return False, f"Tool {tool_name!r} is individually disabled"

        # Phase-level restriction
        if phase and phase in self.phase_allowed_tools:
            allowed = self.phase_allowed_tools[phase]
            if tool_name not in allowed:
                return False, f"Tool {tool_name!r} not allowed in phase {phase!r}"
        return True, ""

    def check_network_allowed(self, tool_name: str) -> tuple[bool, str]:
        """Check whether network access is permitted for the given tool.

        Enforces ``NetworkConfig.allow_internet`` and ``allow_api_calls``
        from ``ResourceSpec.network``.
        """
        # Search / external API tools require network access
        _API_TOOLS = {
            "semantic_scholar_search",
            "semantic_scholar_references",
            "semantic_scholar_citations",
            "crossref_search",
            "arxiv_search",
            "web_search",
        }
        if tool_name in _API_TOOLS:
            if not self.allow_network:
                return False, f"Network access disabled (allow_internet=false); tool {tool_name!r} blocked"
            if not self.allow_api_calls:
                return False, f"API calls disabled (allow_api_calls=false); tool {tool_name!r} blocked"
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
        """Check whether the shell command is in the whitelist.

        Build tool commands (g++, cargo, etc.) are allowed only when
        ``compiled_language`` is True (§7.3.2.7).
        """
        executable = command.strip().split()[0] if command.strip() else ""
        if executable in self.allowed_shell_commands:
            return True, ""
        if self.compiled_language and executable in self.allowed_build_commands:
            return True, ""
        return False, f"Shell command {executable!r} not in whitelist"

    def check_api_rate_limit(self) -> tuple[bool, str]:
        """Check whether an external API call is within rate limits."""
        now = time.monotonic()
        # Remove timestamps older than 60 seconds
        self._api_call_timestamps = [t for t in self._api_call_timestamps if now - t < 60]
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
