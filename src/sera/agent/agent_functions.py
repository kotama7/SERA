"""Agent Function Registry: unified schema definitions for all SERA LLM calls.

Provides:
- ``AgentFunction`` / ``AgentFunctionRegistry`` / ``REGISTRY``
- ``register_function`` decorator
- Common parse utilities (``parse_json_response``, ``extract_code_block``,
  ``validate_against_schema``)

See task/22_agent_functions.md (section 28) for full specification.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class OutputMode(Enum):
    """How the LLM response should be interpreted."""

    JSON = "json"
    CODE = "code"
    FREE_TEXT = "free_text"


@dataclass(frozen=True)
class AgentFunction:
    """Schema definition for a single LLM function call.

    Attributes
    ----------
    name : str
        Unique snake_case identifier.
    description : str
        Human-readable description shown to the LLM / used in tool schemas.
    parameters : dict
        JSON Schema describing the function's input parameters
        (OpenAI function-calling format).
    return_schema : dict | None
        JSON Schema for the expected return value.  Used for validation
        and for prompt augmentation with local providers.
    output_mode : OutputMode
        How the raw LLM response should be parsed.
    phase : str
        Logical grouping: "search", "execution", "spec", "paper",
        "evaluation", "phase0".
    default_temperature : float
        Sampling temperature unless overridden by the caller.
    max_retries : int
        Maximum retry count on parse / validation failure.
    handler : Callable | None
        Optional post-processor ``handler(raw_response) -> parsed_value``.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    return_schema: dict[str, Any] | None = None
    output_mode: OutputMode = OutputMode.JSON
    phase: str = ""
    default_temperature: float = 0.7
    max_retries: int = 3
    handler: Callable[..., Any] | None = None


class AgentFunctionRegistry:
    """Central registry of all ``AgentFunction`` definitions."""

    def __init__(self) -> None:
        self._functions: dict[str, AgentFunction] = {}

    def register(self, func: AgentFunction) -> None:
        """Register a function. Raises ``ValueError`` on duplicate name."""
        if func.name in self._functions:
            raise ValueError(f"Duplicate AgentFunction name: {func.name!r}")
        self._functions[func.name] = func

    def get(self, name: str) -> AgentFunction:
        """Return the function with *name*, or raise ``KeyError``."""
        try:
            return self._functions[name]
        except KeyError:
            raise KeyError(f"AgentFunction not found: {name!r}") from None

    def list_all(self) -> list[AgentFunction]:
        """Return all registered functions."""
        return list(self._functions.values())

    def list_by_phase(self, phase: str) -> list[AgentFunction]:
        """Return functions belonging to *phase*."""
        return [f for f in self._functions.values() if f.phase == phase]

    # ------------------------------------------------------------------
    # Schema conversion helpers
    # ------------------------------------------------------------------

    def to_openai_tools(self, names: list[str] | None = None) -> list[dict]:
        """Convert selected (or all) functions to OpenAI tool-calling format."""
        funcs = self._resolve(names)
        tools: list[dict] = []
        for f in funcs:
            tool: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": f.name,
                    "description": f.description,
                    "parameters": f.parameters or {"type": "object", "properties": {}},
                },
            }
            tools.append(tool)
        return tools

    def to_anthropic_tools(self, names: list[str] | None = None) -> list[dict]:
        """Convert selected (or all) functions to Anthropic tool format."""
        funcs = self._resolve(names)
        tools: list[dict] = []
        for f in funcs:
            tool: dict[str, Any] = {
                "name": f.name,
                "description": f.description,
                "input_schema": f.parameters or {"type": "object", "properties": {}},
            }
            tools.append(tool)
        return tools

    def to_prompt_schema(self, names: list[str] | None = None) -> str:
        """Build a text description of function schemas for prompt injection."""
        funcs = self._resolve(names)
        parts: list[str] = []
        for f in funcs:
            schema_text = json.dumps(f.return_schema, indent=2) if f.return_schema else "free text"
            parts.append(
                f"Function: {f.name}\n"
                f"  Description: {f.description}\n"
                f"  Output mode: {f.output_mode.value}\n"
                f"  Return schema: {schema_text}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, names: list[str] | None) -> list[AgentFunction]:
        if names is None:
            return list(self._functions.values())
        return [self.get(n) for n in names]


# Singleton registry
REGISTRY = AgentFunctionRegistry()


def register_function(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    return_schema: dict[str, Any] | None = None,
    output_mode: OutputMode = OutputMode.JSON,
    phase: str = "",
    default_temperature: float = 0.7,
    max_retries: int = 3,
) -> Callable:
    """Decorator that registers a handler function in ``REGISTRY``.

    The decorated callable becomes the ``handler`` of the ``AgentFunction``.

    Usage::

        @register_function("search_draft", "Draft new hypotheses", ...)
        def _handle_search_draft(response: str) -> list[dict]:
            return parse_json_response(response) or []
    """

    def decorator(fn: Callable) -> Callable:
        func = AgentFunction(
            name=name,
            description=description,
            parameters=parameters or {},
            return_schema=return_schema,
            output_mode=output_mode,
            phase=phase,
            default_temperature=default_temperature,
            max_retries=max_retries,
            handler=fn,
        )
        REGISTRY.register(func)
        return fn

    return decorator


# =====================================================================
# Common parse utilities (consolidating tree_ops._parse_json_response
# and experiment_generator._extract_code)
# =====================================================================


def parse_json_response(response: str) -> dict | list | None:
    """Extract JSON from an LLM response using a 3-stage fallback.

    1. `````json ... ``` `` fenced block
    2. Raw ``json.loads`` of the entire response
    3. Regex search for ``[...]`` or ``{...}``

    Returns ``None`` if no valid JSON is found.
    """
    # Stage 1: fenced JSON block
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Stage 2: raw parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Stage 3: regex for array or object
    for pattern in [r"\[.*\]", r"\{.*\}"]:
        m = re.search(pattern, response, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    return None


def extract_code_block(response: str, language: str = "python") -> str:
    """Extract a fenced code block from *response*.

    Priority:
    1. Language-specific fence (e.g. ````` ```python ... ``` `````)
    2. Generic fence (````` ``` ... ``` `````)
    3. Raw response text
    """
    # Language-specific
    pattern = rf"```{re.escape(language)}\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Generic
    match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response.strip()


def validate_against_schema(data: Any, schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Lightweight JSON Schema validation.

    Returns ``(is_valid, errors)`` where *errors* is a list of human-readable
    error messages.  This intentionally avoids a ``jsonschema`` dependency.
    """
    errors: list[str] = []

    if not schema:
        return True, errors

    expected_type = schema.get("type")
    if expected_type:
        type_map: dict[str, type | tuple[type, ...]] = {
            "object": dict,
            "array": list,
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
        }
        py_type = type_map.get(expected_type)
        if py_type and not isinstance(data, py_type):
            errors.append(f"Expected type {expected_type}, got {type(data).__name__}")
            return False, errors

    if expected_type == "object" and isinstance(data, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                errors.append(f"Missing required key: {key!r}")
        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key in data:
                ok, sub_errors = validate_against_schema(data[key], prop_schema)
                if not ok:
                    errors.extend(f"{key}.{e}" for e in sub_errors)

    if expected_type == "array" and isinstance(data, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                ok, sub_errors = validate_against_schema(item, items_schema)
                if not ok:
                    errors.extend(f"[{i}].{e}" for e in sub_errors)

    return (len(errors) == 0), errors
