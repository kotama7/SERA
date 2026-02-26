"""Agent Function definitions for Phase 2 search operators.

Registers: search_draft, search_debug, search_improve.
"""

from __future__ import annotations

from sera.agent.agent_functions import (
    OutputMode,
    parse_json_response,
    register_function,
)

# ------------------------------------------------------------------
# search_draft
# ------------------------------------------------------------------

_DRAFT_RETURN_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string"},
            "experiment_config": {"type": "object"},
            "rationale": {"type": "string"},
        },
        "required": ["hypothesis", "experiment_config", "rationale"],
    },
}


@register_function(
    name="search_draft",
    description="Generate diverse experiment proposals (hypotheses) for tree search.",
    parameters={
        "type": "object",
        "properties": {
            "problem_description": {"type": "string"},
            "objective": {"type": "string"},
            "variables": {"type": "string"},
            "constraints": {"type": "string"},
            "n": {"type": "integer"},
        },
        "required": ["problem_description", "n"],
    },
    return_schema=_DRAFT_RETURN_SCHEMA,
    output_mode=OutputMode.JSON,
    phase="search",
    default_temperature=0.7,
    max_retries=3,
    allowed_tools=["get_node_info", "list_nodes", "read_metrics"],
    loop_config={"max_steps": 5, "tool_call_budget": 10, "timeout_sec": 120},
)
def handle_search_draft(response: str) -> list[dict]:
    """Parse draft proposals from LLM response."""
    parsed = parse_json_response(response)
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


# ------------------------------------------------------------------
# search_debug
# ------------------------------------------------------------------

_DEBUG_RETURN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "experiment_config": {"type": "object"},
        "experiment_code": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["hypothesis", "experiment_config", "experiment_code", "rationale"],
}


@register_function(
    name="search_debug",
    description="Fix a failed experiment by analysing the error and producing corrected code.",
    parameters={
        "type": "object",
        "properties": {
            "problem_description": {"type": "string"},
            "hypothesis": {"type": "string"},
            "experiment_config": {"type": "object"},
            "error_message": {"type": "string"},
            "experiment_code": {"type": "string"},
        },
        "required": ["problem_description", "hypothesis", "error_message"],
    },
    return_schema=_DEBUG_RETURN_SCHEMA,
    output_mode=OutputMode.JSON,
    phase="search",
    default_temperature=0.5,
    max_retries=3,
    allowed_tools=["read_experiment_log", "read_file", "execute_code_snippet"],
    loop_config={"max_steps": 5, "tool_call_budget": 10, "timeout_sec": 120},
)
def handle_search_debug(response: str) -> dict | None:
    """Parse a debug fix from LLM response."""
    parsed = parse_json_response(response)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and parsed:
        return parsed[0]
    return None


# ------------------------------------------------------------------
# search_improve
# ------------------------------------------------------------------

_IMPROVE_RETURN_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string"},
            "experiment_config": {"type": "object"},
            "rationale": {"type": "string"},
        },
        "required": ["hypothesis", "experiment_config", "rationale"],
    },
}


@register_function(
    name="search_improve",
    description="Propose atomic improvements to an evaluated experiment node.",
    parameters={
        "type": "object",
        "properties": {
            "problem_description": {"type": "string"},
            "objective": {"type": "string"},
            "parent_hypothesis": {"type": "string"},
            "parent_config": {"type": "object"},
            "n_children": {"type": "integer"},
        },
        "required": ["problem_description", "parent_hypothesis", "n_children"],
    },
    return_schema=_IMPROVE_RETURN_SCHEMA,
    output_mode=OutputMode.JSON,
    phase="search",
    default_temperature=0.7,
    max_retries=3,
    allowed_tools=["get_best_node", "read_metrics", "get_search_stats"],
    loop_config={"max_steps": 5, "tool_call_budget": 10, "timeout_sec": 120},
)
def handle_search_improve(response: str) -> list[dict]:
    """Parse improvement proposals from LLM response."""
    parsed = parse_json_response(response)
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []
