"""Agent Function definitions for Phase 1 spec generation.

Registers: spec_generation_problem, spec_generation_plan.
"""

from __future__ import annotations

from sera.agent.agent_functions import (
    OutputMode,
    parse_json_response,
    register_function,
)

# ------------------------------------------------------------------
# spec_generation_problem
# ------------------------------------------------------------------


@register_function(
    name="spec_generation_problem",
    description="Generate a ProblemSpec JSON from Input-1 and related work context.",
    parameters={
        "type": "object",
        "properties": {
            "context": {"type": "string"},
            "spec_type": {"type": "string"},
            "schema_description": {"type": "string"},
        },
        "required": ["context"],
    },
    return_schema={
        "type": "object",
        "properties": {
            "problem_spec": {"type": "object"},
        },
    },
    output_mode=OutputMode.JSON,
    phase="spec",
    default_temperature=0.7,
    max_retries=3,
)
def handle_spec_generation_problem(response: str) -> dict | None:
    """Parse ProblemSpec JSON from LLM response."""
    return parse_json_response(response)


# ------------------------------------------------------------------
# spec_generation_plan
# ------------------------------------------------------------------


@register_function(
    name="spec_generation_plan",
    description="Generate a PlanSpec JSON from Input-1 and ProblemSpec.",
    parameters={
        "type": "object",
        "properties": {
            "context": {"type": "string"},
            "spec_type": {"type": "string"},
            "schema_description": {"type": "string"},
        },
        "required": ["context"],
    },
    return_schema={
        "type": "object",
        "properties": {
            "plan_spec": {"type": "object"},
        },
    },
    output_mode=OutputMode.JSON,
    phase="spec",
    default_temperature=0.7,
    max_retries=3,
)
def handle_spec_generation_plan(response: str) -> dict | None:
    """Parse PlanSpec JSON from LLM response."""
    return parse_json_response(response)
