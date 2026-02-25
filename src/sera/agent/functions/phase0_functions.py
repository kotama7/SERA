"""Agent Function definitions for Phase 0 related-work collection.

Registers: query_generation, paper_clustering.
"""

from __future__ import annotations

from sera.agent.agent_functions import (
    OutputMode,
    parse_json_response,
    register_function,
)

# ------------------------------------------------------------------
# query_generation
# ------------------------------------------------------------------


@register_function(
    name="query_generation",
    description="Generate search queries for related-work collection.",
    parameters={
        "type": "object",
        "properties": {
            "problem_description": {"type": "string"},
            "existing_queries": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["problem_description"],
    },
    return_schema=None,
    output_mode=OutputMode.FREE_TEXT,
    phase="phase0",
    default_temperature=0.7,
    max_retries=2,
)
def handle_query_generation(response: str) -> str:
    """Return generated queries as text."""
    return response.strip()


# ------------------------------------------------------------------
# paper_clustering
# ------------------------------------------------------------------


@register_function(
    name="paper_clustering",
    description="Cluster collected papers into thematic groups.",
    parameters={
        "type": "object",
        "properties": {
            "papers": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["papers"],
    },
    return_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "description": {"type": "string"},
                "paper_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["label", "description", "paper_ids"],
        },
    },
    output_mode=OutputMode.JSON,
    phase="phase0",
    default_temperature=0.5,
    max_retries=3,
)
def handle_paper_clustering(response: str) -> list[dict]:
    """Parse paper clusters from LLM response."""
    parsed = parse_json_response(response)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []
