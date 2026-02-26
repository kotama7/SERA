"""Agent Function definitions for Phase 7 paper generation.

Registers: paper_outline, paper_draft, paper_reflection,
           aggregate_plot_generation, aggregate_plot_fix,
           citation_identify, citation_select, citation_bibtex.
"""

from __future__ import annotations

from sera.agent.agent_functions import (
    OutputMode,
    extract_code_block,
    parse_json_response,
    register_function,
)

# ------------------------------------------------------------------
# paper_outline
# ------------------------------------------------------------------


@register_function(
    name="paper_outline",
    description="Generate a structured paper outline from research results.",
    parameters={
        "type": "object",
        "properties": {
            "results_summary": {"type": "string"},
            "problem_description": {"type": "string"},
        },
        "required": ["results_summary"],
    },
    return_schema=None,
    output_mode=OutputMode.FREE_TEXT,
    phase="paper",
    default_temperature=0.7,
    max_retries=2,
)
def handle_paper_outline(response: str) -> str:
    """Return outline text as-is."""
    return response.strip()


# ------------------------------------------------------------------
# paper_draft
# ------------------------------------------------------------------


@register_function(
    name="paper_draft",
    description="Draft a section of the research paper.",
    parameters={
        "type": "object",
        "properties": {
            "outline": {"type": "string"},
            "section_name": {"type": "string"},
        },
        "required": ["outline"],
    },
    return_schema=None,
    output_mode=OutputMode.FREE_TEXT,
    phase="paper",
    default_temperature=0.7,
    max_retries=2,
)
def handle_paper_draft(response: str) -> str:
    """Return draft text as-is."""
    return response.strip()


# ------------------------------------------------------------------
# paper_reflection
# ------------------------------------------------------------------


@register_function(
    name="paper_reflection",
    description="Reflect on a paper draft and suggest improvements.",
    parameters={
        "type": "object",
        "properties": {
            "draft_text": {"type": "string"},
        },
        "required": ["draft_text"],
    },
    return_schema=None,
    output_mode=OutputMode.FREE_TEXT,
    phase="paper",
    default_temperature=0.7,
    max_retries=2,
)
def handle_paper_reflection(response: str) -> str:
    """Return reflection text as-is."""
    return response.strip()


# ------------------------------------------------------------------
# aggregate_plot_generation
# ------------------------------------------------------------------


@register_function(
    name="aggregate_plot_generation",
    description="Generate plotting code for aggregate figures from experiment results.",
    parameters={
        "type": "object",
        "properties": {
            "results_data": {"type": "string"},
        },
        "required": ["results_data"],
    },
    return_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "code": {"type": "string"},
            },
            "required": ["description", "code"],
        },
    },
    output_mode=OutputMode.JSON,
    phase="paper",
    default_temperature=0.5,
    max_retries=3,
    allowed_tools=["execute_code_snippet"],
    loop_config={"max_steps": 5, "tool_call_budget": 10, "timeout_sec": 120},
)
def handle_aggregate_plot_generation(response: str) -> list[dict]:
    """Parse plot generation specs from LLM response."""
    parsed = parse_json_response(response)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


# ------------------------------------------------------------------
# aggregate_plot_fix
# ------------------------------------------------------------------


@register_function(
    name="aggregate_plot_fix",
    description="Fix a broken plotting script.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "error_message": {"type": "string"},
        },
        "required": ["code", "error_message"],
    },
    return_schema=None,
    output_mode=OutputMode.CODE,
    phase="paper",
    default_temperature=0.5,
    max_retries=3,
    allowed_tools=["execute_code_snippet"],
    loop_config={"max_steps": 5, "tool_call_budget": 10, "timeout_sec": 120},
)
def handle_aggregate_plot_fix(response: str) -> str:
    """Extract fixed plotting code."""
    return extract_code_block(response, "python")


# ------------------------------------------------------------------
# citation_identify
# ------------------------------------------------------------------


@register_function(
    name="citation_identify",
    description="Identify citation needs in a paper draft.",
    parameters={
        "type": "object",
        "properties": {
            "paper_text": {"type": "string"},
        },
        "required": ["paper_text"],
    },
    return_schema=None,
    output_mode=OutputMode.FREE_TEXT,
    phase="paper",
    default_temperature=0.5,
    max_retries=2,
    allowed_tools=["semantic_scholar_search", "web_search"],
    loop_config={"max_steps": 8, "tool_call_budget": 15, "timeout_sec": 180},
)
def handle_citation_identify(response: str) -> str:
    """Return citation identification text."""
    return response.strip()


# ------------------------------------------------------------------
# citation_select
# ------------------------------------------------------------------


@register_function(
    name="citation_select",
    description="Select the best matching citation from candidate papers.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "candidates": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query", "candidates"],
    },
    return_schema=None,
    output_mode=OutputMode.FREE_TEXT,
    phase="paper",
    default_temperature=0.3,
    max_retries=2,
    allowed_tools=["semantic_scholar_search"],
    loop_config={"max_steps": 5, "tool_call_budget": 10, "timeout_sec": 120},
)
def handle_citation_select(response: str) -> str:
    """Return selection text."""
    return response.strip()


# ------------------------------------------------------------------
# citation_bibtex
# ------------------------------------------------------------------


@register_function(
    name="citation_bibtex",
    description="Generate BibTeX entry for a selected citation.",
    parameters={
        "type": "object",
        "properties": {
            "paper_info": {"type": "string"},
        },
        "required": ["paper_info"],
    },
    return_schema=None,
    output_mode=OutputMode.FREE_TEXT,
    phase="paper",
    default_temperature=0.3,
    max_retries=2,
)
def handle_citation_bibtex(response: str) -> str:
    """Return BibTeX text."""
    return response.strip()
