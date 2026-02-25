"""Agent Function definitions for Phase 8 paper evaluation.

Registers: paper_review, paper_review_reflection, meta_review.
"""

from __future__ import annotations

from sera.agent.agent_functions import (
    OutputMode,
    register_function,
)

# ------------------------------------------------------------------
# paper_review
# ------------------------------------------------------------------


@register_function(
    name="paper_review",
    description="Generate a structured review of the research paper.",
    parameters={
        "type": "object",
        "properties": {
            "paper_text": {"type": "string"},
            "review_guidelines": {"type": "string"},
        },
        "required": ["paper_text"],
    },
    return_schema=None,
    output_mode=OutputMode.FREE_TEXT,
    phase="evaluation",
    default_temperature=0.7,
    max_retries=2,
)
def handle_paper_review(response: str) -> str:
    """Return review text."""
    return response.strip()


# ------------------------------------------------------------------
# paper_review_reflection
# ------------------------------------------------------------------


@register_function(
    name="paper_review_reflection",
    description="Reflect on a paper review and refine it.",
    parameters={
        "type": "object",
        "properties": {
            "review_text": {"type": "string"},
            "paper_text": {"type": "string"},
        },
        "required": ["review_text"],
    },
    return_schema=None,
    output_mode=OutputMode.FREE_TEXT,
    phase="evaluation",
    default_temperature=0.7,
    max_retries=2,
)
def handle_paper_review_reflection(response: str) -> str:
    """Return refined review text."""
    return response.strip()


# ------------------------------------------------------------------
# meta_review
# ------------------------------------------------------------------


@register_function(
    name="meta_review",
    description="Generate a meta-review synthesising individual reviews.",
    parameters={
        "type": "object",
        "properties": {
            "reviews": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reviews"],
    },
    return_schema=None,
    output_mode=OutputMode.FREE_TEXT,
    phase="evaluation",
    default_temperature=0.7,
    max_retries=2,
)
def handle_meta_review(response: str) -> str:
    """Return meta-review text."""
    return response.strip()
