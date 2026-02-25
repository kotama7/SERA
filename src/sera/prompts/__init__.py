"""Prompt template package — loads prompts from co-located YAML files."""

from sera.prompts.prompt_loader import (
    get_prompt,
    get_search_category_prompts,
    get_search_prompt,
    get_template,
)

__all__ = [
    "get_prompt",
    "get_search_category_prompts",
    "get_search_prompt",
    "get_template",
]
