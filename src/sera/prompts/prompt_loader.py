"""YAML-based prompt loader with file-level caching.

Loads prompt templates from YAML files co-located in this package
using ``importlib.resources``, caching each file after first read.
"""

from __future__ import annotations

import importlib.resources
from typing import Any

import yaml

_cache: dict[str, dict[str, Any]] = {}


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load and cache a YAML file from the sera.prompts package."""
    if filename in _cache:
        return _cache[filename]

    ref = importlib.resources.files("sera.prompts").joinpath(filename)
    text = ref.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    _cache[filename] = data
    return data


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def get_prompt(name: str) -> str:
    """Return a prompt template from phase3_templates.yaml (experiment generator)."""
    data = _load_yaml("phase3_templates.yaml")
    return data["prompts"][name]["template"]


def get_search_prompt(name: str) -> str:
    """Return a prompt template from search.yaml (tree_ops)."""
    data = _load_yaml("search.yaml")
    return data["prompts"][name]["template"]


def get_search_category_prompts() -> dict[str, str]:
    """Return the draft_category_prompts mapping from search.yaml."""
    data = _load_yaml("search.yaml")
    return dict(data["prompts"]["draft_category_prompts"]["categories"])


def get_template(name: str) -> str:
    """Return a prompt template from the phase template YAML files.

    Searches across all phase*_templates.yaml files until found.
    """
    phase_files = [
        "phase0_templates.yaml",
        "phase1_templates.yaml",
        "phase2_templates.yaml",
        "phase3_templates.yaml",
        "phase7_templates.yaml",
        "phase8_templates.yaml",
    ]
    for filename in phase_files:
        data = _load_yaml(filename)
        prompts = data.get("prompts", {})
        if name in prompts:
            return prompts[name]["template"]
    raise KeyError(f"Prompt template '{name}' not found in any phase YAML file")
