"""Tests for the YAML-based prompt loader."""

from __future__ import annotations

import pytest

from sera.prompts import get_prompt, get_search_category_prompts, get_search_prompt, get_template
from sera.prompts.prompt_loader import _cache


# ---------------------------------------------------------------------------
# Loader basics
# ---------------------------------------------------------------------------


class TestSearchPrompts:
    """Verify all search.yaml prompts load correctly."""

    @pytest.mark.parametrize("name", ["draft", "draft_category", "debug", "improve"])
    def test_search_prompt_loads(self, name: str):
        result = get_search_prompt(name)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_category_prompts(self):
        cats = get_search_category_prompts()
        assert set(cats.keys()) == {"baseline", "open_problem", "novel"}
        for v in cats.values():
            assert isinstance(v, str)
            assert len(v) > 10

    def test_draft_format_placeholders(self):
        prompt = get_search_prompt("draft")
        assert "{problem_description}" in prompt
        assert "{objective}" in prompt
        assert "{variables}" in prompt
        assert "{constraints}" in prompt
        assert "{existing_context}" in prompt
        assert "{n}" in prompt

    def test_debug_format_placeholders(self):
        prompt = get_search_prompt("debug")
        assert "{problem_description}" in prompt
        assert "{hypothesis}" in prompt
        assert "{experiment_config}" in prompt
        assert "{error_message}" in prompt
        assert "{experiment_code}" in prompt
        assert "{code_block_tag}" in prompt

    def test_improve_format_placeholders(self):
        prompt = get_search_prompt("improve")
        assert "{parent_hypothesis}" in prompt
        assert "{parent_config}" in prompt
        assert "{n_children}" in prompt

    def test_draft_format_call(self):
        """Verify .format() works without KeyError."""
        prompt = get_search_prompt("draft")
        result = prompt.format(
            problem_description="test",
            objective="maximize accuracy",
            variables="lr, batch_size",
            constraints="time < 1h",
            existing_context="",
            n=3,
        )
        assert "test" in result
        assert "maximize accuracy" in result


class TestTemplatePrompts:
    """Verify all phase template YAML prompts load correctly."""

    ALL_TEMPLATE_NAMES = [
        "query_generation",
        "paper_clustering",
        "relevance_scoring",
        "spec_generation",
        "draft",
        "debug",
        "improve",
        "experiment_code",
        "paper_outline",
        "paper_full_generation",
        "paper_writeup_reflection",
        "citation_search",
        "citation_select",
        "plot_aggregation",
        "vlm_figure_description",
        "vlm_figure_caption_review",
        "vlm_duplicate_detection",
        "paper_evaluation",
        "reviewer_reflection",
        "meta_review",
        "paper_revision",
    ]

    @pytest.mark.parametrize("name", ALL_TEMPLATE_NAMES)
    def test_template_loads(self, name: str):
        result = get_template(name)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_template("nonexistent")


class TestExperimentPrompt:
    """Verify experiment_code_runtime prompt loads."""

    def test_loads(self):
        prompt = get_prompt("experiment_code_runtime")
        assert isinstance(prompt, str)
        assert "{language_name}" in prompt
        assert "{metric_name}" in prompt

    def test_format_call(self):
        prompt = get_prompt("experiment_code_runtime")
        result = prompt.format(
            language_name="python",
            problem_description="classify iris",
            objective="accuracy",
            direction="maximize",
            hypothesis="use SVM",
            experiment_config="{}",
            data_location="./data",
            template_section="",
            seed_arg_description="--seed <int>",
            metric_name="accuracy",
            higher_is_better="true",
            code_block_tag="python",
        )
        assert "python" in result
        assert "classify iris" in result


# ---------------------------------------------------------------------------
# Backward compatibility: TEMPLATE_REGISTRY
# ---------------------------------------------------------------------------


class TestTemplateRegistryCompat:
    """Verify prompt_templates.py still exports the same interface."""

    def test_registry_keys(self):
        from sera.agent.prompt_templates import TEMPLATE_REGISTRY

        expected_keys = {
            "query_generation",
            "paper_clustering",
            "relevance_scoring",
            "spec_generation",
            "draft",
            "debug",
            "improve",
            "experiment_code",
            "paper_outline",
            "paper_full_generation",
            "paper_writeup_reflection",
            "citation_search",
            "citation_select",
            "plot_aggregation",
            "vlm_figure_description",
            "vlm_figure_caption_review",
            "vlm_duplicate_detection",
            "paper_evaluation",
            "reviewer_reflection",
            "meta_review",
            "paper_revision",
        }
        assert set(TEMPLATE_REGISTRY.keys()) == expected_keys

    def test_registry_values_are_strings(self):
        from sera.agent.prompt_templates import TEMPLATE_REGISTRY

        for name, tmpl in TEMPLATE_REGISTRY.items():
            assert isinstance(tmpl, str), f"{name} is not a string"
            assert len(tmpl) > 50, f"{name} is too short"

    def test_module_level_constants_match_registry(self):
        from sera.agent import prompt_templates
        from sera.agent.prompt_templates import TEMPLATE_REGISTRY

        name_to_const = {
            "query_generation": "QUERY_GENERATION_PROMPT",
            "paper_clustering": "PAPER_CLUSTERING_PROMPT",
            "relevance_scoring": "RELEVANCE_SCORING_PROMPT",
            "spec_generation": "SPEC_GENERATION_PROMPT",
            "draft": "DRAFT_PROMPT",
            "debug": "DEBUG_PROMPT",
            "improve": "IMPROVE_PROMPT",
            "experiment_code": "EXPERIMENT_CODE_PROMPT",
            "paper_outline": "PAPER_OUTLINE_PROMPT",
            "paper_full_generation": "PAPER_FULL_GENERATION_PROMPT",
            "paper_writeup_reflection": "PAPER_WRITEUP_REFLECTION_PROMPT",
            "citation_search": "CITATION_SEARCH_PROMPT",
            "citation_select": "CITATION_SELECT_PROMPT",
            "plot_aggregation": "PLOT_AGGREGATION_PROMPT",
            "vlm_figure_description": "VLM_FIGURE_DESCRIPTION_PROMPT",
            "vlm_figure_caption_review": "VLM_FIGURE_CAPTION_REVIEW_PROMPT",
            "vlm_duplicate_detection": "VLM_DUPLICATE_DETECTION_PROMPT",
            "paper_evaluation": "PAPER_EVALUATION_PROMPT",
            "reviewer_reflection": "REVIEWER_REFLECTION_PROMPT",
            "meta_review": "META_REVIEW_PROMPT",
            "paper_revision": "PAPER_REVISION_PROMPT",
        }
        for reg_name, const_name in name_to_const.items():
            const_val = getattr(prompt_templates, const_name)
            assert const_val is TEMPLATE_REGISTRY[reg_name], (
                f"{const_name} does not match TEMPLATE_REGISTRY['{reg_name}']"
            )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    """Verify file-level caching works."""

    def test_cache_populated(self):
        _cache.clear()
        get_search_prompt("draft")
        assert "search.yaml" in _cache

    def test_same_object_returned(self):
        a = get_search_prompt("draft")
        b = get_search_prompt("draft")
        assert a is b
