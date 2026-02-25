"""
Prompt templates for all SERA phases.

Each template is a string constant with {placeholder} variables that get
filled in at call time via str.format() or str.format_map().

Templates are loaded from YAML files in the sera.prompts package.
"""

from sera.prompts import get_template

# =============================================================================
# Phase 0: Literature Review & Related Work
# =============================================================================

QUERY_GENERATION_PROMPT = get_template("query_generation")
PAPER_CLUSTERING_PROMPT = get_template("paper_clustering")
RELEVANCE_SCORING_PROMPT = get_template("relevance_scoring")

# =============================================================================
# Phase 1: Spec Generation
# =============================================================================

SPEC_GENERATION_PROMPT = get_template("spec_generation")

# =============================================================================
# Phase 2: Idea Evolution Operators
# =============================================================================

DRAFT_PROMPT = get_template("draft")
DEBUG_PROMPT = get_template("debug")
IMPROVE_PROMPT = get_template("improve")

# =============================================================================
# Phase 3: Experiment Code Generation
# =============================================================================

EXPERIMENT_CODE_PROMPT = get_template("experiment_code")

# =============================================================================
# Phase 7: Paper Writing
# =============================================================================

PAPER_OUTLINE_PROMPT = get_template("paper_outline")
PAPER_FULL_GENERATION_PROMPT = get_template("paper_full_generation")
PAPER_WRITEUP_REFLECTION_PROMPT = get_template("paper_writeup_reflection")
CITATION_SEARCH_PROMPT = get_template("citation_search")
CITATION_SELECT_PROMPT = get_template("citation_select")

# =============================================================================
# Phase 7: Figures and Visualization
# =============================================================================

PLOT_AGGREGATION_PROMPT = get_template("plot_aggregation")
VLM_FIGURE_DESCRIPTION_PROMPT = get_template("vlm_figure_description")
VLM_FIGURE_CAPTION_REVIEW_PROMPT = get_template("vlm_figure_caption_review")
VLM_DUPLICATE_DETECTION_PROMPT = get_template("vlm_duplicate_detection")

# =============================================================================
# Phase 8: Paper Evaluation
# =============================================================================

PAPER_EVALUATION_PROMPT = get_template("paper_evaluation")
REVIEWER_REFLECTION_PROMPT = get_template("reviewer_reflection")
META_REVIEW_PROMPT = get_template("meta_review")
PAPER_REVISION_PROMPT = get_template("paper_revision")

# =============================================================================
# Template Registry -- convenience mapping for programmatic access
# =============================================================================

TEMPLATE_REGISTRY: dict[str, str] = {
    "query_generation": QUERY_GENERATION_PROMPT,
    "paper_clustering": PAPER_CLUSTERING_PROMPT,
    "relevance_scoring": RELEVANCE_SCORING_PROMPT,
    "spec_generation": SPEC_GENERATION_PROMPT,
    "draft": DRAFT_PROMPT,
    "debug": DEBUG_PROMPT,
    "improve": IMPROVE_PROMPT,
    "experiment_code": EXPERIMENT_CODE_PROMPT,
    "paper_outline": PAPER_OUTLINE_PROMPT,
    "paper_full_generation": PAPER_FULL_GENERATION_PROMPT,
    "paper_writeup_reflection": PAPER_WRITEUP_REFLECTION_PROMPT,
    "citation_search": CITATION_SEARCH_PROMPT,
    "citation_select": CITATION_SELECT_PROMPT,
    "plot_aggregation": PLOT_AGGREGATION_PROMPT,
    "vlm_figure_description": VLM_FIGURE_DESCRIPTION_PROMPT,
    "vlm_figure_caption_review": VLM_FIGURE_CAPTION_REVIEW_PROMPT,
    "vlm_duplicate_detection": VLM_DUPLICATE_DETECTION_PROMPT,
    "paper_evaluation": PAPER_EVALUATION_PROMPT,
    "reviewer_reflection": REVIEWER_REFLECTION_PROMPT,
    "meta_review": META_REVIEW_PROMPT,
    "paper_revision": PAPER_REVISION_PROMPT,
}
