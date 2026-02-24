"""
Prompt templates for all SERA phases.

Each template is a string constant with {placeholder} variables that get
filled in at call time via str.format() or str.format_map().
"""

# =============================================================================
# Phase 0: Literature Review & Related Work
# =============================================================================

QUERY_GENERATION_PROMPT = """\
You are a research assistant helping to survey the literature for a new research project.

## Task Description
{task_description}

## Research Domain
Field: {field}
Subfield: {subfield}

## Goal
{goal_objective}

## Instructions
Generate 3-5 diverse search queries that would help find the most relevant papers for this \
research task. The queries should:
1. Cover different aspects of the problem (methods, datasets, evaluation metrics, related tasks).
2. Use varied terminology to maximize coverage (synonyms, alternative phrasings).
3. Include at least one broad query and one highly specific query.
4. Target key recent advances as well as foundational work.

Return your queries as a JSON array of strings. Example:
["query 1", "query 2", "query 3"]

Output ONLY the JSON array, no other text.
"""

PAPER_CLUSTERING_PROMPT = """\
You are a research librarian organizing a literature survey.

## Task Context
{task_description}

## Papers to Cluster
{papers_json}

## Instructions
Group the papers above into thematic clusters based on their methodological approach, \
problem formulation, or contribution type. For each cluster, provide:
- A concise name (2-5 words)
- A one-sentence description of what unifies the cluster
- The list of paper IDs belonging to the cluster
- 3-5 representative keywords

Each paper should appear in exactly one cluster. Aim for 3-7 clusters.

Return your result as JSON:
{{
  "clusters": [
    {{
      "name": "...",
      "description": "...",
      "paper_ids": ["id1", "id2"],
      "keywords": ["kw1", "kw2", "kw3"]
    }}
  ]
}}

Output ONLY the JSON, no other text.
"""

RELEVANCE_SCORING_PROMPT = """\
You are an expert reviewer assessing paper relevance to a research task.

## Research Task
{task_description}

## Goal
{goal_objective}

## Paper Under Review
Title: {paper_title}
Authors: {paper_authors}
Year: {paper_year}
Abstract: {paper_abstract}

## Instructions
Score the relevance of this paper to the research task on a scale from 0.0 to 1.0:
- 0.0: Completely irrelevant
- 0.25: Tangentially related (shares domain but not methods or problem)
- 0.5: Moderately relevant (related methods or problem, but different focus)
- 0.75: Highly relevant (directly addresses similar problems or uses applicable methods)
- 1.0: Perfectly relevant (addresses the exact problem or proposes a directly usable method)

Also provide a brief (1-2 sentence) justification.

Return as JSON:
{{"score": <float>, "justification": "<string>"}}

Output ONLY the JSON, no other text.
"""

# =============================================================================
# Phase 1: Spec Generation
# =============================================================================

SPEC_GENERATION_PROMPT = """\
You are SERA, a Self-Evolving Research Agent. Your task is to generate a structured \
research plan based on the user's input and the surveyed literature.

## User Input (Input-1)
Task: {task_brief}
Domain: {field} / {subfield}
Data: {data_description} ({data_format}, {data_size})
Goal: {goal_objective} (direction: {goal_direction})
Baseline: {baseline}
Constraints: {constraints_json}
Notes: {notes}

## Related Work Summary
{related_work_summary}

## Baseline Candidates
{baseline_candidates_json}

## Common Metrics
{common_metrics_json}

## Open Problems
{open_problems_json}

## Instructions
Generate a research plan as a JSON object with the following structure:

{{
  "problem_spec": {{
    "title": "<concise research title>",
    "hypothesis": "<clear, testable hypothesis>",
    "independent_variables": ["<var1>", "<var2>"],
    "dependent_variables": ["<metric1>", "<metric2>"],
    "controls": ["<control1>"],
    "methodology_summary": "<2-3 sentence methodology overview>"
  }},
  "plan_spec": {{
    "approach_family": "<e.g., ensemble, neural, optimization>",
    "steps": [
      {{"name": "<step_name>", "description": "<what this step does>", "operator": "<draft|improve|debug>"}}
    ],
    "expected_baselines": ["<baseline1>", "<baseline2>"],
    "evaluation_metrics": ["<metric1>", "<metric2>"],
    "resource_estimate": {{
      "gpu_hours": <float>,
      "max_concurrent_runs": <int>
    }}
  }}
}}

Ensure:
1. The hypothesis is falsifiable and specific.
2. Steps cover data preparation, model development, training, and evaluation.
3. At least 2 baselines are included.
4. Metrics align with the goal objective.

Output ONLY the JSON, no other text.
"""

# =============================================================================
# Phase 2: Idea Evolution Operators
# =============================================================================

DRAFT_PROMPT = """\
You are a creative research scientist brainstorming new approaches.

## Research Context
Title: {title}
Hypothesis: {hypothesis}
Task: {task_brief}
Goal: {goal_objective} ({goal_direction})
Domain: {field} / {subfield}

## Current Best Approach
Score: {best_score}
Description: {best_approach_description}

## Related Work Insights
{related_work_insights}

## Previous Approaches Tried
{previous_approaches_summary}

## Instructions
Propose a NEW research approach that is substantively different from all previous attempts. \
Your proposal should:
1. Clearly describe the method and its novelty.
2. Explain why it might outperform the current best.
3. Identify potential risks or failure modes.
4. Be implementable within the given resource constraints.

Return as JSON:
{{
  "approach_name": "<concise name>",
  "description": "<detailed description of the approach>",
  "novelty": "<what makes this different from prior work>",
  "expected_advantage": "<why this might work better>",
  "risks": ["<risk1>", "<risk2>"],
  "implementation_notes": "<key implementation details>"
}}

Output ONLY the JSON, no other text.
"""

DEBUG_PROMPT = """\
You are an expert debugging assistant analyzing a failed experiment.

## Experiment Context
Approach: {approach_name}
Description: {approach_description}

## Error Information
Error type: {error_type}
Error message: {error_message}
Traceback:
```
{traceback}
```

## Code That Failed
```{code_block_tag}
{failed_code}
```

## Execution Log (last 50 lines)
```
{execution_log}
```

## Instructions
Diagnose the root cause of this failure and provide a fix. Your response should:
1. Identify the root cause (not just the symptom).
2. Explain why the error occurred.
3. Provide the corrected code.
4. Suggest any defensive measures to prevent similar issues.

Return as JSON:
{{
  "root_cause": "<diagnosis of what went wrong>",
  "explanation": "<why this happened>",
  "fix_description": "<what the fix does>",
  "corrected_code": "<the full corrected Python code>",
  "preventive_measures": ["<measure1>", "<measure2>"]
}}

Output ONLY the JSON, no other text.
"""

IMPROVE_PROMPT = """\
You are an expert ML researcher tasked with making an atomic improvement to an existing approach.

## Research Context
Title: {title}
Task: {task_brief}
Goal: {goal_objective} ({goal_direction})
Primary Metric: {primary_metric}

## Current Approach
Name: {approach_name}
Description: {approach_description}
Current Score ({primary_metric}): {current_score}

## Current Code
```python
{current_code}
```

## Experiment Log (recent results)
{experiment_log}

## Lineage Context
Parent node: {parent_node_id}
Generation: {generation}
Improvements already tried on this branch: {tried_improvements}

## Related Work Hints
{related_work_hints}

## Instructions
Propose exactly ONE atomic improvement to the current approach. An atomic improvement \
is a single, well-defined change that can be independently evaluated. Examples:
- Change learning rate schedule from cosine to warm-restart
- Add label smoothing with epsilon=0.1
- Replace ReLU activations with GELU
- Add gradient clipping at max_norm=1.0
- Increase model depth by 1 layer

Do NOT propose multiple changes bundled together. The improvement must be:
1. Minimal: change as little as possible.
2. Testable: the effect can be measured by re-running the experiment.
3. Justified: backed by reasoning or literature evidence.

Return as JSON:
{{
  "improvement_name": "<concise name for this change>",
  "category": "<hyperparameter|architecture|regularization|data_augmentation|optimization|other>",
  "description": "<what exactly changes and why>",
  "justification": "<reasoning or evidence supporting this change>",
  "expected_effect": "<predicted impact on {primary_metric}>",
  "modified_code": "<the full modified Python code with the improvement applied>",
  "rollback_description": "<how to undo this change if it hurts performance>"
}}

Output ONLY the JSON, no other text.
"""

# =============================================================================
# Phase 3: Experiment Code Generation
# =============================================================================

EXPERIMENT_CODE_PROMPT = """\
You are a meticulous ML engineer writing experiment code for a research project.

## Research Context
Title: {title}
Task: {task_brief}
Goal: {goal_objective} ({goal_direction})

## Approach to Implement
Name: {approach_name}
Description: {approach_description}

## Data Specification
Location: {data_location}
Format: {data_format}
Description: {data_description}

## Evaluation Metrics
{metrics_json}

## Constraints
{constraints_json}

## Reproducibility Requirements
- Random seed: {seed}
- Must log all hyperparameters
- Must save model checkpoints
- Must output results as JSON to stdout

## Available Packages
{available_packages}

## Instructions
Write a complete, self-contained Python script that:
1. Loads and preprocesses the data from {data_location}.
2. Implements the described approach.
3. Trains the model (if applicable) with proper logging.
4. Evaluates on a held-out test set using the specified metrics.
5. Prints a JSON results object to stdout with the format:
   {{"metrics": {{"metric_name": value, ...}}, "metadata": {{"seed": ..., "epochs": ..., ...}}}}

The script must:
- Be runnable with `python experiment.py` (no additional arguments required).
- Handle errors gracefully with informative messages.
- Set random seeds for reproducibility.
- Not exceed {max_runtime_seconds} seconds of runtime.

Output ONLY the Python code, no markdown fences or other text.
"""

# =============================================================================
# Phase 7: Paper Writing
# =============================================================================

PAPER_OUTLINE_PROMPT = """\
You are an experienced academic writer creating a paper outline.

## Research Summary
Title: {title}
Hypothesis: {hypothesis}
Methodology: {methodology_summary}

## Experiment Results
{results_summary}

## Best Approach
{best_approach_description}

## Related Work Clusters
{clusters_summary}

## Paper Format
Format: {paper_format}
Max pages: {max_pages}
Required sections: {required_sections_json}

## Teacher Paper Structure
{teacher_structure_summary}

## Instructions
Create a detailed outline for the research paper. For each section, provide:
1. The section heading.
2. A 2-3 sentence summary of what the section will cover.
3. Key points to include (as bullet points).
4. Approximate word count target.
5. Which figures/tables belong in this section.

Return as JSON:
{{
  "outline": [
    {{
      "section": "<heading>",
      "summary": "<what this section covers>",
      "key_points": ["<point1>", "<point2>"],
      "target_words": <int>,
      "figures": ["<fig_description>"]
    }}
  ]
}}

Output ONLY the JSON, no other text.
"""

PAPER_FULL_GENERATION_PROMPT = """\
You are an expert academic writer producing a complete research paper.

## Paper Outline
{outline_json}

## Research Details
Title: {title}
Authors: {authors}
Abstract draft: {abstract_draft}

## Methodology Details
{methodology_details}

## Experiment Results
{results_details}

## Related Work
{related_work_text}

## Figures and Tables
{figures_tables_json}

## Writing Style Guide
- Follow {paper_format} formatting conventions.
- Use {citation_style} citation style.
- Be concise but thorough.
- Every claim must be supported by evidence or citation.
- Use active voice where appropriate.
{teacher_style_notes}

## Instructions
Write the complete paper in LaTeX format, following the outline provided. Include:
1. All required sections with appropriate depth.
2. Proper citation commands (\\cite{{}}) where references are needed.
3. Figure and table references (\\ref{{}}) matching the provided figures.
4. Mathematical notation where appropriate.
5. A clear narrative flow from introduction through conclusion.

Output ONLY the LaTeX content (no preamble/documentclass -- just the body sections).
"""

PAPER_WRITEUP_REFLECTION_PROMPT = """\
You are a critical academic reviewer providing feedback on a draft paper.

## Paper Draft
{paper_draft}

## Research Context
Title: {title}
Goal: {goal_objective}
Key results: {key_results}

## Evaluation Criteria
1. Clarity: Is the writing clear and unambiguous?
2. Completeness: Are all claims supported? Are there gaps?
3. Flow: Does the narrative progress logically?
4. Technical accuracy: Are methods and results described correctly?
5. Conciseness: Is there unnecessary repetition or verbosity?

## Instructions
Provide a detailed self-reflection on the draft. For each section, identify:
- Strengths (what works well)
- Weaknesses (what needs improvement)
- Specific suggestions for revision

Return as JSON:
{{
  "overall_assessment": "<1-2 sentence summary>",
  "section_feedback": [
    {{
      "section": "<section name>",
      "strengths": ["<strength1>"],
      "weaknesses": ["<weakness1>"],
      "suggestions": ["<suggestion1>"]
    }}
  ],
  "missing_elements": ["<missing1>"],
  "priority_revisions": ["<revision1>", "<revision2>"]
}}

Output ONLY the JSON, no other text.
"""

CITATION_SEARCH_PROMPT = """\
You are a research assistant identifying missing citations in a paper draft.

## Paper Draft
{paper_draft}

## Currently Cited Papers
{current_citations_json}

## Instructions
Review the paper draft and identify claims or statements that need citations but \
currently lack them. For each missing citation, provide:
1. The exact sentence or phrase that needs a citation.
2. What kind of paper should be cited (method, dataset, finding, etc.).
3. A search query to find the appropriate paper.

Return as JSON:
{{
  "missing_citations": [
    {{
      "text": "<the sentence needing a citation>",
      "citation_type": "<method|dataset|finding|benchmark|tool|theory>",
      "search_query": "<query to find the right paper>",
      "importance": "<critical|important|nice_to_have>"
    }}
  ]
}}

Output ONLY the JSON, no other text.
"""

CITATION_SELECT_PROMPT = """\
You are a research assistant selecting the most relevant citation from search results.

## Context
The following sentence needs a citation:
"{text_needing_citation}"

Citation type needed: {citation_type}

## Search Results
{search_results_json}

## Instructions
From the search results above, select the single most appropriate paper to cite. \
Consider:
1. Relevance to the specific claim being made.
2. Publication venue quality.
3. Recency (prefer recent work unless citing foundational results).
4. Citation count (as a proxy for community acceptance).

Return as JSON:
{{
  "selected_paper_id": "<paper_id>",
  "justification": "<why this paper is the best citation>"
}}

Output ONLY the JSON, no other text.
"""

# =============================================================================
# Phase 7: Figures and Visualization
# =============================================================================

PLOT_AGGREGATION_PROMPT = """\
You are a data visualization expert creating publication-quality figures.

## Experiment Results
{results_json}

## Approaches Compared
{approaches_json}

## Metrics to Plot
{metrics_to_plot}

## Paper Format
{paper_format} style, {figure_width_inches} inches wide.

## Instructions
Write a Python script using matplotlib and seaborn that generates an aggregation plot \
comparing the experimental results across all approaches. The plot should:
1. Compare all approaches on the specified metrics.
2. Include error bars (confidence intervals from multiple runs).
3. Use colorblind-friendly colors.
4. Have clear, readable labels and a legend.
5. Follow {paper_format} figure conventions (font sizes, etc.).
6. Save the figure as both PDF and PNG.

The script should:
- Read results from a JSON file at `{results_path}`.
- Save the figure to `{output_path}`.
- Print the output file path to stdout.

Output ONLY the Python code, no markdown fences or other text.
"""

VLM_FIGURE_DESCRIPTION_PROMPT = """\
You are a vision-language model analyzing a figure from a research paper.

## Figure Context
Figure file: {figure_path}
Section: {section_name}
Paper title: {paper_title}

## Instructions
Describe this figure in detail, covering:
1. What type of visualization it is (bar chart, line plot, table, diagram, etc.).
2. What data or relationships it shows.
3. The key takeaway or finding illustrated.
4. The axes labels, legends, and any annotations.
5. Whether the figure is clear and well-formatted for publication.

Return as JSON:
{{
  "figure_type": "<type>",
  "description": "<detailed description>",
  "key_finding": "<main takeaway>",
  "axes": {{"x": "<x-axis label>", "y": "<y-axis label>"}},
  "quality_assessment": "<assessment of visual quality>",
  "suggestions": ["<improvement suggestion>"]
}}

Output ONLY the JSON, no other text.
"""

VLM_FIGURE_CAPTION_REVIEW_PROMPT = """\
You are reviewing a figure and its caption for a research paper.

## Figure Description
{figure_description}

## Current Caption
{current_caption}

## Section Context
{section_context}

## Instructions
Evaluate whether the caption:
1. Accurately describes what the figure shows.
2. Is self-contained (reader can understand the figure from caption alone).
3. Highlights the key finding or takeaway.
4. References all relevant visual elements (axes, colors, symbols).
5. Uses proper cross-references to other figures/tables/sections.

Return as JSON:
{{
  "caption_quality": "<good|needs_revision>",
  "accuracy": "<does the caption match the figure?>",
  "completeness": "<are all elements described?>",
  "suggested_caption": "<improved caption if needed, or null>",
  "missing_references": ["<missing cross-ref>"]
}}

Output ONLY the JSON, no other text.
"""

VLM_DUPLICATE_DETECTION_PROMPT = """\
You are analyzing a set of figures from a research paper to detect duplicates or \
near-duplicates.

## Figures
{figures_json}

## Instructions
Examine the provided figure descriptions and identify any pairs that are:
1. Exact duplicates (same data, same visualization).
2. Near-duplicates (same data, slightly different presentation).
3. Overlapping (partially redundant information).

For each detected pair, recommend whether to merge, remove one, or keep both with \
differentiation.

Return as JSON:
{{
  "duplicate_pairs": [
    {{
      "figure_a": "<figure_id_a>",
      "figure_b": "<figure_id_b>",
      "overlap_type": "<exact|near_duplicate|partial_overlap>",
      "recommendation": "<merge|remove_one|differentiate>",
      "justification": "<why>"
    }}
  ]
}}

Output ONLY the JSON, no other text.
"""

# =============================================================================
# Phase 8: Paper Evaluation
# =============================================================================

PAPER_EVALUATION_PROMPT = """\
You are an expert reviewer for a top-tier {venue} conference. You have been asked to \
evaluate the following research paper.

## Paper to Review
{paper_text}

## Evaluation Criteria
{criteria_json}

## Few-Shot Examples
{few_shot_reviews}

## Reviewer Instructions
Evaluate this paper as if reviewing for {venue}. For each criterion, provide:
1. A score from 1 to {max_score}.
2. A detailed justification (2-4 sentences) referencing specific parts of the paper.

Also provide:
- An overall score (1 to {max_score}).
- A summary of strengths (3-5 bullet points).
- A summary of weaknesses (3-5 bullet points).
- Questions for the authors (1-3 questions).
- A confidence score (1-5, where 5 = very confident).

Return as JSON:
{{
  "criterion_scores": [
    {{
      "criterion": "<name>",
      "score": <int>,
      "justification": "<detailed justification>"
    }}
  ],
  "overall_score": <int>,
  "strengths": ["<strength1>", "<strength2>"],
  "weaknesses": ["<weakness1>", "<weakness2>"],
  "questions_for_authors": ["<question1>"],
  "confidence": <int>,
  "detailed_comments": "<free-form detailed review comments>"
}}

Output ONLY the JSON, no other text.
"""

REVIEWER_REFLECTION_PROMPT = """\
You are reflecting on your own review of a research paper to improve its quality and \
fairness.

## Your Previous Review
{previous_review_json}

## Paper Under Review
{paper_text}

## Reflection Round
This is reflection round {reflection_round} of {max_reflections}.

## Instructions
Critically examine your previous review for:
1. Bias: Are you being unfairly harsh or lenient? Are there implicit biases?
2. Consistency: Do your scores align with your justifications?
3. Completeness: Did you miss any important strengths or weaknesses?
4. Constructiveness: Are your criticisms actionable?
5. Accuracy: Did you misunderstand any part of the paper?

If you find issues with your previous review, provide a revised review. If your review \
was fair and accurate, you may keep the scores but must still reflect.

Return as JSON:
{{
  "reflection_notes": "<what you reconsidered>",
  "changes_made": ["<change1>", "<change2>"],
  "revised_criterion_scores": [
    {{
      "criterion": "<name>",
      "score": <int>,
      "justification": "<revised justification>"
    }}
  ],
  "revised_overall_score": <int>,
  "revised_strengths": ["<strength1>"],
  "revised_weaknesses": ["<weakness1>"],
  "revised_confidence": <int>
}}

Output ONLY the JSON, no other text.
"""

META_REVIEW_PROMPT = """\
You are an Area Chair synthesizing multiple reviews into a meta-review.

## Paper Title
{paper_title}

## Individual Reviews
{reviews_json}

## Evaluation Criteria
{criteria_json}

## Instructions
As Area Chair, synthesize the individual reviews into a meta-review. You should:
1. Identify consensus points (where reviewers agree).
2. Identify disagreements and adjudicate them.
3. Weigh reviewer confidence in your assessment.
4. Provide a final recommendation.

Return as JSON:
{{
  "consensus_strengths": ["<strength where reviewers agree>"],
  "consensus_weaknesses": ["<weakness where reviewers agree>"],
  "disagreements": [
    {{
      "topic": "<what reviewers disagree on>",
      "positions": ["<reviewer 1 view>", "<reviewer 2 view>"],
      "adjudication": "<your resolution>"
    }}
  ],
  "meta_score": <float>,
  "recommendation": "<accept|revise|reject>",
  "key_revision_requirements": ["<requirement1>", "<requirement2>"],
  "meta_review_text": "<narrative meta-review summary>"
}}

Output ONLY the JSON, no other text.
"""

PAPER_REVISION_PROMPT = """\
You are a researcher revising your paper based on reviewer feedback.

## Current Paper
{paper_text}

## Meta-Review
{meta_review_json}

## Individual Reviews
{reviews_json}

## Revision Round
This is revision round {revision_round} of {max_revisions}.

## Instructions
Revise the paper to address the reviewers' concerns. For each criticism:
1. Decide whether to accept, partially accept, or rebut.
2. If accepting, make the specific change in the paper text.
3. Prepare a response explaining what you changed and why.

Focus on:
- Addressing all "key_revision_requirements" from the meta-review.
- Fixing any factual errors or unclear passages identified by reviewers.
- Strengthening weak sections while preserving strong ones.
- NOT introducing new experiments or claims beyond what the data supports.

Return as JSON:
{{
  "revision_log": [
    {{
      "criticism": "<the reviewer criticism>",
      "response_type": "<accept|partial_accept|rebut>",
      "response": "<what you did and why>",
      "sections_modified": ["<section1>"]
    }}
  ],
  "revised_paper": "<the full revised LaTeX paper body>",
  "author_response": "<formal response letter to reviewers>"
}}

Output ONLY the JSON, no other text.
"""

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
