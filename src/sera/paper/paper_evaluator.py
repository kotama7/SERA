"""PaperEvaluator per S12.1 - ensemble LLM-as-judge paper evaluation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PaperScoreResult:
    """Result of paper evaluation by the ensemble of LLM reviewers."""

    scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    confidence: float = 0.0
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    missing_items: list[str] = field(default_factory=list)
    improvement_instructions: list[str] = field(default_factory=list)
    decision: str = ""  # "accept" | "revise" | "reject"
    passed: bool = False
    individual_reviews: list[dict] = field(default_factory=list)
    meta_review: str = ""


class PaperEvaluator:
    """Ensemble-based paper evaluation using LLM-as-judge.

    Generates multiple independent reviews, each with configurable bias
    mode and reflection rounds, then aggregates scores and produces a
    meta-review.
    """

    def __init__(self) -> None:
        pass

    async def evaluate(
        self,
        paper_md: str,
        paper_score_spec: Any,
        agent_llm: Any,
    ) -> PaperScoreResult:
        """Evaluate a paper using ensemble of LLM reviewers.

        Parameters
        ----------
        paper_md:
            The paper content in Markdown.
        paper_score_spec:
            A PaperScoreSpecModel (or duck-typed equivalent) defining
            criteria, rubric, ensemble config, etc.
        agent_llm:
            An AgentLLM instance for generating reviews.

        Returns
        -------
        PaperScoreResult with aggregated scores and reviews.
        """
        ensemble_cfg = getattr(paper_score_spec, "ensemble", None)
        num_reviews = getattr(ensemble_cfg, "num_reviews_ensemble", 3) if ensemble_cfg else 3
        num_reflections = getattr(ensemble_cfg, "num_reviewer_reflections", 2) if ensemble_cfg else 2
        num_fs = getattr(ensemble_cfg, "num_fs_examples", 2) if ensemble_cfg else 2
        bias_mode = getattr(ensemble_cfg, "bias_mode", "critical") if ensemble_cfg else "critical"
        do_meta_review = getattr(ensemble_cfg, "meta_review", True) if ensemble_cfg else True
        temperature = getattr(ensemble_cfg, "temperature", 0.75) if ensemble_cfg else 0.75

        criteria = getattr(paper_score_spec, "criteria", [])
        max_score = getattr(paper_score_spec, "max_score", 10)
        passing_score = getattr(paper_score_spec, "passing_score", 6.0)
        few_shot_reviews = getattr(paper_score_spec, "few_shot_reviews", [])

        # -- Step 1: Generate independent reviews ----------------------
        individual_reviews: list[dict] = []

        for reviewer_idx in range(num_reviews):
            # Alternate bias mode for diversity
            if bias_mode == "critical":
                reviewer_bias = "critical" if reviewer_idx % 2 == 0 else "generous"
            elif bias_mode == "generous":
                reviewer_bias = "generous" if reviewer_idx % 2 == 0 else "critical"
            else:
                reviewer_bias = bias_mode

            review = await self._generate_review(
                paper_md=paper_md,
                criteria=criteria,
                max_score=max_score,
                bias_mode=reviewer_bias,
                num_reflections=num_reflections,
                few_shot_reviews=few_shot_reviews[:num_fs],
                agent_llm=agent_llm,
                reviewer_idx=reviewer_idx,
                temperature=temperature,
            )
            individual_reviews.append(review)

        # -- Step 2: Ensemble aggregation ------------------------------
        result = self._aggregate_reviews(individual_reviews, criteria, max_score, passing_score)
        result.individual_reviews = individual_reviews

        # Meta-review
        if do_meta_review and len(individual_reviews) > 1:
            result.meta_review = await self._generate_meta_review(paper_md, individual_reviews, agent_llm)

        return result

    # ------------------------------------------------------------------
    # Single review generation
    # ------------------------------------------------------------------

    async def _generate_review(
        self,
        paper_md: str,
        criteria: list[Any],
        max_score: int,
        bias_mode: str,
        num_reflections: int,
        few_shot_reviews: list[Any],
        agent_llm: Any,
        reviewer_idx: int,
        temperature: float,
    ) -> dict:
        """Generate a single review with reflection loop."""

        # Build system prompt
        system_parts = [
            f"You are Reviewer #{reviewer_idx + 1} for a scientific paper.",
            f"Your reviewing style is {bias_mode}.",
            "Evaluate the paper carefully and provide structured feedback.",
        ]
        system_prompt = " ".join(system_parts)

        # Build rubric text
        rubric_text = ""
        for criterion in criteria:
            name = getattr(criterion, "name", str(criterion))
            desc = getattr(criterion, "description", "")
            rubric = getattr(criterion, "rubric", {})
            weight = getattr(criterion, "weight", 1.0)
            rubric_text += f"\n- {name} (weight={weight}): {desc}"
            if rubric:
                for score_val, score_desc in sorted(rubric.items()):
                    rubric_text += f"\n  {score_val}: {score_desc}"

        # Few-shot examples
        fs_text = ""
        if few_shot_reviews:
            fs_text = "\n\nHere are example reviews for reference:\n"
            for i, example in enumerate(few_shot_reviews):
                fs_text += f"\nExample {i + 1}:\n{json.dumps(example, default=str)}\n"

        # Full evaluation prompt
        eval_prompt = (
            f"{system_prompt}\n\n"
            f"Evaluation criteria and rubric (max score per criterion: {max_score}):\n"
            f"{rubric_text}\n"
            f"{fs_text}\n\n"
            f"Paper to review:\n{paper_md[:8000]}\n\n"
            "Provide your review in the following format:\n"
            "SUMMARY: <1-2 sentence summary>\n"
            "STRENGTHS:\n- <strength 1>\n- <strength 2>\n"
            "WEAKNESSES:\n- <weakness 1>\n- <weakness 2>\n"
            "QUESTIONS:\n- <question 1>\n"
            "LIMITATIONS:\n- <limitation 1>\n"
            "MISSING:\n- <missing item 1>\n"
            "IMPROVEMENTS:\n- <improvement 1>\n"
            "SCORES:\n"
        )
        for criterion in criteria:
            name = getattr(criterion, "name", str(criterion))
            eval_prompt += f"- {name}: <score 1-{max_score}>\n"
        eval_prompt += f"OVERALL: <score 1-{max_score}>\n"
        eval_prompt += "CONFIDENCE: <0.0-1.0>\n"
        eval_prompt += "DECISION: <accept|revise|reject>"

        review_text = await agent_llm.generate(
            prompt=eval_prompt,
            purpose=f"paper_review_{reviewer_idx}",
            temperature=temperature,
        )

        # -- Reflection loop -------------------------------------------
        for ref_round in range(num_reflections):
            reflection_prompt = (
                "Reflect on your review below. Consider:\n"
                "1. Are your scores justified by the evidence?\n"
                "2. Have you missed any important strengths or weaknesses?\n"
                "3. Are your improvement suggestions actionable?\n\n"
                f"Your current review:\n{review_text}\n\n"
                "Provide your REVISED review in the same format. "
                "If no changes are needed, reproduce the review as-is."
            )
            review_text = await agent_llm.generate(
                prompt=reflection_prompt,
                purpose=f"paper_review_{reviewer_idx}_reflection_{ref_round}",
                temperature=temperature,
            )

        # Parse the review
        return self._parse_review(review_text, criteria, max_score)

    # ------------------------------------------------------------------
    # Review parsing
    # ------------------------------------------------------------------

    def _parse_review(self, review_text: str, criteria: list[Any], max_score: int) -> dict:
        """Parse a structured review into a dict."""
        review: dict[str, Any] = {
            "summary": "",
            "strengths": [],
            "weaknesses": [],
            "questions": [],
            "limitations": [],
            "missing": [],
            "improvements": [],
            "scores": {},
            "overall_score": 0.0,
            "confidence": 0.5,
            "decision": "revise",
            "raw": review_text,
        }

        lines = review_text.split("\n")
        current_section: str | None = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Detect section headers
            upper = stripped.upper()
            if upper.startswith("SUMMARY:"):
                review["summary"] = stripped[len("SUMMARY:") :].strip()
                current_section = None
            elif upper.startswith("STRENGTHS:"):
                current_section = "strengths"
                rest = stripped[len("STRENGTHS:") :].strip()
                if rest:
                    review["strengths"].append(rest)
            elif upper.startswith("WEAKNESSES:"):
                current_section = "weaknesses"
                rest = stripped[len("WEAKNESSES:") :].strip()
                if rest:
                    review["weaknesses"].append(rest)
            elif upper.startswith("QUESTIONS:"):
                current_section = "questions"
                rest = stripped[len("QUESTIONS:") :].strip()
                if rest:
                    review["questions"].append(rest)
            elif upper.startswith("LIMITATIONS:"):
                current_section = "limitations"
                rest = stripped[len("LIMITATIONS:") :].strip()
                if rest:
                    review["limitations"].append(rest)
            elif upper.startswith("MISSING:"):
                current_section = "missing"
                rest = stripped[len("MISSING:") :].strip()
                if rest:
                    review["missing"].append(rest)
            elif upper.startswith("IMPROVEMENTS:"):
                current_section = "improvements"
                rest = stripped[len("IMPROVEMENTS:") :].strip()
                if rest:
                    review["improvements"].append(rest)
            elif upper.startswith("SCORES:"):
                current_section = "scores"
            elif upper.startswith("OVERALL:"):
                try:
                    score_str = re.search(r"[\d.]+", stripped[len("OVERALL:") :])
                    if score_str:
                        review["overall_score"] = min(float(score_str.group()), max_score)
                except (ValueError, AttributeError):
                    pass
                current_section = None
            elif upper.startswith("CONFIDENCE:"):
                try:
                    conf_str = re.search(r"[\d.]+", stripped[len("CONFIDENCE:") :])
                    if conf_str:
                        review["confidence"] = min(float(conf_str.group()), 1.0)
                except (ValueError, AttributeError):
                    pass
                current_section = None
            elif upper.startswith("DECISION:"):
                decision = stripped[len("DECISION:") :].strip().lower()
                if decision in ("accept", "revise", "reject"):
                    review["decision"] = decision
                current_section = None
            elif current_section in ("strengths", "weaknesses", "questions", "limitations", "missing", "improvements"):
                # List item
                item = stripped.lstrip("- *").strip()
                if item:
                    review[current_section].append(item)
            elif current_section == "scores":
                # Parse criterion scores like "- Novelty: 7"
                match = re.match(r"[-*]?\s*(.+?):\s*([\d.]+)", stripped)
                if match:
                    crit_name = match.group(1).strip()
                    try:
                        crit_score = min(float(match.group(2)), max_score)
                        review["scores"][crit_name] = crit_score
                    except ValueError:
                        pass

        return review

    # ------------------------------------------------------------------
    # Ensemble aggregation
    # ------------------------------------------------------------------

    def _aggregate_reviews(
        self,
        reviews: list[dict],
        criteria: list[Any],
        max_score: int,
        passing_score: float,
    ) -> PaperScoreResult:
        """Aggregate multiple reviews into a single PaperScoreResult."""
        result = PaperScoreResult()

        if not reviews:
            return result

        # Aggregate scores per criterion
        all_criterion_names = set()
        for r in reviews:
            all_criterion_names.update(r.get("scores", {}).keys())

        for crit_name in all_criterion_names:
            scores = [r["scores"][crit_name] for r in reviews if crit_name in r.get("scores", {})]
            if scores:
                result.scores[crit_name] = sum(scores) / len(scores)

        # Overall score: average of individual overall scores
        overall_scores = [r["overall_score"] for r in reviews if r.get("overall_score", 0) > 0]
        if overall_scores:
            result.overall_score = sum(overall_scores) / len(overall_scores)
        elif result.scores:
            # Fallback: average of criterion scores
            result.overall_score = sum(result.scores.values()) / len(result.scores)

        # Confidence: average
        confidences = [r["confidence"] for r in reviews if r.get("confidence", 0) > 0]
        result.confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Merge text fields
        summaries = [r["summary"] for r in reviews if r.get("summary")]
        result.summary = " | ".join(summaries) if summaries else ""

        for key in ("strengths", "weaknesses", "questions", "limitations", "missing", "improvements"):
            merged: list[str] = []
            for r in reviews:
                items = r.get(key, [])
                for item in items:
                    if item and item not in merged:
                        merged.append(item)
            if key == "missing":
                result.missing_items = merged
            elif key == "improvements":
                result.improvement_instructions = merged
            else:
                setattr(result, key, merged)

        # Decision: majority vote
        decisions = [r["decision"] for r in reviews if r.get("decision")]
        if decisions:
            from collections import Counter

            decision_counts = Counter(decisions)
            result.decision = decision_counts.most_common(1)[0][0]
        else:
            result.decision = "revise"

        # Passed?
        result.passed = result.overall_score >= passing_score

        return result

    # ------------------------------------------------------------------
    # Meta-review
    # ------------------------------------------------------------------

    async def _generate_meta_review(
        self,
        paper_md: str,
        reviews: list[dict],
        agent_llm: Any,
    ) -> str:
        """Generate an Area-Chair-style meta-review."""
        reviews_text = ""
        for i, r in enumerate(reviews):
            reviews_text += f"\n--- Reviewer {i + 1} ---\n"
            reviews_text += f"Summary: {r.get('summary', '')}\n"
            reviews_text += f"Overall: {r.get('overall_score', '?')}\n"
            reviews_text += f"Decision: {r.get('decision', '?')}\n"
            reviews_text += f"Strengths: {', '.join(r.get('strengths', []))}\n"
            reviews_text += f"Weaknesses: {', '.join(r.get('weaknesses', []))}\n"

        meta_prompt = (
            "You are an Area Chair at a top ML conference. "
            "Synthesize the following reviews into a meta-review.\n\n"
            f"Paper (first 3000 chars):\n{paper_md[:3000]}\n\n"
            f"Reviews:\n{reviews_text}\n\n"
            "Provide a meta-review that:\n"
            "1. Summarizes the consensus and disagreements\n"
            "2. Highlights the most important strengths and weaknesses\n"
            "3. Makes a final recommendation (accept/revise/reject)\n"
            "4. Provides specific improvement instructions if applicable"
        )

        return await agent_llm.generate(prompt=meta_prompt, purpose="meta_review")
