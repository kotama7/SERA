"""Phase 1: LLM-driven Spec generation."""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SpecBuilder:
    """Build ProblemSpec and PlanSpec from Input-1 and Phase 0 outputs using LLM."""

    def __init__(self, agent_llm: Any):
        self.agent_llm = agent_llm

    async def build_problem_spec(self, input1: Any, related_work: Any) -> dict:
        """Generate ProblemSpec JSON using LLM, with up to 3 retries on validation failure."""
        from sera.agent.prompt_templates import SPEC_GENERATION_PROMPT

        context = self._build_context(input1, related_work)
        prompt = SPEC_GENERATION_PROMPT.format(
            context=context,
            spec_type="ProblemSpec",
            schema_description=self._problem_spec_schema(),
        )

        for attempt in range(3):
            temperature = 0.7 + attempt * 0.1
            response = await self.agent_llm.generate(
                prompt=prompt,
                purpose="spec_generation_problem",
                temperature=temperature,
            )
            parsed = self._parse_json(response)
            if parsed is not None:
                try:
                    from sera.specs.problem_spec import ProblemSpecModel
                    spec = ProblemSpecModel(**parsed.get("problem_spec", parsed))
                    return spec.model_dump()
                except Exception as e:
                    logger.warning(f"ProblemSpec validation failed (attempt {attempt+1}): {e}")
                    prompt += f"\n\nPrevious attempt failed validation: {e}\nPlease fix and regenerate."
            else:
                logger.warning(f"JSON parse failed (attempt {attempt+1})")

        # Fallback: return defaults
        logger.warning("All ProblemSpec generation attempts failed, using defaults")
        from sera.specs.problem_spec import ProblemSpecModel
        defaults = ProblemSpecModel(
            title=getattr(input1, "task", input1.get("task", {})).get("brief", "Research") if isinstance(input1, dict) else getattr(getattr(input1, "task", None), "brief", "Research"),
        )
        return defaults.model_dump()

    async def build_plan_spec(self, input1: Any, problem_spec: Any) -> dict:
        """Generate PlanSpec JSON using LLM."""
        from sera.agent.prompt_templates import SPEC_GENERATION_PROMPT

        prompt = SPEC_GENERATION_PROMPT.format(
            context=f"Input-1: {json.dumps(input1 if isinstance(input1, dict) else input1.model_dump(), default=str)}\n"
                    f"ProblemSpec: {json.dumps(problem_spec if isinstance(problem_spec, dict) else problem_spec.model_dump(), default=str)}",
            spec_type="PlanSpec",
            schema_description=self._plan_spec_schema(),
        )

        for attempt in range(3):
            temperature = 0.7 + attempt * 0.1
            response = await self.agent_llm.generate(
                prompt=prompt,
                purpose="spec_generation_plan",
                temperature=temperature,
            )
            parsed = self._parse_json(response)
            if parsed is not None:
                try:
                    from sera.specs.plan_spec import PlanSpecModel
                    spec = PlanSpecModel(**parsed.get("plan_spec", parsed))
                    return spec.model_dump()
                except Exception as e:
                    logger.warning(f"PlanSpec validation failed (attempt {attempt+1}): {e}")
            else:
                logger.warning(f"JSON parse failed (attempt {attempt+1})")

        from sera.specs.plan_spec import PlanSpecModel
        return PlanSpecModel().model_dump()

    def _build_context(self, input1: Any, related_work: Any) -> str:
        """Build context string from Input-1 and RelatedWorkSpec."""
        i1 = input1 if isinstance(input1, dict) else input1.model_dump()
        rw = related_work if isinstance(related_work, dict) else related_work.model_dump()
        return f"Input-1:\n{json.dumps(i1, default=str, indent=2)}\n\nRelatedWork:\n{json.dumps(rw, default=str, indent=2)}"

    def _parse_json(self, response: str) -> dict | None:
        """Extract JSON from LLM response."""
        import re
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def _problem_spec_schema(self) -> str:
        return """ProblemSpec requires:
- title: str (research title)
- objective: {description: str, metric_name: str, direction: "maximize"|"minimize"}
- constraints: [{name: str, type: "bool"|"ge"|"le", threshold: float|bool, epsilon: float}]
- secondary_metrics: [{name: str, direction: "maximize"|"minimize", weight_in_tiebreak: float}]
- manipulated_variables: [{name: str, type: "float"|"int"|"categorical", range: [min,max], scale: "linear"|"log", choices: [...]}]
- observed_variables: [{name: str, type: "float"}]
- evaluation_design: {type: "holdout"|"cross_validation", test_split: float}
- experiment_template: str"""

    def _plan_spec_schema(self) -> str:
        return """PlanSpec requires:
- search_strategy: {name: str, description: str}
- branching: {generator: "llm", operators: [{name: "draft"|"debug"|"improve", description: str}]}
- reward: {formula: str, primary_source: str, constraint_penalty: float, cost_source: str}
- logging: {log_every_node: bool, log_llm_prompts: bool, checkpoint_interval: int}
- artifacts: {save_all_experiments: bool, save_pruned: bool, export_format: "json"|"yaml"}"""
