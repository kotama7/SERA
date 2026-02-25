"""Phase 1: LLM-driven Spec generation."""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SpecBuilder:
    """Build all specs from Input-1 and Phase 0 outputs using LLM and CLI args."""

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

            # Prefer call_function path
            if hasattr(self.agent_llm, "call_function"):
                parsed = await self.agent_llm.call_function(
                    "spec_generation_problem",
                    prompt=prompt,
                    purpose="spec_generation_problem",
                    temperature=temperature,
                )
            else:
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

            # Prefer call_function path
            if hasattr(self.agent_llm, "call_function"):
                parsed = await self.agent_llm.call_function(
                    "spec_generation_plan",
                    prompt=prompt,
                    purpose="spec_generation_plan",
                    temperature=temperature,
                )
            else:
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

    def build_model_spec(self, cli_args: dict) -> dict:
        """Build ModelSpec from CLI arguments with model family auto-detection.

        Parameters
        ----------
        cli_args : dict
            CLI arguments including: base_model, dtype, agent_llm, rank, alpha.
        """
        from sera.specs.model_spec import (
            ModelSpecModel, BaseModelConfig, AgentLLMConfig, AdapterSpec,
            infer_model_family,
        )

        base_model_id = cli_args.get("base_model", "Qwen/Qwen2.5-Coder-7B-Instruct")
        family = infer_model_family(base_model_id)

        base_model_cfg = BaseModelConfig(
            id=base_model_id,
            dtype=cli_args.get("dtype", "bf16"),
            family=family,
        )

        agent_llm_str = cli_args.get("agent_llm", "local:same_as_base")
        if ":" in agent_llm_str:
            provider, model_id = agent_llm_str.split(":", 1)
        else:
            provider, model_id = "local", "same_as_base"

        agent_llm_cfg = AgentLLMConfig(provider=provider, model_id=model_id)

        # Use family-specific default target modules if available
        from sera.specs.model_spec import _DEFAULT_MODEL_FAMILIES
        default_targets = ["q_proj", "v_proj"]
        if family in _DEFAULT_MODEL_FAMILIES:
            default_targets = _DEFAULT_MODEL_FAMILIES[family].get(
                "default_target_modules", default_targets
            )

        adapter_cfg = AdapterSpec(
            rank=cli_args.get("rank", 16),
            alpha=cli_args.get("alpha", 32),
            target_modules=default_targets,
        )

        model_spec = ModelSpecModel(
            base_model=base_model_cfg,
            agent_llm=agent_llm_cfg,
            adapter_spec=adapter_cfg,
        )
        return model_spec.model_dump()

    def build_resource_spec(self, cli_args: dict) -> dict:
        """Build ResourceSpec from CLI arguments.

        Parameters
        ----------
        cli_args : dict
            CLI arguments including: executor, gpu_count, memory_gb,
            cpu_cores, gpu_type, gpu_required, timeout, no_web, work_dir.
        """
        from sera.specs.resource_spec import (
            ResourceSpecModel, ComputeConfig, SandboxConfig,
            StorageConfig, NetworkConfig,
        )

        compute_cfg = ComputeConfig(
            executor_type=cli_args.get("executor", "local"),
            gpu_count=cli_args.get("gpu_count", 1),
            memory_gb=cli_args.get("memory_gb", 32),
            cpu_cores=cli_args.get("cpu_cores", 8),
            gpu_type=cli_args.get("gpu_type", ""),
            gpu_required=cli_args.get("gpu_required", True),
        )
        sandbox_cfg = SandboxConfig(
            experiment_timeout_sec=cli_args.get("timeout", 3600),
        )
        storage_cfg = StorageConfig(
            work_dir=cli_args.get("work_dir", "./sera_workspace"),
        )
        network_cfg = NetworkConfig(
            allow_internet=not cli_args.get("no_web", False),
        )
        resource_spec = ResourceSpecModel(
            compute=compute_cfg,
            sandbox=sandbox_cfg,
            storage=storage_cfg,
            network=network_cfg,
        )
        return resource_spec.model_dump()

    def build_execution_spec(self, cli_args: dict) -> dict:
        """Build ExecutionSpec from CLI arguments.

        Parameters
        ----------
        cli_args : dict
            CLI arguments including: max_nodes, max_depth, branch_factor,
            lambda_cost, beta, repeats, lcb_coef, no_sequential, seq_topk,
            lr, clip, ppo_steps.
        """
        from sera.specs.execution_spec import (
            ExecutionSpecModel, SearchConfig, EvaluationConfig,
            LearningConfig,
        )

        search_cfg = SearchConfig(
            max_nodes=cli_args.get("max_nodes", 100),
            max_depth=cli_args.get("max_depth", 10),
            branch_factor=cli_args.get("branch_factor", 3),
            lambda_cost=cli_args.get("lambda_cost", 0.1),
            beta_exploration=cli_args.get("beta", 0.05),
            repeats=cli_args.get("repeats", 3),
            lcb_coef=cli_args.get("lcb_coef", 1.96),
            sequential_eval=not cli_args.get("no_sequential", False),
            sequential_eval_topk=cli_args.get("seq_topk", 5),
        )
        eval_cfg = EvaluationConfig()
        learn_cfg = LearningConfig(
            lr=cli_args.get("lr", 1e-4),
            clip_range=cli_args.get("clip", 0.2),
            steps_per_update=cli_args.get("ppo_steps", 128),
        )
        exec_spec = ExecutionSpecModel(
            search=search_cfg,
            evaluation=eval_cfg,
            learning=learn_cfg,
        )
        return exec_spec.model_dump()

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
