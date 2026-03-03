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

        fields = self._extract_prompt_fields(input1, related_work)
        prompt = SPEC_GENERATION_PROMPT.format_map(fields)

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
                    logger.warning(f"ProblemSpec validation failed (attempt {attempt + 1}): {e}")
                    prompt += f"\n\nPrevious attempt failed validation: {e}\nPlease fix and regenerate."
            else:
                logger.warning(f"JSON parse failed (attempt {attempt + 1})")

        # Fallback: return defaults with Input-1 fields mapped
        logger.warning("All ProblemSpec generation attempts failed, using defaults")
        from sera.specs.problem_spec import ProblemSpecModel, ObjectiveConfig

        if isinstance(input1, dict):
            task_brief = input1.get("task", {}).get("brief", "Research")
            goal = input1.get("goal", {})
            goal_metric = goal.get("metric", "")
            goal_direction = goal.get("direction", "maximize")
            goal_objective = goal.get("objective", "")
        else:
            task_obj = getattr(input1, "task", None)
            task_brief = getattr(task_obj, "brief", "Research") if task_obj else "Research"
            goal_obj = getattr(input1, "goal", None)
            goal_metric = getattr(goal_obj, "metric", "") if goal_obj else ""
            goal_direction = getattr(goal_obj, "direction", "maximize") if goal_obj else "maximize"
            goal_objective = getattr(goal_obj, "objective", "") if goal_obj else ""

        objective_cfg = ObjectiveConfig(
            description=goal_objective or task_brief,
            metric_name=goal_metric or "score",
            direction=goal_direction,
        )
        defaults = ProblemSpecModel(
            title=task_brief,
            objective=objective_cfg,
        )
        return defaults.model_dump()

    async def build_plan_spec(self, input1: Any, problem_spec: Any) -> dict:
        """Generate PlanSpec JSON using LLM."""
        from sera.agent.prompt_templates import SPEC_GENERATION_PROMPT

        rw = problem_spec if isinstance(problem_spec, dict) else (problem_spec.model_dump() if hasattr(problem_spec, "model_dump") else {})
        fields = self._extract_prompt_fields(input1, rw)
        prompt = SPEC_GENERATION_PROMPT.format_map(fields)

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
                    logger.warning(f"PlanSpec validation failed (attempt {attempt + 1}): {e}")
            else:
                logger.warning(f"JSON parse failed (attempt {attempt + 1})")

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
            ModelSpecModel,
            BaseModelConfig,
            AgentLLMConfig,
            AdapterSpec,
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
            default_targets = _DEFAULT_MODEL_FAMILIES[family].get("default_target_modules", default_targets)

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
            cpu_cores, gpu_type, gpu_required, timeout, no_web, work_dir,
            container_enabled, container_runtime, container_image, etc.
        """
        from sera.specs.resource_spec import (
            ResourceSpecModel,
            ComputeConfig,
            SlurmConfig,
            ContainerConfig,
            SandboxConfig,
            StorageConfig,
            NetworkConfig,
        )

        # Build ContainerConfig if enabled
        container_cfg = ContainerConfig()
        if cli_args.get("container_enabled", False):
            container_cfg = ContainerConfig(
                enabled=True,
                runtime=cli_args.get("container_runtime", "singularity"),
                image=cli_args.get("container_image", ""),
                bind_mounts=cli_args.get("container_bind_mounts", []),
                gpu_enabled=cli_args.get("container_gpu_enabled", True),
                extra_flags=cli_args.get("container_extra_flags", []),
                overlay=cli_args.get("container_overlay", ""),
                writable_tmpfs=cli_args.get("container_writable_tmpfs", False),
            )

        # Build SlurmConfig with container
        slurm_cfg = SlurmConfig(container=container_cfg)

        compute_cfg = ComputeConfig(
            executor_type=cli_args.get("executor", "local"),
            gpu_count=cli_args.get("gpu_count", 1),
            memory_gb=cli_args.get("memory_gb", 32),
            cpu_cores=cli_args.get("cpu_cores", 8),
            gpu_type=cli_args.get("gpu_type", ""),
            gpu_required=cli_args.get("gpu_required", True),
            slurm=slurm_cfg,
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
            ExecutionSpecModel,
            SearchConfig,
            EvaluationConfig,
            LearningConfig,
            LoraRuntimeConfig,
            PruningConfig,
            TerminationConfig,
            PaperExecConfig,
        )

        search_cfg = SearchConfig(
            max_nodes=cli_args.get("max_nodes", 100),
            max_depth=cli_args.get("max_depth", 10),
            branch_factor=cli_args.get("branch_factor", 3),
            lambda_cost=cli_args.get("lambda_cost", 0.1),
            beta_exploration=cli_args.get("beta", 0.05),
            strategy=cli_args.get("strategy", "best_first"),
        )
        eval_cfg = EvaluationConfig(
            repeats=cli_args.get("repeats", 3),
            lcb_coef=cli_args.get("lcb_coef", 1.96),
            sequential_eval=not cli_args.get("no_sequential", False),
            sequential_eval_topk=cli_args.get("seq_topk", 5),
        )
        learn_cfg = LearningConfig(
            lr=cli_args.get("lr", 1e-4),
            clip_range=cli_args.get("clip", 0.2),
            steps_per_update=cli_args.get("ppo_steps", 128),
            kl_control=not cli_args.get("no_kl", False),
        )
        lora_cfg = LoraRuntimeConfig(
            squash_depth=cli_args.get("squash_depth", 6),
            snapshot_on_topk=not cli_args.get("no_snapshot_topk", False),
        )
        pruning_cfg = PruningConfig()
        term_cfg = TerminationConfig()
        paper_cfg = PaperExecConfig()
        exec_spec = ExecutionSpecModel(
            search=search_cfg,
            evaluation=eval_cfg,
            learning=learn_cfg,
            lora_runtime=lora_cfg,
            pruning=pruning_cfg,
            termination=term_cfg,
            paper=paper_cfg,
        )
        return exec_spec.model_dump()

    def _extract_prompt_fields(self, input1: Any, related_work: Any) -> dict[str, str]:
        """Extract individual fields from Input-1 and RelatedWork for prompt template formatting."""
        i1 = input1 if isinstance(input1, dict) else (input1.model_dump() if hasattr(input1, "model_dump") else {})
        rw = related_work if isinstance(related_work, dict) else (related_work.model_dump() if hasattr(related_work, "model_dump") else {})

        task = i1.get("task", {}) if isinstance(i1.get("task"), dict) else {}
        domain = i1.get("domain", {}) if isinstance(i1.get("domain"), dict) else {}
        goal = i1.get("goal", {}) if isinstance(i1.get("goal"), dict) else {}
        data = i1.get("data", {}) if isinstance(i1.get("data"), dict) else {}

        # Build related work summaries
        papers = rw.get("papers", [])
        clusters = rw.get("clusters", [])
        baselines = rw.get("baseline_candidates", [])
        metrics = rw.get("common_metrics", [])
        open_problems = rw.get("open_problems", [])

        rw_summary = "\n".join(
            f"- {p.get('title', 'Untitled')} ({p.get('year', '?')})" for p in papers[:10]
        ) if papers else "No related work available."

        return {
            "task_brief": task.get("brief", "Research task"),
            "field": domain.get("field", ""),
            "subfield": domain.get("subfield", ""),
            "data_location": data.get("location", ""),
            "data_description": data.get("description", ""),
            "data_format": data.get("format", ""),
            "data_size": data.get("size_hint", ""),
            "goal_objective": goal.get("objective", ""),
            "goal_direction": goal.get("direction", "maximize"),
            "baseline": goal.get("baseline", ""),
            "constraints_json": json.dumps(i1.get("constraints", []), default=str),
            "notes": i1.get("notes", ""),
            "related_work_summary": rw_summary,
            "baseline_candidates_json": json.dumps(baselines, default=str),
            "common_metrics_json": json.dumps(metrics, default=str),
            "open_problems_json": json.dumps(open_problems, default=str),
        }

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
