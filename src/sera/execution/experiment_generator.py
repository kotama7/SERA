"""ExperimentGenerator: LLM-driven experiment code generation.

Takes a SearchNode and ProblemSpec, uses the AgentLLM to generate
experiment code, and writes it to the run directory.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from sera.prompts import get_prompt

EXPERIMENT_CODE_PROMPT = get_prompt("experiment_code_runtime")


class ExperimentGenerator:
    """Generate experiment scripts from search nodes.

    Parameters
    ----------
    agent_llm : object
        LLM client with ``generate(prompt, temperature)`` method.
    problem_spec : object
        The problem specification.
    work_dir : str | Path
        Base workspace directory.
    """

    def __init__(
        self,
        agent_llm: Any,
        problem_spec: Any,
        work_dir: str | Path = "./sera_workspace",
    ):
        self.agent_llm = agent_llm
        self.problem_spec = problem_spec
        self.work_dir = Path(work_dir)

    def _get_language_config(self) -> Any:
        """Get language config from problem spec, with defaults for Python."""
        if hasattr(self.problem_spec, "language"):
            return self.problem_spec.language
        # Return a simple namespace with Python defaults
        from types import SimpleNamespace

        return SimpleNamespace(
            name="python",
            interpreter_command="python",
            file_extension=".py",
            seed_arg_format="--seed {seed}",
            code_block_tag="python",
        )

    async def generate(self, node: Any) -> Path:
        """Generate experiment code for a search node.

        Parameters
        ----------
        node : SearchNode
            The node to generate code for. If node.experiment_code is
            already set (e.g., from a debug operation), that code is used
            directly instead of calling the LLM.

        Returns
        -------
        Path
            Path to the written experiment script file.
        """
        lang_config = self._get_language_config()
        script_filename = "experiment" + lang_config.file_extension

        run_dir = self.work_dir / "runs" / node.node_id
        run_dir.mkdir(parents=True, exist_ok=True)
        script_path = run_dir / script_filename

        if node.experiment_code:
            # Use pre-existing code (e.g., from debug operator)
            code = node.experiment_code
        else:
            code = await self._generate_code(node)
            node.experiment_code = code

        script_path.write_text(code, encoding="utf-8")
        logger.info(
            "Generated experiment script: %s (%d chars)",
            script_path,
            len(code),
        )
        return script_path

    async def _generate_code(self, node: Any) -> str:
        """Call the LLM to generate experiment code.

        Parameters
        ----------
        node : SearchNode
            The search node with hypothesis and config.

        Returns
        -------
        str
            The generated code.
        """
        lang_config = self._get_language_config()

        problem_description = getattr(self.problem_spec.objective, "description", "Maximize score")
        objective = self.problem_spec.objective.metric_name
        direction = self.problem_spec.objective.direction
        higher_is_better = str(direction == "maximize").lower()
        data_location = ""
        if hasattr(self.problem_spec, "data_location"):
            data_location = self.problem_spec.data_location
        elif hasattr(self.problem_spec, "evaluation_design"):
            data_location = "provided by evaluation framework"

        # Describe seed argument based on language config
        seed_arg_description = lang_config.seed_arg_format.replace("{seed}", "<int>")

        # Include experiment template if available
        template = getattr(self.problem_spec, "experiment_template", "")
        template_section = ""
        if template:
            template_section = f"Base template (modify and improve this code based on the hypothesis):\n```{lang_config.code_block_tag}\n{template}\n```"

        prompt = EXPERIMENT_CODE_PROMPT.format(
            language_name=lang_config.name,
            problem_description=problem_description,
            objective=objective,
            direction=direction,
            hypothesis=node.hypothesis,
            experiment_config=json.dumps(node.experiment_config, indent=2),
            data_location=data_location,
            metric_name=self.problem_spec.objective.metric_name,
            higher_is_better=higher_is_better,
            seed_arg_description=seed_arg_description,
            code_block_tag=lang_config.code_block_tag,
            template_section=template_section,
        )

        # Prefer call_function path
        if hasattr(self.agent_llm, "call_function"):
            code = await self.agent_llm.call_function(
                "experiment_code_gen", prompt=prompt, purpose="experiment_code_gen", temperature=0.5
            )
            if not code:
                code = "# Error: code generation returned empty result"
            return code

        # Legacy path (deprecated — use call_function instead)
        response = await self.agent_llm.generate(prompt, purpose="experiment_code_gen", temperature=0.5)
        code = self._extract_code(response, lang_config.code_block_tag)
        return code

    @staticmethod
    def _extract_code(response: str, code_block_tag: str = "python") -> str:
        """Extract code from an LLM response.

        Tries to find a fenced code block first, then falls back
        to the raw response.
        """
        # Try language-specific code block
        pattern = rf"```{re.escape(code_block_tag)}\s*(.*?)\s*```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try generic code block
        match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Return raw response
        return response.strip()
