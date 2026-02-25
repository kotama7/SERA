"""AIDE-inspired tree operators: draft, debug, improve per section 6.5.

These three operators form the core branching mechanism of the search tree.
Each operator uses the agent LLM to generate new SearchNode proposals.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sera.search.search_node import SearchNode
from sera.search.validation import validate_experiment_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates (loaded from sera/prompts/search.yaml)
# ---------------------------------------------------------------------------

from sera.prompts import get_search_category_prompts, get_search_prompt

DRAFT_PROMPT = get_search_prompt("draft")
DRAFT_CATEGORY_PROMPTS = get_search_category_prompts()
DRAFT_CATEGORY_PROMPT = get_search_prompt("draft_category")
DEBUG_PROMPT = get_search_prompt("debug")
IMPROVE_PROMPT = get_search_prompt("improve")


class TreeOps:
    """AIDE-inspired branching operators for the search tree.

    Parameters
    ----------
    specs : object
        Contains problem_spec and execution_spec.
    agent_llm : object
        LLM client with an async ``generate(prompt, temperature)`` method.
    rng : random.Random | None
        Optional random number generator for reproducibility.
    agent_loop : AgentLoop | None
        Optional ReAct loop for multi-step tool-using reasoning.
        When provided, operators with tool bindings (debug, improve) use
        the agent loop instead of single-shot LLM calls.
    """

    def __init__(self, specs: Any, agent_llm: Any, rng: Any = None, agent_loop: Any = None):
        self.specs = specs
        self.agent_llm = agent_llm
        self.rng = rng
        self.agent_loop = agent_loop

        # Resolve function → tool bindings from plan_spec
        plan = getattr(specs, "plan", None)
        agent_commands = getattr(plan, "agent_commands", None)
        functions_cfg = getattr(agent_commands, "functions", None)
        self._function_tool_bindings: dict[str, list[str]] = {}
        if functions_cfg is not None:
            bindings = getattr(functions_cfg, "function_tool_bindings", {})
            if isinstance(bindings, dict):
                self._function_tool_bindings = bindings

        # Resolve per-function loop overrides from plan_spec
        self._loop_overrides: dict[str, dict] = {}
        if agent_commands is not None:
            overrides = getattr(agent_commands, "function_loop_overrides", {})
            if isinstance(overrides, dict):
                self._loop_overrides = overrides

    async def draft(self, n: int, all_nodes: dict | None = None) -> list[SearchNode]:
        """Section 6.5.1: Draft new approaches.

        Root init: split n into baseline / open_problem / novel (n//3 each).
        Re-draft: present existing hypotheses, ask for a different approach.
        Uses DRAFT_PROMPT template. Parses JSON array response.
        On JSON parse failure: retry up to 3 times (temperature += 0.1).
        Final fallback: create a single node with default config.
        """
        problem_spec = self.specs.problem

        # Build problem description from spec
        problem_description = getattr(problem_spec.objective, "description", "Maximize score")
        objective = f"{problem_spec.objective.metric_name} ({problem_spec.objective.direction})"
        variables = ", ".join(f"{v.name}({v.type})" for v in problem_spec.manipulated_variables)
        constraints = ", ".join(f"{c.name}({c.type}, threshold={c.threshold})" for c in problem_spec.constraints)

        # Root draft: split into baseline / open_problem / novel categories
        if not all_nodes:
            return await self._draft_root(n, problem_description, objective, variables, constraints)

        # Re-draft: present existing hypotheses, ask for something different
        existing_hypotheses = [
            node.hypothesis for node in all_nodes.values() if node.hypothesis and node.status != "pruned"
        ]
        existing_context = ""
        if existing_hypotheses:
            existing_context = "Existing hypotheses (propose something DIFFERENT):\n" + "\n".join(
                f"- {h}" for h in existing_hypotheses[:10]
            )

        prompt = DRAFT_PROMPT.format(
            problem_description=problem_description,
            objective=objective,
            variables=variables,
            constraints=constraints,
            existing_context=existing_context,
            n=n,
        )

        proposals = await self._generate_proposals(prompt, n)

        # Convert proposals to SearchNodes
        nodes: list[SearchNode] = []
        for proposal in proposals[:n]:
            node = SearchNode(
                hypothesis=proposal.get("hypothesis", ""),
                experiment_config=proposal.get("experiment_config", {}),
                rationale=proposal.get("rationale", ""),
                branching_op="draft",
                depth=0,
            )
            nodes.append(node)

        return nodes

    async def _draft_root(
        self,
        n: int,
        problem_description: str,
        objective: str,
        variables: str,
        constraints: str,
    ) -> list[SearchNode]:
        """Draft root nodes split into baseline / open_problem / novel categories."""
        n_baseline = n // 3
        n_open_problem = n // 3
        n_novel = n // 3 + (n % 3)

        # Gather related work context if available (TASK.md section 6.5.1)
        related_work = getattr(self.specs, "related_work", None)
        baseline_candidates_text = ""
        open_problems_text = ""
        if related_work is not None:
            candidates = getattr(related_work, "baseline_candidates", [])
            if candidates:
                baseline_candidates_text = "\n\nKnown baseline candidates:\n" + "\n".join(
                    f"- {getattr(c, 'name', str(c))}: {getattr(c, 'reported_metric', {})}" for c in candidates[:5]
                )
            problems = getattr(related_work, "open_problems", [])
            if problems:
                open_problems_text = "\n\nKnown open problems:\n" + "\n".join(
                    f"- {getattr(p, 'description', str(p))}" for p in problems[:5]
                )

        categories = [
            ("baseline", n_baseline),
            ("open_problem", n_open_problem),
            ("novel", n_novel),
        ]

        all_nodes: list[SearchNode] = []
        for category, count in categories:
            if count <= 0:
                continue
            category_instruction = DRAFT_CATEGORY_PROMPTS[category]
            # Inject related work context into relevant categories
            if category == "baseline" and baseline_candidates_text:
                category_instruction += baseline_candidates_text
            elif category == "open_problem" and open_problems_text:
                category_instruction += open_problems_text
            prompt = DRAFT_CATEGORY_PROMPT.format(
                problem_description=problem_description,
                objective=objective,
                variables=variables,
                constraints=constraints,
                category_instruction=category_instruction,
                n=count,
            )

            proposals = await self._generate_proposals(prompt, count)

            for proposal in proposals[:count]:
                rationale = proposal.get("rationale", "")
                rationale = f"[{category}] {rationale}"
                node = SearchNode(
                    hypothesis=proposal.get("hypothesis", ""),
                    experiment_config=proposal.get("experiment_config", {}),
                    rationale=rationale,
                    branching_op="draft",
                    depth=0,
                )
                all_nodes.append(node)

        return all_nodes

    async def _generate_proposals(self, prompt: str, n: int) -> list[dict]:
        """Generate proposals via LLM with retry logic.

        Uses ``AgentLLM.call_function("search_draft")`` when available,
        falling back to direct ``generate()`` + ``_parse_json_response()``.
        """
        temperature = getattr(self.specs, "_draft_temperature", 0.7)

        # Prefer call_function path
        if hasattr(self.agent_llm, "call_function"):
            result = await self.agent_llm.call_function(
                "search_draft", prompt=prompt, purpose="draft", temperature=temperature
            )
            if isinstance(result, list) and result:
                return result
            if isinstance(result, dict):
                return [result]
            # Fallback on call_function returning None/empty
            logger.warning("call_function(search_draft) returned empty, using fallback")
            return [
                {
                    "hypothesis": "Baseline approach with default configuration",
                    "experiment_config": {},
                    "rationale": "Fallback: LLM failed to produce valid JSON",
                }
            ]

        # Legacy path (deprecated — use call_function instead)
        proposals = None
        for attempt in range(3):
            try:
                response = await self.agent_llm.generate(
                    prompt, purpose="draft", temperature=temperature + attempt * 0.1
                )
                parsed = self._parse_json_response(response)
                if parsed is not None:
                    if isinstance(parsed, list):
                        proposals = parsed
                    elif isinstance(parsed, dict):
                        proposals = [parsed]
                    break
            except Exception as e:
                logger.warning("Draft attempt %d failed: %s", attempt + 1, e)

        if not proposals:
            logger.warning("All draft attempts failed, creating fallback node")
            proposals = [
                {
                    "hypothesis": "Baseline approach with default configuration",
                    "experiment_config": {},
                    "rationale": "Fallback: LLM failed to produce valid JSON",
                }
            ]

        return proposals

    async def debug(self, failed_node: SearchNode) -> SearchNode:
        """Section 6.5.2: Fix failed experiment code.

        Present experiment_code + error_message to LLM.
        Returns new node with same config but fixed code, debug_depth+1.

        When ``agent_loop`` is available and ``search_debug`` has tool
        bindings (e.g. read_experiment_log, read_file), the ReAct loop
        is used so the agent can inspect logs and files before proposing
        a fix.
        """
        problem_spec = self.specs.problem
        problem_description = getattr(problem_spec.objective, "description", "Maximize score")

        # Get code block tag from language config
        code_block_tag = "python"
        if hasattr(problem_spec, "language"):
            code_block_tag = getattr(problem_spec.language, "code_block_tag", "python")

        prompt = DEBUG_PROMPT.format(
            problem_description=problem_description,
            hypothesis=failed_node.hypothesis,
            experiment_config=json.dumps(failed_node.experiment_config, indent=2),
            error_message=failed_node.error_message or "Unknown error",
            experiment_code=failed_node.experiment_code or "# No code available",
            code_block_tag=code_block_tag,
        )

        temperature = getattr(self.specs, "_debug_temperature", 0.5)
        parsed = None

        # Use AgentLoop if available and search_debug has tool bindings
        tool_bindings = self._function_tool_bindings.get("search_debug", [])
        if self.agent_loop is not None and tool_bindings:
            try:
                tool_hint = (
                    "\n\nBefore answering, use the available tools to investigate the issue. "
                    f"Available tools: {', '.join(tool_bindings)}. "
                    "To call a tool, output ONLY:\n"
                    '<tool_call>\n{"name": "tool_name", "arguments": {"key": "value"}}\n</tool_call>\n'
                    "Do NOT hallucinate tool results. Call the tool and wait for the actual result."
                )
                loop_result = await self.agent_loop.run(
                    task_prompt=prompt + tool_hint,
                    purpose="debug",
                    available_tools=tool_bindings,
                )
                if loop_result.final_output:
                    parsed = self._parse_json_response(loop_result.final_output)
                logger.info(
                    "AgentLoop debug completed: steps=%d, tools=%d, exit=%s",
                    loop_result.total_steps,
                    loop_result.total_tool_calls,
                    loop_result.exit_reason,
                )
            except Exception as e:
                logger.warning("AgentLoop debug failed, falling back to single-shot: %s", e)

        # Fallback to single-shot call_function
        if parsed is None:
            if hasattr(self.agent_llm, "call_function"):
                parsed = await self.agent_llm.call_function(
                    "search_debug", prompt=prompt, purpose="debug", temperature=temperature
                )
            else:
                response = await self.agent_llm.generate(prompt, purpose="debug", temperature=temperature)
                parsed = self._parse_json_response(response)

        if parsed and isinstance(parsed, dict):
            new_node = SearchNode(
                parent_id=failed_node.node_id,
                depth=failed_node.depth + 1,
                hypothesis=parsed.get("hypothesis", failed_node.hypothesis),
                experiment_config=failed_node.experiment_config,  # Debug must not change config
                experiment_code=parsed.get("experiment_code"),
                branching_op="debug",
                rationale=parsed.get("rationale", ""),
                debug_depth=failed_node.debug_depth + 1,
            )
        else:
            # Fallback: clone parent with incremented debug depth
            new_node = SearchNode(
                parent_id=failed_node.node_id,
                depth=failed_node.depth + 1,
                hypothesis=failed_node.hypothesis,
                experiment_config=failed_node.experiment_config,
                experiment_code=failed_node.experiment_code,
                branching_op="debug",
                rationale="Debug: LLM failed to parse response",
                debug_depth=failed_node.debug_depth + 1,
            )

        return new_node

    async def improve(
        self,
        parent: SearchNode,
        all_nodes: dict[str, SearchNode],
        n_children: int,
    ) -> list[SearchNode]:
        """Section 6.5.3: Propose atomic improvements.

        Build context with parent stats, sibling summaries (section 6.8.1).
        Uses IMPROVE_PROMPT. Parse JSON array.
        Validate each child with validate_experiment_config.
        Log warning if diff > 1 key.
        """
        problem_spec = self.specs.problem
        problem_description = getattr(problem_spec.objective, "description", "Maximize score")
        objective = f"{problem_spec.objective.metric_name} ({problem_spec.objective.direction})"
        variables = ", ".join(f"{v.name}({v.type})" for v in problem_spec.manipulated_variables)

        sibling_context = self._build_sibling_context(parent, all_nodes)
        failure_context = self._build_failure_context(parent)

        prompt = IMPROVE_PROMPT.format(
            problem_description=problem_description,
            objective=objective,
            variables=variables,
            parent_hypothesis=parent.hypothesis,
            parent_config=json.dumps(parent.experiment_config, indent=2),
            parent_mu=parent.mu,
            parent_se=parent.se,
            parent_lcb=parent.lcb,
            parent_feasible=parent.feasible,
            sibling_context=sibling_context,
            failure_context=failure_context,
            n_children=n_children,
        )

        temperature = getattr(self.specs, "_improve_temperature", 0.7)

        proposals = None

        # Use AgentLoop if available and search_improve has tool bindings
        tool_bindings = self._function_tool_bindings.get("search_improve", [])
        if self.agent_loop is not None and tool_bindings:
            try:
                tool_hint = (
                    "\n\nBefore proposing improvements, use the available tools to gather information. "
                    f"Available tools: {', '.join(tool_bindings)}. "
                    "To call a tool, output ONLY:\n"
                    '<tool_call>\n{"name": "tool_name", "arguments": {"key": "value"}}\n</tool_call>\n'
                    "Do NOT hallucinate tool results. Call the tool and wait for the actual result."
                )
                loop_result = await self.agent_loop.run(
                    task_prompt=prompt + tool_hint,
                    purpose="improve",
                    available_tools=tool_bindings,
                )
                if loop_result.final_output:
                    parsed = self._parse_json_response(loop_result.final_output)
                    if isinstance(parsed, list) and parsed:
                        proposals = parsed
                    elif isinstance(parsed, dict):
                        proposals = [parsed]
                logger.info(
                    "AgentLoop improve completed: steps=%d, tools=%d, exit=%s",
                    loop_result.total_steps,
                    loop_result.total_tool_calls,
                    loop_result.exit_reason,
                )
            except Exception as e:
                logger.warning("AgentLoop improve failed, falling back to single-shot: %s", e)

        # Fallback to single-shot
        if proposals is None:
            if hasattr(self.agent_llm, "call_function"):
                result = await self.agent_llm.call_function(
                    "search_improve", prompt=prompt, purpose="improve", temperature=temperature
                )
                if isinstance(result, list) and result:
                    proposals = result
                elif isinstance(result, dict):
                    proposals = [result]
            else:
                for attempt in range(3):
                    try:
                        response = await self.agent_llm.generate(
                            prompt, purpose="improve", temperature=temperature + attempt * 0.1
                        )
                        parsed = self._parse_json_response(response)
                        if parsed is not None:
                            if isinstance(parsed, list):
                                proposals = parsed
                            elif isinstance(parsed, dict):
                                proposals = [parsed]
                            break
                    except Exception as e:
                        logger.warning("Improve attempt %d failed: %s", attempt + 1, e)

        if not proposals:
            logger.warning("All improve attempts failed, returning empty list")
            return []

        nodes: list[SearchNode] = []
        for proposal in proposals[:n_children]:
            exp_config = proposal.get("experiment_config", {})

            # Validate config with retry: feed validation errors back to LLM up to 2 times
            is_valid, errors = validate_experiment_config(exp_config, problem_spec)
            if not is_valid:
                logger.warning("Invalid experiment config from improve: %s", errors)
                max_validation_retries = 2
                for retry_i in range(max_validation_retries):
                    retry_prompt = (
                        f"Your previous proposal had validation errors:\n"
                        f"{chr(10).join(errors)}\n\n"
                        f"Original proposal:\n{json.dumps(proposal, indent=2)}\n\n"
                        f"Parent config:\n{json.dumps(parent.experiment_config, indent=2)}\n\n"
                        f"Variables: {variables}\n"
                        f"Objective: {objective}\n\n"
                        f"Please fix the experiment_config and return a corrected JSON object with "
                        f"keys: hypothesis, experiment_config, rationale.\n"
                        f"Output ONLY the JSON, no other text."
                    )
                    retry_temperature = temperature + (retry_i + 1) * 0.1
                    try:
                        if hasattr(self.agent_llm, "call_function"):
                            retry_result = await self.agent_llm.call_function(
                                "search_improve",
                                prompt=retry_prompt,
                                purpose="improve_validation_retry",
                                temperature=retry_temperature,
                            )
                            if isinstance(retry_result, list) and retry_result:
                                retry_result = retry_result[0]
                        else:
                            retry_response = await self.agent_llm.generate(
                                retry_prompt,
                                purpose="improve_validation_retry",
                                temperature=retry_temperature,
                            )
                            retry_result = self._parse_json_response(retry_response)
                            if isinstance(retry_result, list) and retry_result:
                                retry_result = retry_result[0]

                        if isinstance(retry_result, dict):
                            exp_config = retry_result.get("experiment_config", exp_config)
                            proposal = retry_result
                            is_valid, errors = validate_experiment_config(exp_config, problem_spec)
                            if is_valid:
                                logger.info(
                                    "Validation retry %d/%d succeeded for improve proposal",
                                    retry_i + 1,
                                    max_validation_retries,
                                )
                                break
                            else:
                                logger.warning(
                                    "Validation retry %d/%d still invalid: %s",
                                    retry_i + 1,
                                    max_validation_retries,
                                    errors,
                                )
                        else:
                            logger.warning(
                                "Validation retry %d/%d returned non-dict, skipping",
                                retry_i + 1,
                                max_validation_retries,
                            )
                    except Exception as e:
                        logger.warning(
                            "Validation retry %d/%d failed with exception: %s",
                            retry_i + 1,
                            max_validation_retries,
                            e,
                        )

                if not is_valid:
                    logger.warning("Skipping proposal after %d validation retries", max_validation_retries)
                    continue

            # Warn if more than 1 key differs from parent
            diff_keys = [
                k
                for k in set(list(exp_config.keys()) + list(parent.experiment_config.keys()))
                if exp_config.get(k) != parent.experiment_config.get(k)
            ]
            if len(diff_keys) > 1:
                logger.warning(
                    "Improve changed %d keys (%s), expected at most 1",
                    len(diff_keys),
                    diff_keys,
                )

            node = SearchNode(
                parent_id=parent.node_id,
                depth=parent.depth + 1,
                hypothesis=proposal.get("hypothesis", ""),
                experiment_config=exp_config,
                branching_op="improve",
                rationale=proposal.get("rationale", ""),
            )
            nodes.append(node)

        return nodes

    def _parse_json_response(self, response: str) -> list[dict] | dict | None:
        """Extract JSON from LLM response.

        Try ```json blocks first, then raw parse.
        """
        # Try to extract ```json ... ``` block
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw parse of the entire response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find any JSON array or object in the response
        for pattern in [
            r"\[.*\]",
            r"\{.*\}",
        ]:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        return None

    @staticmethod
    def _build_failure_context(node: SearchNode) -> str:
        """Build failure context string from node.failure_context (ECHO)."""
        failure_ctx = getattr(node, "failure_context", [])
        if not failure_ctx:
            return ""

        lines = ["Known failure patterns (avoid these):"]
        for fc in failure_ctx:
            hypothesis = fc.get("hypothesis", "unknown")
            category = fc.get("error_category", "unknown")
            lesson = fc.get("lesson", "")
            lines.append(f"- [{category}] {hypothesis}: {lesson}")
        return "\n".join(lines)

    def _build_sibling_context(self, parent: SearchNode, all_nodes: dict[str, SearchNode]) -> str:
        """Build sibling context per section 6.8.1 rules.

        - Only status=evaluated siblings (same parent_id)
        - Sort by LCB desc, take top sibling_context_k
        - Exclude pruned/failed nodes
        - Include best_node regardless of parent
        - Show: hypothesis, config diff, mu +/- SE, LCB, feasible, eval_runs
        """
        exec_spec = getattr(self.specs, "execution", None)
        search_cfg = getattr(exec_spec, "search", None) if exec_spec else None
        sibling_context_k = getattr(search_cfg, "sibling_context_k", 5) if search_cfg else 5
        if not isinstance(sibling_context_k, int):
            sibling_context_k = 5

        # Find siblings: same parent_id, evaluated, not pruned/failed
        siblings = [
            node
            for node in all_nodes.values()
            if node.parent_id == parent.parent_id
            and node.node_id != parent.node_id
            and node.status == "evaluated"
            and node.lcb is not None
        ]

        # Sort by LCB descending
        siblings.sort(key=lambda n: n.lcb if n.lcb is not None else float("-inf"), reverse=True)
        siblings = siblings[:sibling_context_k]

        # Find global best node
        best_node = None
        for node in all_nodes.values():
            if node.status == "evaluated" and node.lcb is not None:
                if best_node is None or (node.lcb > best_node.lcb):
                    best_node = node

        # Include best if not already in siblings and not the parent itself
        if best_node and best_node.node_id != parent.node_id:
            if best_node.node_id not in {s.node_id for s in siblings}:
                siblings.append(best_node)

        # Include up to 2 constraint violation examples from pruned siblings (section 6.8.1)
        pruned_violations = [
            node
            for node in all_nodes.values()
            if node.parent_id == parent.parent_id
            and node.node_id != parent.node_id
            and node.status == "pruned"
            and not getattr(node, "feasible", True)
        ][:2]

        if not siblings and not pruned_violations:
            return "No sibling experiments available yet."

        lines = ["Related experiments:"]

        # Show constraint violation examples first as warnings
        for pv in pruned_violations:
            diff_parts = []
            all_keys = set(list(pv.experiment_config.keys()) + list(parent.experiment_config.keys()))
            for k in sorted(all_keys):
                sv = pv.experiment_config.get(k)
                pk = parent.experiment_config.get(k)
                if sv != pk:
                    diff_parts.append(f"{k}: {pk} -> {sv}")
            diff_str = ", ".join(diff_parts) if diff_parts else "same config"
            lines.append(
                f"- [CONSTRAINT VIOLATION - pruned] {pv.hypothesis}\n"
                f"  Config diff: {diff_str}\n"
                f"  Reason: infeasible (violated constraints)"
            )

        for sib in siblings:
            is_best = best_node and sib.node_id == best_node.node_id
            label = " [BEST]" if is_best else ""

            # Config diff from parent
            diff_parts = []
            all_keys = set(list(sib.experiment_config.keys()) + list(parent.experiment_config.keys()))
            for k in sorted(all_keys):
                sv = sib.experiment_config.get(k)
                pv = parent.experiment_config.get(k)
                if sv != pv:
                    diff_parts.append(f"{k}: {pv} -> {sv}")

            diff_str = ", ".join(diff_parts) if diff_parts else "same config"

            lines.append(
                f"- {sib.hypothesis}{label}\n"
                f"  Config diff: {diff_str}\n"
                f"  mu={sib.mu:.4f} +/- SE={sib.se:.4f}, LCB={sib.lcb:.4f}\n"
                f"  feasible={sib.feasible}, eval_runs={sib.eval_runs}"
            )

        return "\n".join(lines)
