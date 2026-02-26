"""SearchManager: Best-First search loop per section 6.11.

Orchestrates the entire research loop: drafting initial hypotheses,
evaluating experiments, debugging failures, improving successes, and
managing PPO updates and pruning.
"""

from __future__ import annotations

import heapq
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

from sera.learning.rollout import PPORollout, PPORolloutV2, PPORolloutV3
from sera.search.search_node import SearchNode
from sera.search.priority import compute_priority
from sera.utils.checkpoint import save_checkpoint, load_latest_checkpoint

logger = logging.getLogger(__name__)


def sync_adapter_assignment(
    search_node: SearchNode,
    ppo_updated: bool,
    new_adapter_node_id: str | None,
    all_nodes: dict[str, SearchNode],
) -> None:
    """Synchronise search-tree node with LoRA lineage tree (section 9.1.1).

    Three rules:
    1. No PPO update -> inherit parent's adapter_node_id.
    2. PPO updated -> set the new adapter_node_id.
    3. Root node (parent_id is None) -> adapter_node_id = "adapter_root".
    """
    if search_node.parent_id is None:
        # Rule 3: Root nodes get adapter_root
        search_node.adapter_node_id = "adapter_root"
    elif ppo_updated and new_adapter_node_id is not None:
        # Rule 2: PPO update produced a new adapter
        search_node.adapter_node_id = new_adapter_node_id
    else:
        # Rule 1: Inherit parent's adapter_node_id
        parent = all_nodes.get(search_node.parent_id)
        if parent and parent.adapter_node_id:
            search_node.adapter_node_id = parent.adapter_node_id


class SearchManager:
    """Best-first search manager for the SERA research loop.

    Parameters
    ----------
    specs : object
        Combined spec object with execution_spec, problem_spec, etc.
    agent_llm : object
        LLM client for hypothesis generation.
    executor : Executor
        Experiment runner (local, slurm, docker).
    evaluator : Evaluator
        Statistical evaluation of experiment results.
    ppo_trainer : object | None
        PPO trainer for adapter updates.
    lineage_manager : object | None
        Adapter lineage manager.
    tree_ops : TreeOps
        Branching operators (draft, debug, improve).
    pruner : object | None
        Node pruner.
    logger_obj : object | None
        JSONL logger for structured logging.
    checkpoint_dir : str | Path
        Directory for saving checkpoints.
    """

    def __init__(
        self,
        specs: Any,
        agent_llm: Any,
        executor: Any,
        evaluator: Any,
        ppo_trainer: Any,
        lineage_manager: Any,
        tree_ops: Any,
        pruner: Any,
        logger_obj: Any = None,
        checkpoint_dir: str | Path = "./checkpoints",
        failure_extractor: Any = None,
        turn_reward_evaluator: Any = None,
    ):
        self.specs = specs
        self.agent_llm = agent_llm
        self.executor = executor
        self.evaluator = evaluator
        self.ppo_trainer = ppo_trainer
        self.lineage_manager = lineage_manager
        self.tree_ops = tree_ops
        self.pruner = pruner
        self.logger = logger_obj
        self.checkpoint_dir = Path(checkpoint_dir)
        self.failure_extractor = failure_extractor
        self.turn_reward_evaluator = turn_reward_evaluator

        # Search state
        self.open_list: list[tuple[float, str]] = []  # (neg_priority, node_id)
        self.closed_set: set[str] = set()
        self.all_nodes: dict[str, SearchNode] = {}
        self.best_node: SearchNode | None = None
        self.step: int = 0
        self.ppo_buffer: list[dict] = []
        self._pending_slurm_handles: list = []  # Active SLURM job handles for SIGINT cleanup
        self._start_time: float = time.monotonic()
        self._plateau_counter: int = 0
        self._best_lcb_for_plateau: float | None = None

        self._setup_signal_handler()

    def _setup_signal_handler(self) -> None:
        """SIGINT handler: cancel pending SLURM jobs, save checkpoint, then exit(20)."""

        def _handler(signum: int, frame: Any) -> None:
            logger.warning("SIGINT received, saving checkpoint and exiting")
            # Cancel pending SLURM jobs if executor supports it
            if hasattr(self.executor, "cancel_all"):
                try:
                    # Collect any active job handles tracked by the executor
                    pending_handles = getattr(self, "_pending_slurm_handles", [])
                    if pending_handles:
                        logger.info("Cancelling %d pending SLURM jobs", len(pending_handles))
                        self.executor.cancel_all(pending_handles)
                except Exception as e:
                    logger.error("Failed to cancel SLURM jobs on SIGINT: %s", e)
            try:
                state = self.save_state()
                save_checkpoint(state, self.checkpoint_dir, self.step)
            except Exception as e:
                logger.error("Failed to save checkpoint on SIGINT: %s", e)
            sys.exit(20)

        signal.signal(signal.SIGINT, _handler)

    async def run(self) -> SearchNode | None:
        """Main research loop per section 6.11.

        1. Draft root children
        2. Loop: select_next_node -> evaluate/debug/draft/improve
        3. PPO update when buffer full
        4. Prune periodically
        5. Checkpoint periodically

        Returns
        -------
        SearchNode | None
            The best node found, or None if no valid results.
        """
        exec_spec = self.specs.execution

        # SLURM batch config
        slurm_batch_size = getattr(exec_spec.search, "slurm_batch_size", 1)
        slurm_max_concurrent = getattr(exec_spec.search, "slurm_max_concurrent", 4)
        slurm_poll_interval_sec = getattr(exec_spec.search, "slurm_poll_interval_sec", 10.0)
        use_slurm_batch = self._is_slurm_executor() and slurm_batch_size > 1

        # Step 1: Draft initial children
        n_initial = getattr(exec_spec.search, "initial_root_children", 5)
        initial_nodes = await self.tree_ops.draft(n_initial, self.all_nodes)
        for node in initial_nodes:
            self._add_node(node)

        # Step 2: Main loop
        while not self._should_terminate():
            # SLURM batch path: submit multiple pending nodes at once
            if use_slurm_batch:
                # Check if there are pending nodes suitable for batching
                pending_count = sum(
                    1 for _, nid in self.open_list
                    if nid not in self.closed_set and nid in self.all_nodes
                    and self.all_nodes[nid].status == "pending"
                )
                if pending_count >= slurm_batch_size:
                    await self._run_batched_pipeline(
                        batch_size=slurm_batch_size,
                        max_concurrent=slurm_max_concurrent,
                        poll_interval_sec=slurm_poll_interval_sec,
                    )
                    # Continue to PPO/prune/checkpoint steps below
                    # (step counter is incremented inside _run_batched_pipeline)
                    await self._run_post_step_tasks(exec_spec)
                    continue

            self.step += 1

            node, operator = self.select_next_node()
            if node is None:
                logger.info("No more nodes to expand, terminating")
                break

            logger.info(
                "Step %d: %s on node %s (depth=%d)",
                self.step,
                operator,
                node.node_id[:8],
                node.depth,
            )

            try:
                if operator == "evaluate":
                    await self._evaluate_node(node)
                elif operator == "debug":
                    # ECHO: extract failure knowledge before debug
                    if self.failure_extractor is not None:
                        try:
                            summary = self.failure_extractor.extract(node)
                            # Inject into pending siblings (same parent)
                            siblings = [
                                n
                                for n in self.all_nodes.values()
                                if n.parent_id == node.parent_id
                                and n.node_id != node.node_id
                                and n.status in ("pending", "evaluated")
                            ]
                            self.failure_extractor.inject(summary, siblings)
                        except Exception as e:
                            logger.warning("ECHO extraction failed for node %s: %s", node.node_id[:8], e)

                    child = await self.tree_ops.debug(node)
                    node.children_ids.append(child.node_id)
                    self._add_node(child)
                elif operator == "draft":
                    # Re-draft (not root init): cap at min(branch_factor, 2) to limit
                    # diversity-triggered drafts to 1-2 nodes
                    branch_factor = getattr(exec_spec.search, "branch_factor", 3)
                    redraft_count = min(branch_factor, 2)
                    new_nodes = await self.tree_ops.draft(
                        redraft_count,
                        self.all_nodes,
                    )
                    for n in new_nodes:
                        self._add_node(n)
                elif operator == "improve":
                    n_children = getattr(exec_spec.search, "branch_factor", 3)
                    children = await self.tree_ops.improve(node, self.all_nodes, n_children)
                    for child in children:
                        node.children_ids.append(child.node_id)
                        self._add_node(child)
                    node.status = "expanded"
            except Exception as e:
                logger.error(
                    "Error processing node %s with op %s: %s",
                    node.node_id[:8],
                    operator,
                    e,
                )
                node.mark_failed(str(e))

            # Log node state
            if self.logger:
                self.logger.log(
                    {
                        "event": "node_selected",
                        "step": self.step,
                        "node_id": node.node_id,
                        "parent_id": node.parent_id,
                        "depth": node.depth,
                        "operator": operator,
                        "chosen_reason": operator,
                        "status": node.status,
                        "mu": node.mu,
                        "se": node.se,
                        "lcb": node.lcb,
                        "priority": node.priority,
                        "open_list_size": len(self.open_list),
                        "total_nodes": len(self.all_nodes),
                        "budget_consumed": {
                            "steps": self.step,
                            "nodes": len(self.all_nodes),
                            "wall_time_sec": sum(n.wall_time_sec for n in self.all_nodes.values()),
                        },
                    }
                )

            # Step 3: PPO update when enough evaluated nodes (skipped if ppo_trainer is None)
            if self.ppo_trainer is not None and self.ppo_buffer:
                evaluated_list = [n for n in self.all_nodes.values() if n.status == "evaluated"]
                n_evaluated = len(evaluated_list)
                if self.ppo_trainer.should_update(n_evaluated, evaluated_list):
                    try:
                        logger.info(
                            "PPO update triggered (buffer size=%d, n_evaluated=%d)", len(self.ppo_buffer), n_evaluated
                        )
                        rollouts = []
                        for entry in self.ppo_buffer:
                            entry_turn_rewards = entry.get("turn_rewards", {})
                            if entry.get("tool_trajectory"):
                                rollouts.append(
                                    PPORolloutV3(
                                        node_id=entry["node_id"],
                                        prompt=entry.get("hypothesis", ""),
                                        response=str(entry.get("config", {})),
                                        log_prob=0.0,
                                        reward=entry.get("reward", 0.0),
                                        value=0.0,
                                        turn_rewards=entry.get("turn_rewards", {}),
                                        tool_trajectory=entry.get("tool_trajectory", []),
                                        total_tool_calls=entry.get("total_tool_calls", 0),
                                        tool_success_rate=entry.get("tool_success_rate", 0.0),
                                    )
                                )
                            elif entry_turn_rewards:
                                rollouts.append(
                                    PPORolloutV2(
                                        node_id=entry["node_id"],
                                        prompt=entry.get("hypothesis", ""),
                                        response=str(entry.get("config", {})),
                                        log_prob=0.0,
                                        reward=entry.get("reward", 0.0),
                                        value=0.0,
                                        turn_rewards=entry_turn_rewards,
                                    )
                                )
                            else:
                                rollouts.append(
                                    PPORollout(
                                        node_id=entry["node_id"],
                                        prompt=entry.get("hypothesis", ""),
                                        response=str(entry.get("config", {})),
                                        log_prob=0.0,
                                        reward=entry.get("reward", 0.0),
                                        value=0.0,
                                    )
                                )
                        ppo_result = await self.ppo_trainer.update(rollouts, self.agent_llm, self.specs, all_nodes=self.all_nodes)

                        # If PPO produced a new adapter, update nodes' adapter_node_ids
                        new_adapter_id = ppo_result.get("new_adapter_node_id") if isinstance(ppo_result, dict) else None
                        if new_adapter_id:
                            for entry in self.ppo_buffer:
                                nid = entry["node_id"]
                                if nid in self.all_nodes:
                                    sync_adapter_assignment(self.all_nodes[nid], True, new_adapter_id, self.all_nodes)
                            logger.info("Assigned new adapter %s to %d nodes", new_adapter_id[:8], len(self.ppo_buffer))

                        self.ppo_buffer.clear()
                    except Exception as e:
                        logger.error("PPO update failed: %s", e)

            # Step 4: Prune periodically
            prune_interval = (
                getattr(getattr(exec_spec, "pruning", None), "prune_interval", 10)
                if getattr(exec_spec, "pruning", None)
                else 10
            )
            if self.pruner is not None and self.step % prune_interval == 0:
                try:
                    open_nodes = [
                        self.all_nodes[nid]
                        for _, nid in self.open_list
                        if nid in self.all_nodes and nid not in self.closed_set
                    ]
                    workspace_dir = self.checkpoint_dir.parent if self.checkpoint_dir else None
                    pruned_ids = self.pruner.prune(
                        open_nodes, self.closed_set, self.all_nodes, exec_spec,
                        workspace_dir=workspace_dir,
                    )
                    if pruned_ids:
                        pruned_set = set(pruned_ids)
                        self.open_list = [(p, nid) for p, nid in self.open_list if nid not in pruned_set]
                        self.closed_set.update(pruned_set)
                except Exception as e:
                    logger.error("Pruning failed: %s", e)

            # Step 5: Checkpoint periodically
            plan_spec = getattr(self.specs, "plan", None)
            logging_cfg = getattr(plan_spec, "logging", None) if plan_spec else None
            checkpoint_interval = getattr(logging_cfg, "checkpoint_interval", 10) if logging_cfg else 10
            if self.step % checkpoint_interval == 0:
                try:
                    state = self.save_state()
                    save_checkpoint(state, self.checkpoint_dir, self.step)
                except Exception as e:
                    logger.error("Checkpoint save failed: %s", e)

        return self.best_node

    async def _evaluate_node(self, node: SearchNode) -> None:
        """Run evaluation on a node, update stats, and check for improvement."""
        node.status = "running"

        try:
            # Initial evaluation
            await self.evaluator.evaluate_initial(node)

            # If evaluate_initial called mark_failed or detected OOM, don't override
            if node.status in ("failed", "oom"):
                logger.info("Node %s %s during evaluation", node.node_id[:8], node.status)
                self.closed_set.discard(node.node_id)  # Allow debug operator to pick it up
                return

            # Check if in top-k for full eval
            k = getattr(getattr(self.specs.execution, "evaluation", None), "sequential_eval_topk", 5)
            if self._is_topk(node, k):
                await self.evaluator.evaluate_full(node)

            node.mark_evaluated()
            self._update_best(node)

            # Re-queue for improve operator
            priority = compute_priority(node, self.specs.execution)
            node.priority = priority
            self.closed_set.discard(node.node_id)
            heapq.heappush(self.open_list, (-priority, node.node_id))

            # Compute turn-level rewards if evaluator is available
            turn_rewards: dict[str, float] = {}
            if self.turn_reward_evaluator is not None and node.mu is not None:
                parent_node = self.all_nodes.get(node.parent_id) if node.parent_id else None
                try:
                    turn_rewards = self.turn_reward_evaluator.evaluate_all(node, parent_node, self.all_nodes)
                except Exception as e:
                    logger.warning("Turn reward evaluation failed for node %s: %s", node.node_id[:8], e)

            # Build tool trajectory from last agent loop result
            tool_trajectory: list[dict] = []
            loop_result = getattr(self.agent_llm, "_last_loop_result", None)
            if loop_result is not None:
                tool_trajectory = self._build_tool_trajectory(loop_result)

            # Add to PPO buffer
            if node.mu is not None:
                entry: dict[str, Any] = {
                    "node_id": node.node_id,
                    "hypothesis": node.hypothesis,
                    "config": node.experiment_config,
                    "reward": node.mu,
                    "feasible": node.feasible,
                }
                if turn_rewards:
                    entry["turn_rewards"] = turn_rewards
                if tool_trajectory:
                    entry["tool_trajectory"] = tool_trajectory
                    entry["total_tool_calls"] = len(tool_trajectory)
                    entry["tool_success_rate"] = (
                        sum(1 for t in tool_trajectory if t["success"]) / len(tool_trajectory)
                        if tool_trajectory
                        else 0.0
                    )
                self.ppo_buffer.append(entry)
        except Exception as e:
            node.mark_failed(str(e))
            self.closed_set.discard(node.node_id)  # Allow debug operator to pick it up
            logger.error(
                "Evaluation failed for node %s: %s",
                node.node_id[:8],
                e,
            )

    def select_next_node(self) -> tuple[SearchNode | None, str]:
        """Section 6.4: Auto-select operator based on node state.

        Priority order:
        1. pending -> evaluate
        2. failed (debug_depth < max) -> debug  (checked BEFORE improve)
        3. evaluated + low diversity -> draft
        4. evaluated -> improve

        Returns (node, operator) or (None, "") if no nodes available.
        """
        exec_spec = self.specs.execution
        max_debug_depth = getattr(getattr(exec_spec, "search", None), "max_debug_depth", 3)
        max_depth = getattr(exec_spec.search, "max_depth", 10)

        # Pre-compute debuggable nodes (failed with room for debug retries)
        debuggable = [
            n
            for n in self.all_nodes.values()
            if n.status == "failed" and n.debug_depth < max_debug_depth and n.node_id not in self.closed_set
        ]

        # Step 1: Pop from priority queue (pending→evaluate, evaluated→improve)
        while self.open_list:
            neg_priority, node_id = heapq.heappop(self.open_list)
            if node_id in self.closed_set:
                continue
            if node_id not in self.all_nodes:
                continue

            node = self.all_nodes[node_id]
            self.closed_set.add(node_id)

            if node.status == "pending":
                return node, "evaluate"
            elif node.status == "evaluated":
                # Check max_depth before improve
                if node.depth >= max_depth:
                    continue
                # Spec §6.4: debug failed nodes before improving evaluated ones
                if debuggable:
                    debug_node = min(debuggable, key=lambda n: n.debug_depth)
                    self.closed_set.add(debug_node.node_id)
                    # Push current evaluated node back for later improve
                    heapq.heappush(self.open_list, (neg_priority, node_id))
                    self.closed_set.discard(node_id)
                    return debug_node, "debug"
                # Spec §6.4: diversity check before improve
                if self._needs_diversity_draft(exec_spec):
                    # Push current evaluated node back for later improve
                    heapq.heappush(self.open_list, (neg_priority, node_id))
                    self.closed_set.discard(node_id)
                    return (self.best_node or SearchNode()), "draft"
                return node, "improve"

        # Step 2: Fallback — check for debuggable nodes when heap is empty
        if debuggable:
            node = min(debuggable, key=lambda n: n.debug_depth)
            self.closed_set.add(node.node_id)
            return node, "debug"

        # Step 3: Diversity check for draft re-trigger (TASK.md section 6.4)
        if self._needs_diversity_draft(exec_spec):
            return (self.best_node or SearchNode()), "draft"

        # No nodes in open list; check if we need more diversity
        max_nodes = getattr(exec_spec.search, "max_nodes", 100)
        if len(self.all_nodes) < max_nodes:
            # Return a sentinel to trigger drafting
            return (self.best_node or SearchNode()), "draft"

        return None, ""

    def _needs_diversity_draft(self, exec_spec: Any) -> bool:
        """Check if diversity is too low and drafting new approaches is needed (§6.4 step 5)."""
        evaluated = [n for n in self.all_nodes.values() if n.status == "evaluated"]
        unique_methods = len({n.experiment_config.get("method", "") for n in evaluated})
        min_diverse = getattr(exec_spec.search, "min_diverse_methods", 3)
        draft_after = getattr(exec_spec.search, "draft_trigger_after", 10)
        max_nodes = getattr(exec_spec.search, "max_nodes", 100)
        return len(evaluated) >= draft_after and unique_methods < min_diverse and len(self.all_nodes) < max_nodes

    def _should_terminate(self) -> bool:
        """Check all termination conditions from ExecutionSpec.termination."""
        exec_spec = self.specs.execution
        termination = getattr(exec_spec, "termination", None)

        # min_nodes_before_stop: don't terminate early if too few nodes explored
        min_nodes = getattr(termination, "min_nodes_before_stop", 10) if termination else 10
        if len(self.all_nodes) < min_nodes:
            return False

        # Max steps
        max_steps = getattr(termination, "max_steps", None) if termination else None
        if max_steps is None:
            max_steps = getattr(exec_spec.search, "max_nodes", 100)
        if self.step >= max_steps:
            logger.info("Terminating: max steps %d reached", max_steps)
            return True

        # Max nodes
        max_nodes = getattr(exec_spec.search, "max_nodes", 100)
        if len(self.all_nodes) >= max_nodes:
            logger.info("Terminating: max nodes %d reached", max_nodes)
            return True

        # Max wall time
        max_wall = getattr(termination, "max_wall_time_hours", None) if termination else None
        if max_wall is not None:
            elapsed_hours = (time.monotonic() - self._start_time) / 3600.0
            if elapsed_hours >= max_wall:
                logger.info("Terminating: wall time %.1fh >= %.1fh", elapsed_hours, max_wall)
                return True

        # Plateau detection
        stop_on_plateau = getattr(termination, "stop_on_plateau", True) if termination else True
        if stop_on_plateau:
            patience = getattr(termination, "plateau_patience", 10) if termination else 10
            min_improv = getattr(termination, "plateau_min_improvement", 0.001) if termination else 0.001
            current_best = self.best_node.lcb if self.best_node and self.best_node.lcb is not None else None
            if current_best is not None:
                if self._best_lcb_for_plateau is None or (current_best - self._best_lcb_for_plateau) >= min_improv:
                    self._best_lcb_for_plateau = current_best
                    self._plateau_counter = 0
                else:
                    self._plateau_counter += 1
                if self._plateau_counter >= patience:
                    logger.info("Terminating: plateau for %d steps", self._plateau_counter)
                    return True

        # Budget exceeded check (exit code 12)
        pruning_cfg = getattr(exec_spec, "pruning", None)
        budget_cfg = getattr(pruning_cfg, "budget_limit", None) if pruning_cfg else None
        budget_limit = getattr(budget_cfg, "limit", None) if budget_cfg else None
        if budget_limit is not None:
            total_cost = sum(n.total_cost for n in self.all_nodes.values())
            if total_cost > budget_limit:
                logger.info("Terminating: budget exceeded (%.1f > %.1f)", total_cost, budget_limit)
                self._budget_exceeded = True
                return True

        # No open nodes: check if there's still work to do
        if not self.open_list and self.step > 0:
            pending = sum(1 for n in self.all_nodes.values() if n.status == "pending")
            debuggable = sum(
                1 for n in self.all_nodes.values() if n.status == "failed" and n.node_id not in self.closed_set
            )
            # Can still draft or improve if under max_nodes
            can_expand = len(self.all_nodes) < max_nodes
            if pending == 0 and debuggable == 0 and not can_expand:
                logger.info("Terminating: no more nodes to process")
                return True

        return False

    def _update_best(self, node: SearchNode) -> None:
        """Update best_node if node has higher LCB (with secondary metric tiebreaker)."""
        if node.lcb is None or not node.feasible:
            return
        if self.best_node is None or self.best_node.lcb is None:
            self.best_node = node
        elif node.lcb > self.best_node.lcb:
            self.best_node = node
        else:
            # When LCBs are close (within plateau_min_improvement), compare secondary metrics
            plateau_min = getattr(getattr(self.specs.execution, "termination", None), "plateau_min_improvement", 0.001)
            if abs(node.lcb - self.best_node.lcb) < plateau_min:
                # Try secondary metrics tiebreaker
                secondary = getattr(self.specs, "problem", None)
                secondary_metrics = getattr(secondary, "secondary_metrics", []) if secondary else []
                if secondary_metrics and node.metrics_raw and self.best_node.metrics_raw:
                    # Weighted sum tiebreak: compute a single score from all secondary metrics
                    node_score = 0.0
                    best_score = 0.0
                    has_any = False
                    for sm in secondary_metrics:
                        sm_name = getattr(sm, "name", "")
                        sm_dir = getattr(sm, "direction", "maximize")
                        sm_weight = getattr(sm, "weight_in_tiebreak", 1.0)
                        node_val = (
                            node.metrics_raw[-1].get(sm_name)
                            if isinstance(node.metrics_raw[-1], dict)
                            else None
                        )
                        best_val = (
                            self.best_node.metrics_raw[-1].get(sm_name)
                            if isinstance(self.best_node.metrics_raw[-1], dict)
                            else None
                        )
                        if node_val is not None and best_val is not None:
                            has_any = True
                            sign = 1.0 if sm_dir == "maximize" else -1.0
                            node_score += sm_weight * sign * float(node_val)
                            best_score += sm_weight * sign * float(best_val)
                    if has_any and node_score > best_score:
                        self.best_node = node
                elif (node.mu or 0) > (self.best_node.mu or 0):
                    self.best_node = node
                else:
                    return
            else:
                return
        logger.info(
            "New best node: %s (LCB=%.4f, mu=%.4f)",
            node.node_id[:8],
            node.lcb,
            node.mu or 0.0,
        )

        # Notify PPO trainer of current best LCB for plateau detection
        if self.ppo_trainer is not None and hasattr(self.ppo_trainer, "notify_step"):
            best_lcb = self.best_node.lcb if self.best_node and self.best_node.lcb is not None else None
            if best_lcb is not None:
                self.ppo_trainer.notify_step(best_lcb)

    def _is_topk(self, node: SearchNode, k: int) -> bool:
        """Check if node is in top-k by LCB among all evaluated/running nodes."""
        if node.lcb is None:
            return True  # unevaluated nodes always get full eval

        candidates = [
            n
            for n in self.all_nodes.values()
            if n.lcb is not None and n.feasible and n.status in ("evaluated", "running")
        ]
        # Sort by LCB desc, then by mu desc as tiebreaker
        candidates.sort(
            key=lambda n: (n.lcb, n.mu if n.mu is not None else float("-inf")),
            reverse=True,
        )

        top_k_ids = {n.node_id for n in candidates[:k]}
        return node.node_id in top_k_ids

    def _build_tool_trajectory(self, loop_result: Any) -> list[dict]:
        """Build tool trajectory from AgentLoopResult for PPORolloutV3."""
        trajectory = []
        if loop_result is None or not hasattr(loop_result, "turns"):
            return trajectory
        for turn in loop_result.turns:
            for tr in turn.tool_results or []:
                trajectory.append(
                    {
                        "tool_name": tr.tool_name,
                        "success": tr.success,
                        "wall_time_sec": tr.wall_time_sec,
                        "call_id": getattr(tr, "call_id", ""),
                    }
                )
        return trajectory

    async def _run_post_step_tasks(self, exec_spec: Any) -> None:
        """Run PPO update, pruning, and checkpoint steps shared by single and batch paths."""
        # PPO update
        if self.ppo_trainer is not None and self.ppo_buffer:
            evaluated_list = [n for n in self.all_nodes.values() if n.status == "evaluated"]
            n_evaluated = len(evaluated_list)
            if self.ppo_trainer.should_update(n_evaluated, evaluated_list):
                try:
                    logger.info(
                        "PPO update triggered (buffer size=%d, n_evaluated=%d)",
                        len(self.ppo_buffer),
                        n_evaluated,
                    )
                    rollouts = []
                    for entry in self.ppo_buffer:
                        entry_turn_rewards = entry.get("turn_rewards", {})
                        if entry.get("tool_trajectory"):
                            rollouts.append(
                                PPORolloutV3(
                                    node_id=entry["node_id"],
                                    prompt=entry.get("hypothesis", ""),
                                    response=str(entry.get("config", {})),
                                    log_prob=0.0,
                                    reward=entry.get("reward", 0.0),
                                    value=0.0,
                                    turn_rewards=entry.get("turn_rewards", {}),
                                    tool_trajectory=entry.get("tool_trajectory", []),
                                    total_tool_calls=entry.get("total_tool_calls", 0),
                                    tool_success_rate=entry.get("tool_success_rate", 0.0),
                                )
                            )
                        elif entry_turn_rewards:
                            rollouts.append(
                                PPORolloutV2(
                                    node_id=entry["node_id"],
                                    prompt=entry.get("hypothesis", ""),
                                    response=str(entry.get("config", {})),
                                    log_prob=0.0,
                                    reward=entry.get("reward", 0.0),
                                    value=0.0,
                                    turn_rewards=entry_turn_rewards,
                                )
                            )
                        else:
                            rollouts.append(
                                PPORollout(
                                    node_id=entry["node_id"],
                                    prompt=entry.get("hypothesis", ""),
                                    response=str(entry.get("config", {})),
                                    log_prob=0.0,
                                    reward=entry.get("reward", 0.0),
                                    value=0.0,
                                )
                            )
                    ppo_result = await self.ppo_trainer.update(rollouts, self.agent_llm, self.specs, all_nodes=self.all_nodes)
                    new_adapter_id = ppo_result.get("new_adapter_node_id") if isinstance(ppo_result, dict) else None
                    if new_adapter_id:
                        for entry in self.ppo_buffer:
                            nid = entry["node_id"]
                            if nid in self.all_nodes:
                                sync_adapter_assignment(self.all_nodes[nid], True, new_adapter_id, self.all_nodes)
                        logger.info("Assigned new adapter %s to %d nodes", new_adapter_id[:8], len(self.ppo_buffer))
                    self.ppo_buffer.clear()
                except Exception as e:
                    logger.error("PPO update failed: %s", e)

        # Prune periodically
        prune_interval = (
            getattr(getattr(exec_spec, "pruning", None), "prune_interval", 10)
            if getattr(exec_spec, "pruning", None)
            else 10
        )
        if self.pruner is not None and self.step % prune_interval == 0:
            try:
                open_nodes = [
                    self.all_nodes[nid]
                    for _, nid in self.open_list
                    if nid in self.all_nodes and nid not in self.closed_set
                ]
                workspace_dir = self.checkpoint_dir.parent if self.checkpoint_dir else None
                pruned_ids = self.pruner.prune(
                    open_nodes, self.closed_set, self.all_nodes, exec_spec,
                    workspace_dir=workspace_dir,
                )
                if pruned_ids:
                    pruned_set = set(pruned_ids)
                    self.open_list = [(p, nid) for p, nid in self.open_list if nid not in pruned_set]
                    self.closed_set.update(pruned_set)
            except Exception as e:
                logger.error("Pruning failed: %s", e)

        # Checkpoint periodically
        plan_spec = getattr(self.specs, "plan", None)
        logging_cfg = getattr(plan_spec, "logging", None) if plan_spec else None
        checkpoint_interval = getattr(logging_cfg, "checkpoint_interval", 10) if logging_cfg else 10
        if self.step % checkpoint_interval == 0:
            try:
                state = self.save_state()
                save_checkpoint(state, self.checkpoint_dir, self.step)
            except Exception as e:
                logger.error("Checkpoint save failed: %s", e)

    def _is_slurm_executor(self) -> bool:
        """Check if the executor is a SlurmExecutor instance."""
        try:
            from sera.execution.slurm_executor import SlurmExecutor
            return isinstance(self.executor, SlurmExecutor)
        except ImportError:
            return False

    async def _run_batched_pipeline(
        self,
        batch_size: int,
        max_concurrent: int,
        poll_interval_sec: float,
    ) -> None:
        """Submit multiple pending nodes in batch via SLURM, evaluate results, and feed PPO buffer.

        Collects up to ``batch_size`` pending nodes from the open list, submits them
        in a single batch via the executor's ``run_batch()`` method, evaluates the
        results, and processes PPO updates.

        Parameters
        ----------
        batch_size : int
            Maximum number of nodes to submit per batch.
        max_concurrent : int
            Maximum number of concurrent SLURM jobs (caps batch collection).
        poll_interval_sec : float
            Polling interval for SLURM job status (passed to executor).
        """
        # Collect pending nodes from the open list up to min(batch_size, max_concurrent)
        effective_batch = min(batch_size, max_concurrent)
        batch_nodes: list[SearchNode] = []

        while self.open_list and len(batch_nodes) < effective_batch:
            neg_priority, node_id = heapq.heappop(self.open_list)
            if node_id in self.closed_set or node_id not in self.all_nodes:
                continue
            node = self.all_nodes[node_id]
            if node.status != "pending":
                # Put non-pending nodes back for single-node processing
                heapq.heappush(self.open_list, (neg_priority, node_id))
                break
            self.closed_set.add(node_id)
            batch_nodes.append(node)

        if not batch_nodes:
            return

        logger.info("SLURM batch pipeline: submitting %d nodes", len(batch_nodes))

        # Build task dicts for run_batch
        exec_spec = self.specs.execution
        timeout_sec = getattr(getattr(exec_spec, "search", None), "timeout_sec", None)
        work_dir = self.executor.work_dir if hasattr(self.executor, "work_dir") else self.checkpoint_dir.parent

        tasks = []
        for node in batch_nodes:
            node.status = "running"
            run_dir = Path(work_dir) / "runs" / node.node_id
            script_path = run_dir / "experiment.py"
            tasks.append({
                "node_id": node.node_id,
                "script_path": script_path,
                "seed": 0,  # Seed will be derived by executor or evaluator
                "timeout_sec": timeout_sec,
            })

        # Submit batch and collect results
        try:
            results = await self.executor.run_batch(tasks)
        except Exception as e:
            logger.error("SLURM batch submission failed: %s", e)
            for node in batch_nodes:
                node.mark_failed(f"Batch submission failed: {e}")
                self.closed_set.discard(node.node_id)
            return

        # Process each result
        for node, result in zip(batch_nodes, results):
            self.step += 1
            try:
                if not result.success:
                    if result.exit_code == -7:
                        node.status = "oom"
                    elif result.exit_code == -9:
                        node.status = "timeout"
                    else:
                        node.mark_failed(f"exit_code={result.exit_code}")
                    self.closed_set.discard(node.node_id)
                    continue

                # Run evaluator on the successful result
                await self.evaluator.evaluate_initial(node)

                if node.status in ("failed", "oom"):
                    self.closed_set.discard(node.node_id)
                    continue

                k = getattr(getattr(exec_spec, "evaluation", None), "sequential_eval_topk", 5)
                if self._is_topk(node, k):
                    await self.evaluator.evaluate_full(node)

                node.mark_evaluated()
                self._update_best(node)

                # Re-queue for improve
                priority = compute_priority(node, exec_spec)
                node.priority = priority
                self.closed_set.discard(node.node_id)
                heapq.heappush(self.open_list, (-priority, node.node_id))

                # Add to PPO buffer
                if node.mu is not None:
                    entry: dict[str, Any] = {
                        "node_id": node.node_id,
                        "hypothesis": node.hypothesis,
                        "config": node.experiment_config,
                        "reward": node.mu,
                        "feasible": node.feasible,
                    }
                    self.ppo_buffer.append(entry)

            except Exception as e:
                node.mark_failed(str(e))
                self.closed_set.discard(node.node_id)
                logger.error("Batch evaluation failed for node %s: %s", node.node_id[:8], e)

            # Log each node
            if self.logger:
                self.logger.log({
                    "event": "node_selected",
                    "step": self.step,
                    "node_id": node.node_id,
                    "operator": "evaluate_batch",
                    "chosen_reason": "evaluate_batch",
                    "status": node.status,
                    "mu": node.mu,
                    "se": node.se,
                    "lcb": node.lcb,
                    "total_nodes": len(self.all_nodes),
                })

        logger.info(
            "SLURM batch pipeline complete: %d/%d nodes evaluated",
            sum(1 for n in batch_nodes if n.status == "evaluated"),
            len(batch_nodes),
        )

    def _add_node(self, node: SearchNode) -> None:
        """Add a node to the search tree and open list."""
        # Sync adapter_node_id with lineage tree (section 9.1.1)
        sync_adapter_assignment(node, False, None, self.all_nodes)

        self.all_nodes[node.node_id] = node
        priority = compute_priority(node, self.specs.execution)
        node.priority = priority
        # heapq is a min-heap, so negate priority for max-priority-first
        heapq.heappush(self.open_list, (-priority, node.node_id))

    def save_state(self) -> dict:
        """Serialize state for checkpoint."""
        return {
            "step": self.step,
            "all_nodes": {nid: node.to_dict() for nid, node in self.all_nodes.items()},
            "closed_set": list(self.closed_set),
            "best_node_id": (self.best_node.node_id if self.best_node else None),
            "open_list": list(self.open_list),
            "ppo_buffer": self.ppo_buffer,
        }

    def load_state(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.step = state.get("step", 0)
        self.all_nodes = {nid: SearchNode.from_dict(d) for nid, d in state.get("all_nodes", {}).items()}
        self.closed_set = set(state.get("closed_set", []))
        self.open_list = state.get("open_list", [])
        self.ppo_buffer = state.get("ppo_buffer", [])

        best_id = state.get("best_node_id")
        if best_id and best_id in self.all_nodes:
            self.best_node = self.all_nodes[best_id]
        else:
            self.best_node = None
