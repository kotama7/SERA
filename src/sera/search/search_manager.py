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
from pathlib import Path
from typing import Any

from sera.learning.rollout import PPORollout, PPORolloutV2
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

        self._setup_signal_handler()

    def _setup_signal_handler(self) -> None:
        """SIGINT handler: save checkpoint then exit(20)."""

        def _handler(signum: int, frame: Any) -> None:
            logger.warning("SIGINT received, saving checkpoint and exiting")
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

        # Step 1: Draft initial children
        n_initial = getattr(exec_spec.search, "initial_root_children", 5)
        initial_nodes = await self.tree_ops.draft(n_initial, self.all_nodes)
        for node in initial_nodes:
            self._add_node(node)

        # Step 2: Main loop
        while not self._should_terminate():
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
                                n for n in self.all_nodes.values()
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
                    new_nodes = await self.tree_ops.draft(
                        getattr(exec_spec.search, "branch_factor", 3),
                        self.all_nodes,
                    )
                    for n in new_nodes:
                        self._add_node(n)
                elif operator == "improve":
                    n_children = getattr(exec_spec.search, "branch_factor", 3)
                    children = await self.tree_ops.improve(
                        node, self.all_nodes, n_children
                    )
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
                        "event": "node_processed",
                        "step": self.step,
                        "node_id": node.node_id,
                        "parent_id": node.parent_id,
                        "depth": node.depth,
                        "operator": operator,
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
                            "wall_time_sec": sum(
                                n.wall_time_sec for n in self.all_nodes.values()
                            ),
                        },
                    }
                )

            # Step 3: PPO update when enough evaluated nodes (skipped if ppo_trainer is None)
            if self.ppo_trainer is not None and self.ppo_buffer:
                n_evaluated = sum(
                    1 for n in self.all_nodes.values() if n.status == "evaluated"
                )
                if self.ppo_trainer.should_update(n_evaluated):
                    try:
                        logger.info("PPO update triggered (buffer size=%d, n_evaluated=%d)", len(self.ppo_buffer), n_evaluated)
                        rollouts = []
                        for entry in self.ppo_buffer:
                            entry_turn_rewards = entry.get("turn_rewards", {})
                            if entry_turn_rewards:
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
                        ppo_result = await self.ppo_trainer.update(rollouts, self.agent_llm, self.specs)

                        # If PPO produced a new adapter, update nodes' adapter_node_ids
                        new_adapter_id = ppo_result.get("new_adapter_node_id") if isinstance(ppo_result, dict) else None
                        if new_adapter_id:
                            for entry in self.ppo_buffer:
                                nid = entry["node_id"]
                                if nid in self.all_nodes:
                                    sync_adapter_assignment(
                                        self.all_nodes[nid], True, new_adapter_id, self.all_nodes
                                    )
                            logger.info("Assigned new adapter %s to %d nodes", new_adapter_id[:8], len(self.ppo_buffer))

                        self.ppo_buffer.clear()
                    except Exception as e:
                        logger.error("PPO update failed: %s", e)

            # Step 4: Prune periodically
            prune_interval = getattr(
                getattr(exec_spec, "pruning", None), "prune_interval", 10
            ) if getattr(exec_spec, "pruning", None) else 10
            if self.pruner is not None and self.step % prune_interval == 0:
                try:
                    open_nodes = [
                        self.all_nodes[nid]
                        for _, nid in self.open_list
                        if nid in self.all_nodes and nid not in self.closed_set
                    ]
                    pruned_ids = self.pruner.prune(open_nodes, self.closed_set, self.all_nodes, exec_spec)
                    if pruned_ids:
                        self.open_list = [
                            (p, nid) for p, nid in self.open_list if nid not in set(pruned_ids)
                        ]
                except Exception as e:
                    logger.error("Pruning failed: %s", e)

            # Step 5: Checkpoint periodically
            checkpoint_interval = 10
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
                logger.info(
                    "Node %s %s during evaluation", node.node_id[:8], node.status
                )
                self.closed_set.discard(node.node_id)  # Allow debug operator to pick it up
                return

            # Check if in top-k for full eval
            k = getattr(self.specs.execution.search, "sequential_eval_topk", 5)
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
                    turn_rewards = self.turn_reward_evaluator.evaluate_all(
                        node, parent_node, self.all_nodes
                    )
                except Exception as e:
                    logger.warning("Turn reward evaluation failed for node %s: %s", node.node_id[:8], e)

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

        - pending -> evaluate
        - failed (debug_depth < max) -> debug
        - evaluated + low diversity -> draft
        - evaluated -> improve

        Returns (node, operator) or (None, "") if no nodes available.
        """
        exec_spec = self.specs.execution
        max_debug_depth = getattr(getattr(exec_spec, "search", None), "max_debug_depth", 3)

        # Check for failed nodes that can be debugged (oom nodes have status="oom", not "failed")
        for node in self.all_nodes.values():
            if (
                node.status == "failed"
                and node.debug_depth < max_debug_depth
                and node.node_id not in self.closed_set
            ):
                self.closed_set.add(node.node_id)
                return node, "debug"

        # Step 3: Diversity check for draft re-trigger (TASK.md section 6.4)
        evaluated = [n for n in self.all_nodes.values() if n.status == "evaluated"]
        unique_methods = len({n.experiment_config.get("method", "") for n in evaluated})
        min_diverse = getattr(exec_spec.search, "min_diverse_methods", 3)
        draft_after = getattr(exec_spec.search, "draft_trigger_after", 10)
        if len(evaluated) >= draft_after and unique_methods < min_diverse:
            max_nodes = getattr(exec_spec.search, "max_nodes", 100)
            if len(self.all_nodes) < max_nodes:
                return (self.best_node or SearchNode()), "draft"

        # Pop from priority queue
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
                return node, "improve"

        # No nodes in open list; check if we need more diversity
        evaluated_count = sum(
            1
            for n in self.all_nodes.values()
            if n.status == "evaluated"
        )
        max_nodes = getattr(exec_spec.search, "max_nodes", 100)
        if len(self.all_nodes) < max_nodes:
            # Return a sentinel to trigger drafting
            return (self.best_node or SearchNode()), "draft"

        return None, ""

    def _should_terminate(self) -> bool:
        """Check all termination conditions from ExecutionSpec.termination."""
        exec_spec = self.specs.execution
        termination = getattr(exec_spec, "termination", None)

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

        # No open nodes: check if there's still work to do
        if not self.open_list and self.step > 0:
            pending = sum(
                1 for n in self.all_nodes.values() if n.status == "pending"
            )
            debuggable = sum(
                1
                for n in self.all_nodes.values()
                if n.status == "failed" and n.node_id not in self.closed_set
            )
            # Can still draft or improve if under max_nodes
            can_expand = len(self.all_nodes) < max_nodes
            if pending == 0 and debuggable == 0 and not can_expand:
                logger.info("Terminating: no more nodes to process")
                return True

        return False

    def _update_best(self, node: SearchNode) -> None:
        """Update best_node if node has higher LCB (with mu as tiebreaker)."""
        if node.lcb is None or not node.feasible:
            return
        if self.best_node is None or self.best_node.lcb is None:
            self.best_node = node
        elif node.lcb > self.best_node.lcb:
            self.best_node = node
        elif node.lcb == self.best_node.lcb and (node.mu or 0) > (self.best_node.mu or 0):
            self.best_node = node
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
            "all_nodes": {
                nid: node.to_dict() for nid, node in self.all_nodes.items()
            },
            "closed_set": list(self.closed_set),
            "best_node_id": (
                self.best_node.node_id if self.best_node else None
            ),
            "open_list": list(self.open_list),
            "ppo_buffer": self.ppo_buffer,
        }

    def load_state(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.step = state.get("step", 0)
        self.all_nodes = {
            nid: SearchNode.from_dict(d)
            for nid, d in state.get("all_nodes", {}).items()
        }
        self.closed_set = set(state.get("closed_set", []))
        self.open_list = state.get("open_list", [])
        self.ppo_buffer = state.get("ppo_buffer", [])

        best_id = state.get("best_node_id")
        if best_id and best_id in self.all_nodes:
            self.best_node = self.all_nodes[best_id]
        else:
            self.best_node = None
