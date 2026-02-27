"""Pruner for the search tree per section 10.4.

Implements several pruning strategies that operate on the search tree to
remove unpromising branches and free resources:

1. **Protection list** -- best node + ancestors, top-k + ancestors, and
   running nodes are never pruned.
2. **Pareto pruning** -- nodes dominated in the (primary LCB, cost) space.
3. **LCB threshold pruning** -- nodes whose LCB falls below
   ``best_lcb * threshold_fraction``.
4. **Budget pruning** -- when total cost exceeds a limit, the worst-LCB
   nodes are removed first.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Pruner:
    """Search-tree pruner.

    All public methods are stateless -- the pruner reads configuration
    from the execution spec passed at call time.
    """

    def prune(
        self,
        open_list: list[Any],
        closed_set: set[str],
        all_nodes: dict[str, Any],
        exec_spec: Any,
        workspace_dir: Path | None = None,
    ) -> list[Any]:
        """Run all pruning passes and return pruned nodes.

        The pruned nodes have their ``status`` set to ``"pruned"``.

        Parameters
        ----------
        open_list : list[SearchNode]
            Nodes still eligible for expansion.
        closed_set : set[str]
            Node ids that are already closed (evaluated, etc).
        all_nodes : dict[str, SearchNode]
            Mapping from ``node_id`` to ``SearchNode`` for the whole tree.
        exec_spec : ExecutionSpecModel
            Execution spec for pruning hyper-parameters.

        Returns
        -------
        list[SearchNode]
            The pruned ``SearchNode`` objects.
        """
        # Build protection set
        protected = self._build_protection_set(all_nodes, exec_spec)

        pruned_ids: list[str] = []

        # 1. Pareto pruning (conditional on pruning.pareto flag)
        pruning_cfg = getattr(exec_spec, "pruning", None)
        pareto_enabled = getattr(pruning_cfg, "pareto", True) if pruning_cfg else True
        if pareto_enabled:
            pruned_ids.extend(self._pareto_prune(open_list, all_nodes, exec_spec, protected))

        # 2. LCB threshold pruning
        pruned_ids.extend(self._lcb_threshold_prune(open_list, all_nodes, exec_spec, protected))

        # 3. Budget pruning
        pruned_ids.extend(self._budget_prune(open_list, all_nodes, exec_spec, protected))

        # Deduplicate
        pruned_ids = list(dict.fromkeys(pruned_ids))

        # Mark nodes
        for nid in pruned_ids:
            if nid in all_nodes:
                all_nodes[nid].status = "pruned"

        # Remove pruned nodes from open_list in-place
        pruned_set = set(pruned_ids)
        open_list[:] = [n for n in open_list if n.node_id not in pruned_set]

        # Clean up run directories if save_pruned is disabled
        save_pruned = getattr(pruning_cfg, "save_pruned", True) if pruning_cfg else True
        if not save_pruned and workspace_dir is not None:
            runs_dir = workspace_dir / "runs"
            for nid in pruned_ids:
                run_dir = runs_dir / nid
                if run_dir.exists():
                    try:
                        shutil.rmtree(run_dir)
                        logger.info("Cleaned up run directory for pruned node %s", nid)
                    except Exception as e:
                        logger.warning("Failed to clean up run dir for %s: %s", nid, e)

        if pruned_ids:
            logger.info("Pruned %d nodes: %s", len(pruned_ids), pruned_ids)

        return [all_nodes[nid] for nid in pruned_ids if nid in all_nodes]

    # ------------------------------------------------------------------
    # Protection list
    # ------------------------------------------------------------------

    def _build_protection_set(
        self,
        all_nodes: dict[str, Any],
        exec_spec: Any,
    ) -> set[str]:
        """Build the set of node ids that must not be pruned.

        Protected nodes include:
        - The best node (highest LCB) and all its ancestors.
        - The top-k nodes and all their ancestors.
        - All currently running nodes.
        """
        protected: set[str] = set()

        evaluated = [n for n in all_nodes.values() if n.lcb is not None and n.status not in ("pruned", "failed")]

        if not evaluated:
            # Protect all running nodes
            for n in all_nodes.values():
                if n.status == "running":
                    protected.add(n.node_id)
            return protected

        # Sort by LCB descending
        evaluated.sort(key=lambda n: n.lcb, reverse=True)

        # Best node + ancestors
        best = evaluated[0]
        protected |= self._ancestors(best.node_id, all_nodes)

        # Top-k
        pruning_cfg = getattr(exec_spec, "pruning", None)
        keep_topk = getattr(pruning_cfg, "keep_topk", None)
        if keep_topk is None:
            keep_topk = getattr(pruning_cfg, "keep_top_k", 5) if pruning_cfg else 5
        for node in evaluated[:keep_topk]:
            protected |= self._ancestors(node.node_id, all_nodes)

        # Running nodes
        for n in all_nodes.values():
            if n.status == "running":
                protected.add(n.node_id)

        return protected

    def _ancestors(self, node_id: str, all_nodes: dict[str, Any]) -> set[str]:
        """Return *node_id* and all its ancestors."""
        result: set[str] = set()
        current = node_id
        visited: set[str] = set()
        while current is not None and current not in visited:
            visited.add(current)
            result.add(current)
            node = all_nodes.get(current)
            if node is None:
                break
            current = node.parent_id
        return result

    # ------------------------------------------------------------------
    # LCB threshold pruning
    # ------------------------------------------------------------------

    def _lcb_threshold_prune(
        self,
        open_list: list[Any],
        all_nodes: dict[str, Any],
        exec_spec: Any,
        protected: set[str],
    ) -> list[str]:
        """Prune nodes whose LCB is below ``best_lcb * threshold_fraction``.

        If ``pruning.reward_threshold`` is set to a non-zero value it is
        used directly.  Otherwise an "auto" threshold of
        ``best_lcb * 0.5`` is computed.
        """
        evaluated = [n for n in all_nodes.values() if n.lcb is not None and n.status not in ("pruned",)]
        if not evaluated:
            return []

        best_lcb = max(n.lcb for n in evaluated)

        pruning_cfg = getattr(exec_spec, "pruning", None)
        # Read lcb_threshold (fraction of best LCB) first, fall back to reward_threshold
        lcb_threshold_frac = getattr(pruning_cfg, "lcb_threshold", None) if pruning_cfg else None
        explicit_threshold = (
            pruning_cfg.reward_threshold if pruning_cfg and hasattr(pruning_cfg, "reward_threshold") else 0.0
        )

        if lcb_threshold_frac is not None:
            threshold = best_lcb * lcb_threshold_frac
        elif explicit_threshold != 0.0:
            threshold = explicit_threshold
        else:
            # Auto threshold: 50 % of best LCB
            threshold = best_lcb * 0.5

        pruned: list[str] = []
        for node in open_list:
            if node.node_id in protected:
                continue
            if node.lcb is not None and node.lcb < threshold:
                pruned.append(node.node_id)

        return pruned

    # ------------------------------------------------------------------
    # Pareto pruning
    # ------------------------------------------------------------------

    def _pareto_prune(
        self,
        open_list: list[Any],
        all_nodes: dict[str, Any],
        exec_spec: Any,
        protected: set[str],
    ) -> list[str]:
        """Remove nodes dominated in the (primary-LCB, cost) Pareto front.

        A node A *dominates* node B if A has higher (or equal) LCB **and**
        lower (or equal) cost, with at least one strict inequality.
        """
        candidates = [
            n for n in open_list if n.lcb is not None and n.node_id not in protected and n.status not in ("pruned",)
        ]

        if len(candidates) < 2:
            return []

        pruned: list[str] = []
        for i, a in enumerate(candidates):
            dominated = False
            for j, b in enumerate(candidates):
                if i == j:
                    continue
                if b.node_id in pruned:
                    continue
                # b dominates a?
                if b.lcb >= a.lcb and b.total_cost <= a.total_cost:
                    if b.lcb > a.lcb or b.total_cost < a.total_cost:
                        dominated = True
                        break
            if dominated:
                pruned.append(a.node_id)

        return pruned

    # ------------------------------------------------------------------
    # Budget pruning
    # ------------------------------------------------------------------

    def _budget_prune(
        self,
        open_list: list[Any],
        all_nodes: dict[str, Any],
        exec_spec: Any,
        protected: set[str],
    ) -> list[str]:
        """If total cost exceeds the budget limit, prune worst-LCB nodes.

        The budget limit is derived from
        ``termination.max_wallclock_hours * 3600`` or a default of
        14400 seconds.
        """
        term = getattr(exec_spec, "termination", None)
        if term and hasattr(term, "max_wall_time_hours") and term.max_wall_time_hours is not None:
            budget = term.max_wall_time_hours * 3600.0
        elif term and hasattr(term, "max_wallclock_hours") and term.max_wallclock_hours is not None:
            budget = term.max_wallclock_hours * 3600.0
        else:
            budget = 14400.0

        total_cost = sum(n.total_cost for n in all_nodes.values())
        if total_cost <= budget:
            return []

        # Sort open_list by LCB ascending (worst first)
        candidates = [n for n in open_list if n.node_id not in protected and n.status not in ("pruned",)]
        candidates.sort(key=lambda n: n.lcb if n.lcb is not None else float("-inf"))

        pruned: list[str] = []
        for node in candidates:
            if total_cost <= budget:
                break
            pruned.append(node.node_id)
            total_cost -= node.total_cost

        return pruned
