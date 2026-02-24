"""Evaluator ABC per section 8.

Defines the interface for experiment evaluation, with two-phase
evaluation: initial (quick) and full (thorough).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Evaluator(ABC):
    """Abstract base class for experiment evaluators.

    The evaluation protocol follows a two-phase approach:
    1. ``evaluate_initial``: Run a small number of seeds to get a quick
       estimate. This is done for every node.
    2. ``evaluate_full``: Run the remaining seeds to get a statistically
       robust estimate. Only done for top-k nodes.
    """

    @abstractmethod
    async def evaluate_initial(self, node: Any) -> None:
        """Run the initial (quick) evaluation on a node.

        Should run ``sequential_eval_initial`` seeds and update
        the node's metrics_raw, mu, se, and lcb.

        Parameters
        ----------
        node : SearchNode
            The node to evaluate. Modified in place.
        """
        ...

    @abstractmethod
    async def evaluate_full(self, node: Any) -> None:
        """Run the full evaluation on a node.

        Should run the remaining seeds (up to total repeats) and
        update the node's statistics.

        Parameters
        ----------
        node : SearchNode
            The node to fully evaluate. Modified in place.
        """
        ...
