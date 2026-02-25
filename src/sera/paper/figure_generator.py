"""FigureGenerator per S11.3 - creates publication-quality figures from evidence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Force non-interactive backend before any pyplot import
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_MAX_FIGURES = 12
_DPI = 300


class FigureGenerator:
    """Generates publication-quality figures from experimental evidence.

    All figures are saved as PNG at 300 DPI into ``output_dir``.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._figure_count = 0

    def _check_limit(self) -> bool:
        """Return True if we can still produce more figures."""
        return self._figure_count < _MAX_FIGURES

    def _save(self, fig: plt.Figure, name: str) -> Path:
        """Save *fig* and increment the counter."""
        out = self.output_dir / name
        fig.savefig(out, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        self._figure_count += 1
        return out

    # ------------------------------------------------------------------
    # CI bar chart
    # ------------------------------------------------------------------

    def ci_bar_chart(self, nodes: list[Any], output_name: str = "ci_bar_chart.png") -> Path:
        """Matplotlib errorbar chart of mu +/- CI per method.

        Parameters
        ----------
        nodes:
            List of SearchNode-like objects with ``mu``, ``se``,
            ``experiment_config``, and ``hypothesis`` attributes.
        output_name:
            File name for the output PNG.
        """
        if not self._check_limit():
            raise RuntimeError("Maximum figure count reached")

        methods: list[str] = []
        means: list[float] = []
        errors: list[float] = []

        for node in nodes:
            mu = node.mu if node.mu is not None else 0.0
            se = node.se if (node.se is not None and node.se != float("inf")) else 0.0
            label = node.experiment_config.get("method", node.hypothesis[:40])
            methods.append(label)
            means.append(mu)
            # 95% CI half-width ~ 1.96 * SE
            errors.append(1.96 * se)

        fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.2), 5))
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=errors, capsize=5, color="steelblue", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Metric (\u03bc \u00b1 95% CI)")
        ax.set_title("Method Comparison")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return self._save(fig, output_name)

    # ------------------------------------------------------------------
    # Convergence curve
    # ------------------------------------------------------------------

    def convergence_curve(
        self,
        data: list[tuple[int, float]],
        output_name: str = "convergence_curve.png",
    ) -> Path:
        """Step vs best_lcb line plot.

        Parameters
        ----------
        data:
            List of ``(step, best_lcb)`` tuples.
        """
        if not self._check_limit():
            raise RuntimeError("Maximum figure count reached")

        if not data:
            # Produce an empty plot with a note
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No convergence data", transform=ax.transAxes, ha="center", va="center", fontsize=14)
            return self._save(fig, output_name)

        steps, lcbs = zip(*data)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, lcbs, marker="o", markersize=3, linewidth=1.5, color="darkgreen")
        ax.set_xlabel("Search Step")
        ax.set_ylabel("Best LCB")
        ax.set_title("Convergence of Best LCB over Search Steps")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, output_name)

    # ------------------------------------------------------------------
    # Search tree
    # ------------------------------------------------------------------

    def search_tree(
        self,
        nodes: list[Any],
        top_n: int = 10,
        output_name: str = "search_tree.png",
    ) -> Path | None:
        """Graphviz tree visualisation of the search tree.

        Falls back gracefully if graphviz is not installed.

        Parameters
        ----------
        nodes:
            List of SearchNode-like objects with ``node_id``, ``parent_id``,
            ``hypothesis``, ``mu``, ``status``.
        top_n:
            Number of top nodes to highlight.
        """
        if not self._check_limit():
            raise RuntimeError("Maximum figure count reached")

        try:
            import graphviz  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("graphviz Python package not installed; skipping search_tree figure.")
            return None

        dot = graphviz.Digraph(format="png")
        dot.attr(rankdir="TB", fontsize="10")

        # Sort by LCB to find the top-N
        sorted_nodes = sorted(
            nodes,
            key=lambda n: n.lcb if n.lcb is not None else float("-inf"),
            reverse=True,
        )
        top_ids = {n.node_id for n in sorted_nodes[:top_n]}

        for node in nodes:
            short_id = node.node_id[:8]
            mu_str = f"{node.mu:.3f}" if node.mu is not None else "?"
            label = f"{short_id}\\n{node.hypothesis[:30]}\\nmu={mu_str}"
            color = "gold" if node.node_id in top_ids else "lightblue"
            if node.status == "failed":
                color = "lightcoral"
            dot.node(node.node_id, label=label, style="filled", fillcolor=color)
            if node.parent_id:
                dot.edge(node.parent_id, node.node_id)

        out_path = self.output_dir / output_name
        # graphviz render returns the path without the extension it adds
        try:
            rendered = dot.render(
                filename=str(out_path.with_suffix("")),
                cleanup=True,
            )
            self._figure_count += 1
            return Path(rendered)
        except Exception as exc:
            logger.warning("graphviz rendering failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Ablation table
    # ------------------------------------------------------------------

    def ablation_table(
        self,
        data: dict[str, dict],
        output_name: str = "ablation_table.png",
    ) -> Path:
        """Heatmap / table image of ablation results.

        Parameters
        ----------
        data:
            Dict mapping variable name to ``{mu, se, lcb, config}``.
        """
        if not self._check_limit():
            raise RuntimeError("Maximum figure count reached")

        if not data:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No ablation data", transform=ax.transAxes, ha="center", va="center", fontsize=14)
            ax.axis("off")
            return self._save(fig, output_name)

        variables = list(data.keys())
        mus = [data[v].get("mu", 0) or 0 for v in variables]
        lcbs = [data[v].get("lcb", 0) or 0 for v in variables]

        fig, ax = plt.subplots(figsize=(max(6, len(variables) * 1.5), 4))
        x = np.arange(len(variables))
        width = 0.35
        ax.bar(x - width / 2, mus, width, label="\u03bc", color="steelblue")
        ax.bar(x + width / 2, lcbs, width, label="LCB", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title("Ablation Study")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return self._save(fig, output_name)

    # ------------------------------------------------------------------
    # Aggregate plots (LLM-driven)
    # ------------------------------------------------------------------

    async def aggregate_plots(
        self,
        evidence: Any,
        agent_llm: Any,
        n_reflections: int = 2,
    ) -> list[Path]:
        """LLM-generated aggregation plots.

        Asks the LLM to propose matplotlib code for additional figures,
        executes the code in a restricted scope, and collects the output
        PNG paths.

        Parameters
        ----------
        evidence:
            An EvidenceStore instance.
        agent_llm:
            An AgentLLM instance for generating code.
        n_reflections:
            Number of reflection rounds for improving proposed code.
        """
        paths: list[Path] = []
        if agent_llm is None:
            return paths

        summaries = evidence.get_experiment_summaries()
        convergence = evidence.get_convergence_data()

        import json as _json

        context = _json.dumps(
            {
                "summaries": summaries,
                "convergence_points": len(convergence),
                "num_nodes": len(evidence.all_evaluated_nodes),
            },
            default=str,
        )

        prompt = (
            "You are a scientific visualization expert. Given the following "
            "experimental evidence summary, propose up to 3 additional matplotlib "
            "plots that would be informative for a research paper. For each plot, "
            "provide Python code that uses matplotlib with Agg backend, creates a "
            "figure, and calls fig.savefig(output_path, dpi=300, bbox_inches='tight').\n\n"
            "Return ONLY a JSON list of objects with keys: "
            "'description' (str), 'code' (str with {output_path} placeholder).\n\n"
            f"Evidence:\n{context}"
        )

        code_response = await agent_llm.generate(prompt=prompt, purpose="aggregate_plot_generation")

        # Try to parse the LLM response as JSON
        try:
            # Extract JSON from potential markdown fencing
            text = code_response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            plot_specs = _json.loads(text)
        except (_json.JSONDecodeError, IndexError):
            logger.warning("Could not parse LLM aggregate plot response")
            return paths

        if not isinstance(plot_specs, list):
            return paths

        for idx, spec in enumerate(plot_specs[:3]):
            if not self._check_limit():
                break
            code = spec.get("code", "")
            if not code:
                continue

            out_name = f"aggregate_plot_{idx}.png"
            out_path = self.output_dir / out_name
            code = code.replace("{output_path}", f"'{out_path}'")

            # Reflection loop
            for _r in range(n_reflections):
                try:
                    exec_globals = {"plt": plt, "np": np, "matplotlib": matplotlib}
                    exec(code, exec_globals)  # noqa: S102
                    if out_path.exists():
                        paths.append(out_path)
                        self._figure_count += 1
                        break
                except Exception as exc:
                    logger.warning("Aggregate plot %d failed: %s", idx, exc)
                    # Ask LLM to fix
                    fix_prompt = (
                        f"The following matplotlib code raised an error:\n"
                        f"```python\n{code}\n```\n"
                        f"Error: {exc}\n"
                        f"Fix the code and return ONLY the corrected Python code."
                    )
                    fixed = await agent_llm.generate(prompt=fix_prompt, purpose="aggregate_plot_fix")
                    # Extract code from response
                    if "```python" in fixed:
                        code = fixed.split("```python")[1].split("```")[0].strip()
                    elif "```" in fixed:
                        code = fixed.split("```")[1].split("```")[0].strip()
                    else:
                        code = fixed.strip()
                    code = code.replace("{output_path}", f"'{out_path}'")

        plt.close("all")
        return paths
