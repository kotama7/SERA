"""sera visualize command implementation."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def run_visualize(
    work_dir: str,
    step: int | None = None,
    output: str | None = None,
) -> None:
    """Generate an interactive HTML visualization of the search tree.

    Parameters
    ----------
    work_dir : str
        Path to the sera workspace.
    step : int | None
        Specific checkpoint step to visualize. ``None`` uses the latest.
    output : str | None
        Output path for the HTML file. ``None`` uses default.
    """
    workspace = Path(work_dir)

    from sera.visualization.tree_visualizer import TreeVisualizer

    visualizer = TreeVisualizer(workspace)

    output_path = Path(output) if output else None

    try:
        result_path = visualizer.generate_html(step=step, output_path=output_path)
        console.print(f"[green]Visualization generated: {result_path}[/green]")
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure you have run 'sera research' first to create checkpoints.[/yellow]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Visualization failed: {e}[/red]")
        raise SystemExit(1)
