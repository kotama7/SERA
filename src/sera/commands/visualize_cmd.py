"""sera visualize command implementation."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

console = Console()


def run_visualize(
    work_dir: str,
    step: int | None = None,
    output: str | None = None,
    live: bool = False,
    port: int = 8080,
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
    live : bool
        If True, start an HTTP server that auto-refreshes when checkpoints change.
    port : int
        Port for the live server (default 8080).
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

    if live:
        _run_live_server(workspace, visualizer, result_path, port)


def _run_live_server(
    workspace: Path,
    visualizer: "TreeVisualizer",
    html_path: Path,
    port: int,
) -> None:
    """Start an HTTP server that watches for checkpoint changes and regenerates.

    Uses stdlib http.server with a polling-based file watcher (no external
    dependencies). The generated HTML includes a meta refresh tag for
    auto-reload in the browser.
    """
    import http.server
    import threading

    checkpoint_dir = workspace / "checkpoints"

    def _get_latest_mtime() -> float:
        """Get the most recent modification time of any checkpoint file."""
        try:
            files = sorted(checkpoint_dir.glob("search_state_step_*.json"))
            if files:
                return max(f.stat().st_mtime for f in files)
        except OSError:
            pass
        return 0.0

    last_mtime = _get_latest_mtime()

    def _watcher():
        """Background thread that polls checkpoints and regenerates HTML."""
        nonlocal last_mtime
        while True:
            time.sleep(5)
            current_mtime = _get_latest_mtime()
            if current_mtime > last_mtime:
                last_mtime = current_mtime
                try:
                    visualizer.generate_html(output_path=html_path)
                    console.print("[dim]Visualization refreshed[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Refresh failed: {e}[/yellow]")

    # Inject auto-refresh meta tag into the HTML
    _inject_auto_refresh(html_path, interval_sec=10)

    # Start watcher thread
    watcher_thread = threading.Thread(target=_watcher, daemon=True)
    watcher_thread.start()

    # Start HTTP server
    serve_dir = html_path.parent

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, format, *args):
            pass  # suppress access logs

    console.print(f"[bold cyan]Live visualization at http://localhost:{port}/{html_path.name}[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        server = http.server.HTTPServer(("", port), Handler)
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[green]Live server stopped[/green]")


def _inject_auto_refresh(html_path: Path, interval_sec: int = 10) -> None:
    """Add a meta refresh tag to the generated HTML for auto-reload."""
    content = html_path.read_text(encoding="utf-8")
    refresh_tag = f'<meta http-equiv="refresh" content="{interval_sec}">'
    if refresh_tag not in content:
        content = content.replace("<head>", f"<head>{refresh_tag}", 1)
        html_path.write_text(content, encoding="utf-8")
