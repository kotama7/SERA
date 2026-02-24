"""sera status and show-node command implementations."""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def run_status(work_dir: str) -> None:
    """Display current search state summary."""
    workspace = Path(work_dir)
    search_log = workspace / "logs" / "search_log.jsonl"

    if not search_log.exists():
        console.print("[yellow]No search log found. Run 'sera research' first.[/yellow]")
        return

    entries = []
    with open(search_log) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        console.print("[yellow]Search log is empty.[/yellow]")
        return

    # Summary stats
    total = len(entries)
    latest = entries[-1]

    console.print(f"\n[bold]SERA Search Status[/bold]")
    console.print(f"  Total events: {total}")
    console.print(f"  Total nodes: {latest.get('total_nodes', 'N/A')}")
    console.print(f"  Open list size: {latest.get('open_list_size', 'N/A')}")
    console.print(f"  Best LCB: {latest.get('lcb', 'N/A')}")
    console.print(f"  Last event: {latest.get('event', 'N/A')}")
    console.print(f"  Timestamp: {latest.get('timestamp', 'N/A')}")

    # Budget info
    budget = latest.get("budget_consumed", {})
    if budget:
        console.print(f"  Budget consumed: {budget}")

    # Show top nodes from eval log
    eval_log = workspace / "logs" / "eval_log.jsonl"
    if eval_log.exists():
        evals = []
        with open(eval_log) as f:
            for line in f:
                if line.strip():
                    evals.append(json.loads(line))

        if evals:
            table = Table(title="Top Evaluated Nodes")
            table.add_column("Node ID", style="cyan", max_width=12)
            table.add_column("μ", justify="right")
            table.add_column("SE", justify="right")
            table.add_column("LCB", justify="right")
            table.add_column("Repeats", justify="right")

            # Deduplicate by node_id, keep latest
            by_node = {}
            for e in evals:
                nid = e.get("node_id", "")
                by_node[nid] = e

            sorted_nodes = sorted(by_node.values(), key=lambda x: x.get("lcb", float("-inf")), reverse=True)[:10]
            for n in sorted_nodes:
                table.add_row(
                    n.get("node_id", "")[:12],
                    f"{n.get('mu', 'N/A'):.4f}" if isinstance(n.get("mu"), (int, float)) else "N/A",
                    f"{n.get('se', 'N/A'):.4f}" if isinstance(n.get("se"), (int, float)) else "N/A",
                    f"{n.get('lcb', 'N/A'):.4f}" if isinstance(n.get("lcb"), (int, float)) else "N/A",
                    str(n.get("n_repeats_done", "N/A")),
                )
            console.print(table)


def run_show_node(node_id: str, work_dir: str) -> None:
    """Display detailed information about a specific node."""
    workspace = Path(work_dir)
    run_dir = workspace / "runs" / node_id

    if not run_dir.exists():
        console.print(f"[red]Node {node_id} not found in runs/[/red]")
        return

    console.print(f"\n[bold]Node: {node_id}[/bold]")

    # Metrics
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.loads(f.read())
        console.print(f"\n[cyan]Metrics:[/cyan]")
        console.print(json.dumps(metrics, indent=2))

    # Experiment code (any extension)
    scripts = list(run_dir.glob("experiment.*"))
    for exp_path in scripts:
        console.print(f"\n[cyan]Experiment code:[/cyan] {exp_path}")

    # Logs
    for log_name in ["stdout.log", "stderr.log"]:
        log_path = run_dir / log_name
        if log_path.exists():
            content = log_path.read_text()
            if content.strip():
                console.print(f"\n[cyan]{log_name} (last 20 lines):[/cyan]")
                lines = content.strip().split("\n")
                for line in lines[-20:]:
                    console.print(f"  {line}")
