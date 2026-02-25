"""sera export-best command implementation."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from rich.console import Console

console = Console()


def run_export_best(work_dir: str) -> None:
    """Export best artifacts to outputs/best/."""
    workspace = Path(work_dir)
    output_dir = workspace / "outputs" / "best"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find best node from search log
    search_log_path = workspace / "logs" / "search_log.jsonl"
    if not search_log_path.exists():
        console.print("[yellow]No search log found.[/yellow]")
        return

    # Two-pass: prefer finite LCB, fall back to mu
    best_node_id = None
    best_lcb = None
    best_mu = None
    best_se = None
    best_by_mu_node_id = None
    best_by_mu_val = None
    best_by_mu_lcb = None
    best_by_mu_se = None

    with open(search_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            lcb = entry.get("lcb")
            mu = entry.get("mu")
            se = entry.get("se")
            node_id = entry.get("node_id")

            # Track best by finite LCB
            if lcb is not None and not (math.isinf(lcb) or math.isnan(lcb)):
                if best_lcb is None or lcb > best_lcb:
                    best_lcb = lcb
                    best_node_id = node_id
                    best_mu = mu
                    best_se = se

            # Fallback: track best by mu
            if mu is not None:
                if best_by_mu_val is None or mu > best_by_mu_val:
                    best_by_mu_val = mu
                    best_by_mu_node_id = node_id
                    best_by_mu_lcb = lcb
                    best_by_mu_se = se

    # Use mu-based fallback if no finite LCB found
    if best_node_id is None and best_by_mu_node_id is not None:
        best_node_id = best_by_mu_node_id
        best_lcb = best_by_mu_lcb
        best_mu = best_by_mu_val
        best_se = best_by_mu_se

    if best_node_id is None:
        console.print("[yellow]No evaluated nodes found in search log.[/yellow]")
        return

    # Copy best node artifacts
    run_dir = workspace / "runs" / best_node_id
    if run_dir.exists():
        # Copy experiment script (any extension)
        scripts = list(run_dir.glob("experiment.*"))
        for exp_script in scripts:
            shutil.copy2(exp_script, output_dir / exp_script.name)

        # Copy metrics
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            shutil.copy2(metrics_path, output_dir / "metrics_summary.json")

    # Save best node info
    best_info = {
        "node_id": best_node_id,
        "lcb": best_lcb,
        "mu": best_mu,
        "se": best_se,
    }
    with open(output_dir / "best_node.json", "w") as f:
        json.dump(best_info, f, indent=2, default=str)

    # Generate report
    report = {
        "best_node_id": best_node_id,
        "best_lcb": best_lcb,
        "best_mu": best_mu,
        "best_se": best_se,
    }

    # Load specs if available
    specs_dir = workspace / "specs"
    for spec_file in specs_dir.glob("*.yaml"):
        try:
            import yaml

            with open(spec_file) as f:
                report[spec_file.stem] = yaml.safe_load(f)
        except Exception:
            pass

    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, default=str, indent=2)

    # Export adapter if available
    lineage_dir = workspace / "lineage" / "nodes"
    if lineage_dir.exists() and best_node_id:
        # Find adapter_node_id from the search state checkpoint
        adapter_node_id = None
        checkpoint_dir = workspace / "checkpoints"
        if checkpoint_dir.exists():
            from sera.utils.checkpoint import load_latest_checkpoint

            state = load_latest_checkpoint(checkpoint_dir)
            if state:
                node_data = state.get("all_nodes", {}).get(best_node_id, {})
                adapter_node_id = node_data.get("adapter_node_id")

        if adapter_node_id and (lineage_dir / adapter_node_id).exists():
            try:
                from sera.lineage.lineage_manager import LineageManager

                lm = LineageManager(lineage_dir=workspace / "lineage")
                full_weights = lm.materialize(adapter_node_id)
                if full_weights:
                    from safetensors.torch import save_file

                    adapter_out = output_dir / "adapter.safetensors"
                    clean = {k: v.contiguous().cpu() for k, v in full_weights.items()}
                    save_file(clean, str(adapter_out))
                    console.print(f"  Adapter exported: {adapter_out}")
            except Exception as e:
                console.print(f"  [yellow]Adapter export failed: {e}[/yellow]")

    console.print(f"[green]Best artifacts exported to {output_dir}[/green]")
