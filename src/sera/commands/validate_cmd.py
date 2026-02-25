"""sera validate-specs command implementation."""

from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def run_validate_specs(work_dir: str) -> None:
    """Validate all specs for consistency."""
    workspace = Path(work_dir)
    specs_dir = workspace / "specs"

    if not specs_dir.exists():
        console.print("[red]specs/ directory not found.[/red]")
        raise SystemExit(1)

    errors = []
    warnings = []

    # Check all required files
    required_files = [
        "input1.yaml",
        "related_work_spec.yaml",
        "paper_spec.yaml",
        "paper_score_spec.yaml",
        "teacher_paper_set.yaml",
        "problem_spec.yaml",
        "model_spec.yaml",
        "resource_spec.yaml",
        "plan_spec.yaml",
        "execution_spec.yaml",
        "execution_spec.yaml.lock",
    ]

    for filename in required_files:
        path = specs_dir / filename
        if not path.exists():
            errors.append(f"Missing: {filename}")
        else:
            console.print(f"  [green]OK[/green] {filename}")

    # Validate ExecutionSpec integrity
    from sera.phase1.spec_freezer import SpecFreezer

    freezer = SpecFreezer()
    if (specs_dir / "execution_spec.yaml").exists() and (specs_dir / "execution_spec.yaml.lock").exists():
        if freezer.verify(specs_dir):
            console.print("  [green]OK[/green] ExecutionSpec hash verified")
        else:
            errors.append("ExecutionSpec hash mismatch (tampered)")

    # Validate each spec can be loaded
    spec_loaders = {
        "input1.yaml": ("sera.specs.input1", "Input1Model", None),
        "problem_spec.yaml": ("sera.specs.problem_spec", "ProblemSpecModel", "problem_spec"),
        "model_spec.yaml": ("sera.specs.model_spec", "ModelSpecModel", "model_spec"),
        "resource_spec.yaml": ("sera.specs.resource_spec", "ResourceSpecModel", "resource_spec"),
        "plan_spec.yaml": ("sera.specs.plan_spec", "PlanSpecModel", "plan_spec"),
        "execution_spec.yaml": ("sera.specs.execution_spec", "ExecutionSpecModel", "execution_spec"),
        "paper_spec.yaml": ("sera.specs.paper_spec", "PaperSpecModel", "paper_spec"),
        "paper_score_spec.yaml": ("sera.specs.paper_score_spec", "PaperScoreSpecModel", "paper_score_spec"),
    }

    for filename, (module_path, class_name, yaml_key) in spec_loaders.items():
        path = specs_dir / filename
        if not path.exists():
            continue
        try:
            import importlib

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            with open(path) as f:
                data = yaml.safe_load(f)
            if yaml_key and isinstance(data, dict) and yaml_key in data:
                data = data[yaml_key]
            cls(**(data or {}))
            console.print(f"  [green]Valid[/green] {filename} ({class_name})")
        except Exception as e:
            errors.append(f"Validation failed for {filename}: {e}")

    # Validate adapter_spec_hash consistency
    model_path = specs_dir / "model_spec.yaml"
    if model_path.exists():
        with open(model_path) as f:
            model_data = yaml.safe_load(f)
        ms = model_data.get("model_spec", model_data) if model_data else {}
        adapter = ms.get("adapter_spec", {})
        if adapter:
            from sera.utils.hashing import compute_adapter_spec_hash

            adapter_hash = compute_adapter_spec_hash(adapter)
            console.print(f"  [cyan]adapter_spec_hash[/cyan]: {adapter_hash}")

    # Report
    console.print()
    if errors:
        console.print(f"[red]{len(errors)} error(s) found:[/red]")
        for e in errors:
            console.print(f"  [red]ERROR: {e}[/red]")
        raise SystemExit(1)
    elif warnings:
        console.print(f"[yellow]{len(warnings)} warning(s):[/yellow]")
        for w in warnings:
            console.print(f"  [yellow]WARN: {w}[/yellow]")
    else:
        console.print("[green]All specs valid![/green]")
