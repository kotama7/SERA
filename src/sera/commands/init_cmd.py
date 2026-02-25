"""sera init command implementation."""

from pathlib import Path
import shutil
import yaml
from rich.console import Console

console = Console()


def run_init(input1_path: str, work_dir: str) -> None:
    """Initialize SERA workspace from Input-1 YAML."""
    input1_file = Path(input1_path)
    workspace = Path(work_dir)

    if not input1_file.exists():
        console.print(f"[red]Error: Input-1 file not found: {input1_file}[/red]")
        raise SystemExit(1)

    # Validate Input-1
    with open(input1_file) as f:
        data = yaml.safe_load(f)

    from sera.specs.input1 import Input1Model

    try:
        Input1Model(**data)
    except Exception as e:
        console.print(f"[red]Error: Invalid Input-1: {e}[/red]")
        raise SystemExit(1)

    # Create workspace directory structure
    dirs = [
        "specs",
        "related_work/results",
        "related_work/teacher_papers",
        "lineage/nodes",
        "runs",
        "logs",
        "checkpoints",
        "outputs/best",
        "paper/figures",
        "docs/modules",
    ]
    for d in dirs:
        (workspace / d).mkdir(parents=True, exist_ok=True)

    # Copy Input-1 to specs/
    dest = workspace / "specs" / "input1.yaml"
    shutil.copy2(input1_file, dest)

    console.print(f"[green]Workspace initialized at {workspace}[/green]")
    console.print(f"[green]Input-1 copied to {dest}[/green]")
    console.print("\nNext steps:")
    console.print("  1. sera phase0-related-work")
    console.print("  2. sera freeze-specs")
    console.print("  3. sera research")
