"""Step 9: Review Phase 0 results."""

from __future__ import annotations

from pathlib import Path

from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import console, step_header


def step9_review(state: WizardState, lang: str, work_dir: Path) -> None:
    """Step 9: Review collected papers from Phase 0."""
    import yaml

    step_header(9, "Review", lang)

    rw_path = work_dir / "specs" / "related_work_spec.yaml"
    if rw_path.exists():
        rw_data = yaml.safe_load(rw_path.read_text())
        papers = rw_data.get("papers", [])
        console.print(f"\n  Collected {len(papers)} papers:")
        for i, p in enumerate(papers[:10], 1):
            title = p.get("title", "Unknown")
            year = p.get("year", "?")
            console.print(f"  {i}. [{year}] {title}")
        if len(papers) > 10:
            console.print(f"  ... and {len(papers) - 10} more")
    else:
        console.print("  [yellow]No related work spec found[/yellow]")
