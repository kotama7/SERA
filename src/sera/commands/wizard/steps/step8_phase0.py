"""Step 8: Phase 0 parameters and execution."""

from __future__ import annotations

import os
from pathlib import Path

from rich.prompt import Confirm, IntPrompt

from sera.commands.wizard.i18n import get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import console, step_header


def step8_phase0(state: WizardState, lang: str, work_dir: Path) -> None:
    """Step 8: Configure Phase 0 parameters and run related work collection."""
    import yaml

    step_header(8, "Phase 0", lang)

    # Check API keys before running Phase 0
    api_keys = {
        "SEMANTIC_SCHOLAR_API_KEY": os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
        "SERPAPI_API_KEY": os.environ.get("SERPAPI_API_KEY"),
    }
    missing_keys = [k for k, v in api_keys.items() if not v]
    if missing_keys:
        console.print(f"  [yellow]Warning: Missing API keys: {', '.join(missing_keys)}[/yellow]")
        console.print("  [yellow]Phase 0 may have limited functionality without these keys.[/yellow]")
        if not Confirm.ask("Continue anyway?", default=True):
            return

    params = state.phase0_params
    params.setdefault("topk", 10)
    params.setdefault("teacher_papers", 5)
    params.setdefault("citation_depth", 1)
    params.setdefault("years_bias", 5)

    console.print(f"  top_k_papers: {params['topk']}")
    console.print(f"  teacher_papers: {params['teacher_papers']}")
    console.print(f"  citation_depth: {params['citation_depth']}")
    console.print(f"  years_bias: {params['years_bias']}")

    if Confirm.ask("Modify parameters?", default=False):
        params["topk"] = IntPrompt.ask("top_k_papers", default=params["topk"])
        params["teacher_papers"] = IntPrompt.ask("teacher_papers", default=params["teacher_papers"])

    console.print(get_message("phase0_running", lang))

    # Save Input-1 and run Phase 0
    input1_path = work_dir / "specs" / "input1.yaml"
    input1_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input1_path, "w") as f:
        yaml.dump(state.input1_data, f, default_flow_style=False, allow_unicode=True)

    from sera.commands.init_cmd import run_init

    run_init(str(input1_path), str(work_dir))

    from sera.commands.phase0_cmd import run_phase0

    run_phase0(
        str(work_dir),
        topk=params["topk"],
        teacher_papers=params["teacher_papers"],
        citation_depth=params["citation_depth"],
        years_bias=params["years_bias"],
    )

    # Count collected papers
    rw_path = work_dir / "specs" / "related_work_spec.yaml"
    n_papers = 0
    if rw_path.exists():
        rw_data = yaml.safe_load(rw_path.read_text())
        n_papers = len(rw_data.get("papers", []))

    console.print(get_message("phase0_done", lang, n_papers=n_papers))
