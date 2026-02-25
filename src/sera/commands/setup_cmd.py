"""Interactive Setup Wizard per section 27.

Guides users through Input-1 construction, Phase 0 (related work), and
Phase 1 (spec freezing) with step-by-step prompts, validation, and
state persistence for resume.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt

console = Console()

# ---------------------------------------------------------------------------
# i18n message dictionaries
# ---------------------------------------------------------------------------

MESSAGES: dict[str, dict[str, str]] = {
    "ja": {
        "welcome": "SERA セットアップウィザードへようこそ！",
        "step_header": "Step {step}/{total}",
        "data_desc": "データの説明を入力してください（例: 画像分類用のCIFAR-10データセット）",
        "data_loc": "データの場所を入力してください（パス/URI/リポジトリURL）",
        "data_format": "データ形式を選択してください",
        "data_size": "データサイズの目安を選択してください",
        "domain_field": "研究分野を入力してください（例: HPC, NLP, CV, materials）",
        "domain_subfield": "より具体的な分野を入力してください（空可）",
        "task_brief": "研究タスクを1〜3文で説明してください",
        "task_type": "タスクの種類を選択してください",
        "goal_objective": "研究の目標を記述してください（例: 実行時間を最小化）",
        "goal_direction": "目標の方向を選択してください",
        "goal_metric": "評価指標の名前を入力してください（例: accuracy, runtime_sec）",
        "goal_baseline": "既知のベースライン値があれば入力してください（空可）",
        "add_constraint": "制約条件を追加しますか？",
        "constraint_name": "制約条件の名前",
        "constraint_type": "制約の種類を選択してください",
        "constraint_threshold": "閾値を入力してください",
        "notes": "その他の備考があれば入力してください（空可）",
        "preview_confirm": "この内容でよろしいですか？",
        "phase0_params": "Phase 0 パラメータを確認してください",
        "phase0_running": "Phase 0 実行中...",
        "phase0_done": "Phase 0 完了: {n_papers} 論文を収集しました",
        "review_papers": "収集された論文をレビューしてください",
        "spec_review": "生成されたSpecをレビューしてください",
        "freeze_confirm": "この設定でSpecをフリーズしますか？",
        "setup_done": "セットアップ完了！ sera research で探索を開始できます",
        "direction_estimate": "目標「{obj}」から direction = \"{dir}\" と推定しました。正しいですか？",
        "back_help": "(back: 戻る / help: ヘルプ / quit: 中断保存)",
        "state_saved": "状態を保存しました。--resume で再開できます",
        "resuming": "前回の状態から再開します（Step {step}）",
        "env_detect_gpu": "GPU検出: {info}",
        "env_detect_slurm": "SLURM検出: {info}",
    },
    "en": {
        "welcome": "Welcome to SERA Setup Wizard!",
        "step_header": "Step {step}/{total}",
        "data_desc": "Describe your data (e.g., 'CIFAR-10 image classification dataset')",
        "data_loc": "Enter data location (path/URI/repository URL)",
        "data_format": "Select data format",
        "data_size": "Select approximate data size",
        "domain_field": "Enter research field (e.g., HPC, NLP, CV, materials)",
        "domain_subfield": "Enter subfield (optional, press Enter to skip)",
        "task_brief": "Describe the research task in 1-3 sentences",
        "task_type": "Select task type",
        "goal_objective": "Describe the research goal (e.g., 'Minimize runtime')",
        "goal_direction": "Select optimization direction",
        "goal_metric": "Enter metric name (e.g., accuracy, runtime_sec)",
        "goal_baseline": "Enter baseline value if known (optional)",
        "add_constraint": "Add a constraint?",
        "constraint_name": "Constraint name",
        "constraint_type": "Select constraint type",
        "constraint_threshold": "Enter threshold value",
        "notes": "Any additional notes (optional)",
        "preview_confirm": "Does this look correct?",
        "phase0_params": "Review Phase 0 parameters",
        "phase0_running": "Running Phase 0...",
        "phase0_done": "Phase 0 complete: collected {n_papers} papers",
        "review_papers": "Review collected papers",
        "spec_review": "Review generated specs",
        "freeze_confirm": "Freeze specs with these settings?",
        "setup_done": "Setup complete! Run sera research to start exploration",
        "direction_estimate": "Estimated direction = \"{dir}\" from goal \"{obj}\". Correct?",
        "back_help": "(back: go back / help: show help / quit: save & exit)",
        "state_saved": "State saved. Use --resume to continue later",
        "resuming": "Resuming from previous state (Step {step})",
        "env_detect_gpu": "GPU detected: {info}",
        "env_detect_slurm": "SLURM detected: {info}",
    },
}

TOTAL_STEPS = 11

# ---------------------------------------------------------------------------
# Direction estimation
# ---------------------------------------------------------------------------


def estimate_direction(objective: str) -> str | None:
    """Estimate optimization direction from objective text."""
    minimize_kw = [
        "最小", "minimize", "reduce", "lower", "decrease", "短縮",
        "削減", "抑制", "loss", "error", "latency", "runtime", "time",
    ]
    maximize_kw = [
        "最大", "maximize", "increase", "improve", "higher", "向上",
        "精度", "accuracy", "score", "throughput", "performance",
    ]
    obj_lower = objective.lower()
    min_score = sum(1 for kw in minimize_kw if kw in obj_lower)
    max_score = sum(1 for kw in maximize_kw if kw in obj_lower)

    if min_score > max_score:
        return "minimize"
    elif max_score > min_score:
        return "maximize"
    return None


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def detect_environment() -> dict[str, Any]:
    """Auto-detect GPU and SLURM availability."""
    import shutil
    import subprocess

    env: dict[str, Any] = {"gpu_available": False, "gpu_info": "", "slurm_available": False}

    # GPU detection
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                env["gpu_available"] = True
                env["gpu_info"] = out.stdout.strip().split("\n")[0]
        except Exception:
            pass

    # SLURM detection
    if shutil.which("sinfo"):
        env["slurm_available"] = True
        try:
            out = subprocess.run(
                ["sinfo", "--summarize", "--noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0:
                env["slurm_info"] = out.stdout.strip()[:200]
        except Exception:
            pass

    return env


# ---------------------------------------------------------------------------
# Wizard state
# ---------------------------------------------------------------------------


class WizardState:
    """Persistent wizard state for resume support."""

    def __init__(self, work_dir: Path):
        self.state_path = work_dir / ".wizard_state.json"
        self.current_step: int = 1
        self.input1_data: dict[str, Any] = {}
        self.phase0_params: dict[str, Any] = {}
        self.phase1_params: dict[str, Any] = {}

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current_step": self.current_step,
            "input1_data": self.input1_data,
            "phase0_params": self.phase0_params,
            "phase1_params": self.phase1_params,
        }
        self.state_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def load(self) -> bool:
        if not self.state_path.exists():
            return False
        data = json.loads(self.state_path.read_text())
        self.current_step = data.get("current_step", 1)
        self.input1_data = data.get("input1_data", {})
        self.phase0_params = data.get("phase0_params", {})
        self.phase1_params = data.get("phase1_params", {})
        return True


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def _prompt(msg_key: str, lang: str, **kwargs: Any) -> str:
    """Get a localized message."""
    return MESSAGES[lang][msg_key].format(**kwargs)


def _select(prompt_text: str, choices: list[str], default: str = "") -> str:
    """Show numbered choices and return the selected value."""
    for i, c in enumerate(choices, 1):
        console.print(f"  [{i}] {c}")
    while True:
        raw = Prompt.ask(prompt_text, default=default or str(1))
        try:
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        except ValueError:
            if raw in choices:
                return raw
        console.print(f"  [red]Please enter 1-{len(choices)}[/red]")


def _step_header(step: int, title: str, lang: str) -> None:
    header = _prompt("step_header", lang, step=step, total=TOTAL_STEPS)
    console.print(Panel(f"[bold]{header}[/bold] {title}", style="cyan"))


def step1_data(state: WizardState, lang: str) -> None:
    """Step 1: Data information."""
    _step_header(1, "Data", lang)
    data = state.input1_data.setdefault("data", {})
    data["description"] = Prompt.ask(_prompt("data_desc", lang), default=data.get("description", ""))
    data["location"] = Prompt.ask(_prompt("data_loc", lang), default=data.get("location", ""))
    data["format"] = _select(
        _prompt("data_format", lang),
        ["csv", "json", "parquet", "code", "pdf", "mixed"],
        default=data.get("format", "csv"),
    )
    data["size_hint"] = _select(
        _prompt("data_size", lang),
        ["small(<1GB)", "medium(1-100GB)", "large(>100GB)"],
        default=data.get("size_hint", "small(<1GB)"),
    )


def step2_domain(state: WizardState, lang: str) -> None:
    """Step 2: Domain information."""
    _step_header(2, "Domain", lang)
    domain = state.input1_data.setdefault("domain", {})
    domain["field"] = Prompt.ask(_prompt("domain_field", lang), default=domain.get("field", ""))
    domain["subfield"] = Prompt.ask(_prompt("domain_subfield", lang), default=domain.get("subfield", ""))


def step3_task(state: WizardState, lang: str) -> None:
    """Step 3: Task information."""
    _step_header(3, "Task", lang)
    task = state.input1_data.setdefault("task", {})
    task["brief"] = Prompt.ask(_prompt("task_brief", lang), default=task.get("brief", ""))
    task["type"] = _select(
        _prompt("task_type", lang),
        ["optimization", "prediction", "generation", "analysis", "comparison"],
        default=task.get("type", "optimization"),
    )


def step4_goal(state: WizardState, lang: str) -> None:
    """Step 4: Goal information."""
    _step_header(4, "Goal", lang)
    goal = state.input1_data.setdefault("goal", {})
    goal["objective"] = Prompt.ask(_prompt("goal_objective", lang), default=goal.get("objective", ""))
    goal["metric"] = Prompt.ask(_prompt("goal_metric", lang), default=goal.get("metric", "score"))

    # Direction estimation
    estimated = estimate_direction(goal["objective"])
    if estimated:
        msg = _prompt("direction_estimate", lang, obj=goal["objective"], dir=estimated)
        if Confirm.ask(msg, default=True):
            goal["direction"] = estimated
        else:
            goal["direction"] = _select(
                _prompt("goal_direction", lang), ["minimize", "maximize"],
            )
    else:
        goal["direction"] = _select(
            _prompt("goal_direction", lang), ["minimize", "maximize"],
        )

    goal["baseline"] = Prompt.ask(_prompt("goal_baseline", lang), default=goal.get("baseline", ""))


def step5_constraints(state: WizardState, lang: str) -> None:
    """Step 5: Constraints."""
    _step_header(5, "Constraints", lang)
    constraints = state.input1_data.setdefault("constraints", [])
    console.print(f"  Current constraints: {len(constraints)}")

    while Confirm.ask(_prompt("add_constraint", lang), default=False):
        c: dict[str, Any] = {}
        c["name"] = Prompt.ask(_prompt("constraint_name", lang))
        c["type"] = _select(
            _prompt("constraint_type", lang),
            ["ge", "le", "eq", "bool"],
        )
        if c["type"] != "bool":
            c["threshold"] = Prompt.ask(_prompt("constraint_threshold", lang))
        constraints.append(c)
        console.print(f"  [green]Added constraint: {c['name']} ({c['type']})[/green]")


def step6_notes(state: WizardState, lang: str) -> None:
    """Step 6: Notes."""
    _step_header(6, "Notes", lang)
    state.input1_data["notes"] = Prompt.ask(
        _prompt("notes", lang), default=state.input1_data.get("notes", "")
    )


def step7_preview(state: WizardState, lang: str) -> bool:
    """Step 7: Preview and confirm Input-1."""
    _step_header(7, "Preview", lang)
    import yaml

    console.print(Panel(yaml.dump(state.input1_data, default_flow_style=False, allow_unicode=True), title="Input-1"))
    return Confirm.ask(_prompt("preview_confirm", lang), default=True)


def step8_phase0(state: WizardState, lang: str, work_dir: Path) -> None:
    """Step 8: Phase 0 parameters and execution."""
    _step_header(8, "Phase 0", lang)

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

    console.print(_prompt("phase0_running", lang))

    # Save Input-1 and run Phase 0
    import yaml

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

    console.print(_prompt("phase0_done", lang, n_papers=n_papers))


def step9_review(state: WizardState, lang: str, work_dir: Path) -> None:
    """Step 9: Review Phase 0 results."""
    _step_header(9, "Review", lang)
    import yaml

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


def step10_specs(state: WizardState, lang: str, work_dir: Path) -> None:
    """Step 10: Generate and review specs."""
    _step_header(10, "Specs", lang)

    # Detect environment
    env = detect_environment()
    if env["gpu_available"]:
        console.print(_prompt("env_detect_gpu", lang, info=env["gpu_info"]))
    if env["slurm_available"]:
        console.print(_prompt("env_detect_slurm", lang, info=env.get("slurm_info", "available")))

    params = state.phase1_params
    params.setdefault("executor", "slurm" if env["slurm_available"] else "local")
    params.setdefault("gpu_required", env["gpu_available"])
    params.setdefault("max_nodes", 100)
    params.setdefault("repeats", 3)

    console.print(f"  executor: {params['executor']}")
    console.print(f"  gpu_required: {params['gpu_required']}")
    console.print(f"  max_nodes: {params['max_nodes']}")
    console.print(f"  repeats: {params['repeats']}")

    if Confirm.ask("Modify parameters?", default=False):
        params["executor"] = _select("Executor", ["local", "slurm", "docker"], default=params["executor"])
        params["max_nodes"] = IntPrompt.ask("max_nodes", default=params["max_nodes"])
        params["repeats"] = IntPrompt.ask("repeats", default=params["repeats"])


def step11_freeze(state: WizardState, lang: str, work_dir: Path) -> None:
    """Step 11: Freeze specs."""
    _step_header(11, "Freeze", lang)

    if not Confirm.ask(_prompt("freeze_confirm", lang), default=True):
        return

    params = state.phase1_params
    cli_args = {
        "work_dir": str(work_dir),
        "auto": True,
        "max_nodes": params.get("max_nodes", 100),
        "repeats": params.get("repeats", 3),
        "executor": params.get("executor", "local"),
        "gpu_required": params.get("gpu_required", True),
    }

    from sera.commands.phase1_cmd import run_freeze_specs
    run_freeze_specs(str(work_dir), auto=True, cli_args=cli_args)

    console.print(f"\n  [bold green]{_prompt('setup_done', lang)}[/bold green]")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

STEPS = [
    step1_data,
    step2_domain,
    step3_task,
    step4_goal,
    step5_constraints,
    step6_notes,
    # step7 is special (returns bool)
]


def run_setup(
    work_dir: str = "./sera_workspace",
    resume: bool = False,
    from_input1: str | None = None,
    skip_phase0: bool = False,
    lang: str = "ja",
) -> None:
    """Run the interactive setup wizard."""
    if lang not in MESSAGES:
        lang = "ja"

    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)

    state = WizardState(wd)

    # Resume from saved state
    if resume and state.load():
        console.print(_prompt("resuming", lang, step=state.current_step))
    else:
        state.current_step = 1

    console.print(Panel(_prompt("welcome", lang), style="bold blue"))

    # Phase A: Input-1 construction (Steps 1-7)
    if from_input1:
        import yaml

        with open(from_input1) as f:
            state.input1_data = yaml.safe_load(f)
        state.current_step = max(state.current_step, 8)
        console.print(f"  Loaded Input-1 from {from_input1}")
    else:
        step_fns = [step1_data, step2_domain, step3_task, step4_goal, step5_constraints, step6_notes]
        for i, fn in enumerate(step_fns, 1):
            if state.current_step > i:
                continue
            state.current_step = i
            try:
                fn(state, lang)
                state.save()
            except (KeyboardInterrupt, EOFError):
                state.save()
                console.print(f"\n  {_prompt('state_saved', lang)}")
                sys.exit(0)

        # Step 7: Preview
        if state.current_step <= 7:
            state.current_step = 7
            try:
                confirmed = step7_preview(state, lang)
                if not confirmed:
                    state.current_step = 1
                    state.save()
                    console.print("  Restarting from Step 1...")
                    return run_setup(work_dir, resume=True, lang=lang)
                state.save()
            except (KeyboardInterrupt, EOFError):
                state.save()
                console.print(f"\n  {_prompt('state_saved', lang)}")
                sys.exit(0)

    # Phase B: Phase 0 (Steps 8-9)
    if not skip_phase0:
        for step_num, fn in [(8, step8_phase0), (9, step9_review)]:
            if state.current_step > step_num:
                continue
            state.current_step = step_num
            try:
                fn(state, lang, wd)
                state.save()
            except (KeyboardInterrupt, EOFError):
                state.save()
                console.print(f"\n  {_prompt('state_saved', lang)}")
                sys.exit(0)

    # Phase C: Phase 1 (Steps 10-11)
    for step_num, fn in [(10, step10_specs), (11, step11_freeze)]:
        if state.current_step > step_num:
            continue
        state.current_step = step_num
        try:
            fn(state, lang, wd)
            state.save()
        except (KeyboardInterrupt, EOFError):
            state.save()
            console.print(f"\n  {_prompt('state_saved', lang)}")
            sys.exit(0)

    # Cleanup state file
    if state.state_path.exists():
        state.state_path.unlink()
