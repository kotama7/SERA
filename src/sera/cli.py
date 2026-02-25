"""SERA CLI - Self-Evolving Research Agent."""

import typer

from sera.utils.logging import setup_structlog

app = typer.Typer(name="sera", help="Self-Evolving Research Agent")


@app.callback()
def main_callback():
    """Initialize structured logging on CLI startup."""
    setup_structlog()


@app.command()
def init(input1_path: str, work_dir: str = "./sera_workspace"):
    """Input-1 を読み込み、workspace を初期化"""
    from sera.commands.init_cmd import run_init

    run_init(input1_path, work_dir)


@app.command()
def phase0_related_work(
    work_dir: str = "./sera_workspace",
    topk: int = 10,
    teacher_papers: int = 5,
    citation_depth: int = 1,
    years_bias: int = 5,
    api_priority: str = "semantic_scholar,crossref,arxiv,web",
):
    """Phase 0: 先行研究収集"""
    from sera.commands.phase0_cmd import run_phase0

    run_phase0(work_dir, topk, teacher_papers, citation_depth, years_bias, api_priority)


@app.command()
def freeze_specs(
    work_dir: str = "./sera_workspace",
    auto: bool = False,
    topk: int = 10,
    teacher_papers: int = 5,
    citation_depth: int = 1,
    years_bias: int = 5,
    max_nodes: int = 100,
    max_depth: int = 10,
    branch_factor: int = 3,
    lambda_cost: float = 0.1,
    beta: float = 0.05,
    repeats: int = 3,
    lcb_coef: float = 1.96,
    no_sequential: bool = False,
    seq_topk: int = 5,
    rank: int = 16,
    alpha: int = 32,
    lr: float = 1e-4,
    clip: float = 0.2,
    ppo_steps: int = 128,
    no_kl: bool = False,
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    dtype: str = "bf16",
    agent_llm: str = "local:same_as_base",
    executor: str = "local",
    gpu_count: int = 1,
    memory_gb: int = 32,
    cpu_cores: int = 8,
    gpu_type: str = "",
    gpu_required: bool = True,
    timeout: int = 3600,
    no_web: bool = False,
    strategy: str = "best_first",
    squash_depth: int = 6,
    no_snapshot_topk: bool = False,
    api_priority: str = "semantic_scholar,crossref,arxiv,web",
    no_turn_rewards: bool = False,
    turn_w_phase0: float = 0.15,
    turn_w_phase2: float = 0.25,
    turn_w_phase3: float = 0.20,
    turn_w_phase5: float = 0.20,
    turn_w_phase7: float = 0.20,
):
    """Phase 1: 全Spec確定、ExecutionSpec固定"""
    from sera.commands.phase1_cmd import run_freeze_specs

    run_freeze_specs(work_dir, auto, locals())


@app.command()
def research(
    work_dir: str = "./sera_workspace",
    resume: bool = False,
    skip_phase0: bool = False,
    skip_paper: bool = False,
):
    """Phase 2-6: 研究ループ"""
    from sera.commands.research_cmd import run_research

    run_research(work_dir, resume, skip_phase0=skip_phase0, skip_paper=skip_paper)


@app.command()
def export_best(work_dir: str = "./sera_workspace"):
    """best成果物を outputs/best/ に集約"""
    from sera.commands.export_cmd import run_export_best

    run_export_best(work_dir)


@app.command()
def generate_paper(work_dir: str = "./sera_workspace"):
    """Phase 7: 論文生成"""
    from sera.commands.paper_cmd import run_generate_paper

    run_generate_paper(work_dir)


@app.command()
def evaluate_paper(work_dir: str = "./sera_workspace"):
    """Phase 8: 論文評価・改善ループ"""
    from sera.commands.paper_cmd import run_evaluate_paper

    run_evaluate_paper(work_dir)


@app.command()
def status(work_dir: str = "./sera_workspace"):
    """現在の探索状態サマリ表示"""
    from sera.commands.status_cmd import run_status

    run_status(work_dir)


@app.command()
def show_node(node_id: str, work_dir: str = "./sera_workspace"):
    """ノード詳細表示"""
    from sera.commands.status_cmd import run_show_node

    run_show_node(node_id, work_dir)


@app.command()
def replay(
    node_id: str = typer.Option(..., "--node-id", help="Node ID to replay"),
    seed: int = typer.Option(..., "--seed", help="Random seed for replay"),
    work_dir: str = "./sera_workspace",
):
    """特定ノードの実験再実行"""
    from sera.commands.replay_cmd import run_replay

    run_replay(node_id, seed, work_dir)


@app.command()
def validate_specs(work_dir: str = "./sera_workspace"):
    """Spec整合性チェック"""
    from sera.commands.validate_cmd import run_validate_specs

    run_validate_specs(work_dir)


@app.command()
def setup(
    work_dir: str = "./sera_workspace",
    resume: bool = False,
    from_input1: str | None = None,
    skip_phase0: bool = False,
    lang: str = "ja",
):
    """対話型セットアップウィザード"""
    from sera.commands.setup_cmd import run_setup

    run_setup(work_dir, resume, from_input1, skip_phase0, lang)


@app.command()
def visualize(
    work_dir: str = "./sera_workspace",
    step: int | None = None,
    output: str | None = None,
):
    """探索木の可視化HTML生成"""
    from sera.commands.visualize_cmd import run_visualize

    run_visualize(work_dir, step=step, output=output)


if __name__ == "__main__":
    app()
