"""sera replay command implementation."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def _find_experiment_script(run_dir: Path) -> Path | None:
    """Find experiment script in run directory (any extension)."""
    scripts = list(run_dir.glob("experiment.*"))
    if scripts:
        return scripts[0]
    return None


def run_replay(node_id: str, seed: int, work_dir: str) -> None:
    """Re-run a specific node's experiment with given seed."""
    workspace = Path(work_dir)
    run_dir = workspace / "runs" / node_id

    if not run_dir.exists():
        console.print(f"[red]Node {node_id} not found in runs/[/red]")
        raise SystemExit(1)

    exp_path = _find_experiment_script(run_dir)
    if exp_path is None:
        console.print(f"[red]No experiment script found for node {node_id}[/red]")
        raise SystemExit(1)

    # Load resource spec for timeout
    import yaml

    specs_dir = workspace / "specs"
    timeout = 3600
    resource_path = specs_dir / "resource_spec.yaml"
    if resource_path.exists():
        with open(resource_path) as f:
            data = yaml.safe_load(f)
        rs = data.get("resource_spec", data) if data else {}
        sandbox = rs.get("sandbox", {})
        timeout = sandbox.get("experiment_timeout_sec", 3600)

    # Load language config from problem spec if available
    interpreter_command = None
    seed_arg_format = None
    problem_path = specs_dir / "problem_spec.yaml"
    if problem_path.exists():
        with open(problem_path) as f:
            pdata = yaml.safe_load(f)
        if pdata and "language" in pdata:
            lang = pdata["language"]
            interpreter_command = lang.get("interpreter_command")
            seed_arg_format = lang.get("seed_arg_format")

    # Select executor based on resource spec
    executor_type = "local"
    slurm_cfg = None
    compute_cfg = None
    if resource_path.exists():
        with open(resource_path) as f:
            rdata = yaml.safe_load(f)
        rs = rdata.get("resource_spec", rdata) if rdata else {}
        compute = rs.get("compute", {})
        executor_type = compute.get("executor_type", "local")
        if executor_type == "slurm":
            slurm_cfg = rs.get("slurm", {})
            from sera.specs.resource_spec import ComputeConfig
            compute_cfg = ComputeConfig(**compute)

    if executor_type == "slurm" and slurm_cfg is not None:
        from sera.execution.slurm_executor import SlurmExecutor
        from sera.specs.resource_spec import SlurmConfig

        executor = SlurmExecutor(
            work_dir=workspace,
            slurm_config=SlurmConfig(**slurm_cfg),
            compute_config=compute_cfg,
            interpreter_command=interpreter_command,
            seed_arg_format=seed_arg_format,
        )
    else:
        from sera.execution.local_executor import LocalExecutor

        executor = LocalExecutor(
            work_dir=workspace,
            interpreter_command=interpreter_command,
            seed_arg_format=seed_arg_format,
        )

    replay_id = f"{node_id}_replay_{seed}"
    replay_dir = workspace / "runs" / replay_id
    replay_dir.mkdir(parents=True, exist_ok=True)

    # Copy experiment script
    import shutil

    dest_script = replay_dir / exp_path.name
    shutil.copy2(exp_path, dest_script)

    console.print(f"[cyan]Replaying node {node_id} with seed {seed}...[/cyan]")

    result = executor.run(
        node_id=replay_id,
        script_path=dest_script,
        seed=seed,
        timeout_sec=timeout,
    )

    if result.success:
        console.print("[green]Replay succeeded![/green]")
        if result.metrics_path and result.metrics_path.exists():
            import json

            with open(result.metrics_path) as f:
                metrics = json.loads(f.read())
            console.print(f"Metrics: {json.dumps(metrics, indent=2)}")
    else:
        console.print(f"[red]Replay failed with exit code {result.exit_code}[/red]")
        if result.stderr_path.exists():
            console.print(result.stderr_path.read_text()[-500:])
