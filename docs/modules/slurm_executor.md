# SlurmExecutor / SlurmJobHandle

SLURM ジョブスケジューラ経由で実験を実行する `Executor` 実装。submitit ライブラリを使用し、ComputeConfig → SLURM パラメータの自動マッピング、コンテナラッピング（Singularity/Apptainer/Docker）、非同期バッチ実行をサポートする。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `SlurmExecutor` / `SlurmJobHandle` | `src/sera/execution/slurm_executor.py` |

## 依存関係

- `submitit` -- SLURM ジョブ投入ライブラリ（`pip install sera[slurm]`）
- `sera.execution.executor` (`Executor`, `RunResult`) -- 基底クラスと結果型
- `sera.specs.resource_spec` (`ComputeConfig`, `SlurmConfig`) -- SLURM/計算設定

---

## SlurmJobHandle (dataclass)

投入された SLURM ジョブの型付きハンドル。

```python
@dataclass
class SlurmJobHandle:
    job: Any                  # submitit ジョブオブジェクト
    node_id: str
    seed: int
    run_dir: Path
    start_time: float
    timeout_sec: int | None
    slurm_log_dir: Path
    stdout_path: Path
    stderr_path: Path
    metrics_path: Path
    status: str = "pending"
```

`job_id` プロパティで SLURM ジョブ ID を文字列として返す。

---

## _build_container_cmd (モジュールレベル関数)

コンテナラッピングされたコマンドを構築する。

```python
def _build_container_cmd(
    container_config: dict[str, Any],
    run_dir: str,
    inner_cmd: list[str],
) -> list[str]
```

**対応ランタイム:**

| ランタイム | コマンド形式 |
|-----------|------------|
| `singularity` / `apptainer` | `singularity exec [--nv] [--bind ...] [--env ...] [--overlay ...] [--writable-tmpfs] <image> <cmd>` |
| `docker` | `docker run --rm [--gpus all] [-v ...] [-e ...] [-w ...] <image> <cmd>` |

**コンテナ設定フィールド:**

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `runtime` | `str` | コンテナランタイム（`"singularity"` / `"apptainer"` / `"docker"`） |
| `image` | `str` | コンテナイメージパス |
| `bind_mounts` | `list[str]` | 追加バインドマウント |
| `env_vars` | `dict[str, str]` | 環境変数 |
| `gpu_enabled` | `bool` | GPU パススルー（デフォルト: `True`） |
| `extra_flags` | `list[str]` | 追加フラグ |
| `overlay` | `str` | オーバーレイファイルシステム |
| `writable_tmpfs` | `bool` | 書き込み可能 tmpfs |

`run_dir` は自動的にコンテナにバインドされる。

---

## _run_experiment (モジュールレベル関数)

submitit 経由で SLURM ジョブ内で実行される callable。

```python
def _run_experiment(
    interpreter_command: str,
    script_path: str,
    seed: int,
    run_dir: str,
    modules: list[str],
    seed_arg_format: str = "--seed {seed}",
    container_config: dict[str, Any] | None = None,
) -> int
```

**処理フロー:**

1. 環境モジュールのロード（`module load` コマンド）
2. インタープリタコマンドとシード引数からコマンドを構築
3. `container_config` が有効な場合: `_build_container_cmd()` でラッピング
4. `subprocess.Popen` で実行、exit code を返す

---

## SlurmExecutor

### コンストラクタ

```python
def __init__(
    self,
    work_dir: str | Path,
    slurm_config: SlurmConfig,
    compute_config: ComputeConfig | None = None,
    python_executable: str = "python",
    interpreter_command: str | None = None,
    seed_arg_format: str | None = None,
    poll_interval_sec: float = 10,
)
```

- `submitit` が未インストールの場合は `ImportError` を送出
- `sacct` コマンドの利用可否を起動時にチェック（`_check_sacct_available`）
- `SlurmConfig.container.enabled=True` の場合、`ContainerConfig` を dict にシリアライズ（SLURM ジョブ間の pickle 転送用）

---

### run(node_id, script_path, seed, timeout_sec) -> RunResult

SLURM ジョブとして実験を投入し、完了を待って結果を返す。

**処理フロー:**

1. run ディレクトリと SLURM ログディレクトリを作成
2. submitit パラメータを構築:
   - `SlurmConfig` から: `partition`, `time_limit`, `account`, `modules`
   - `ComputeConfig` から（低優先度）: `gpu_count` → `slurm_gpus_per_node`, `memory_gb` → `slurm_mem`, `cpu_cores` → `slurm_cpus_per_task`, `gpu_type` → `constraint`
   - `sbatch_extra` から（高優先度）: 既存パラメータをオーバーライド
3. ジョブを投入（`executor.submit(_run_experiment, ...)`）
4. ポーリングで完了を待機（`_poll_job` / `_poll_job_squeue`）
5. submitit ログを収集（`_collect_submitit_logs`）
6. OOM 検知（`_detect_oom`）
7. `RunResult` を構築して返す

**パラメータ優先順位:** `sbatch_extra` > `ComputeConfig` > `SlurmConfig`

`sbatch_extra` で `gres` / `gpus-per-node` が指定された場合は `ComputeConfig` の GPU 設定を無視。`mem` / `mem-per-cpu` / `cpus-per-task` も同様。

---

### submit_async(node_id, script_path, seed, timeout_sec) -> SlurmJobHandle [async]

SLURM ジョブを非ブロッキングで投入し、ハンドルを即座に返す。`collect_result()` で後から結果を取得する。

### collect_result(node_id, job, start_time, ...) -> RunResult [async]

投入済みの SLURM ジョブを非同期ポーリングし、`RunResult` を返す。

### run_batch(tasks) -> list[RunResult] [async]

複数の SLURM ジョブをバッチ実行する 3 フェーズパイプライン。

**Phase A -- Submit:** 全ジョブを非ブロッキングで投入

**Phase B -- Poll:** 全ジョブの完了を `asyncio.sleep` ベースでポーリング

**Phase C -- Collect:** `asyncio.gather` で全結果を並列収集

### cancel_all(handles) -> None

全投入済みジョブをキャンセルする（`scancel` 経由）。

### poll_jobs(handles) -> list[str]

各ハンドルのジョブステータスを返す。

### wait_all(handles, timeout_sec) -> list[RunResult] [async]

全ジョブの完了またはタイムアウトまで待機し、結果を返す。

---

## ポーリング戦略

| メソッド | 使用条件 | 方式 |
|---------|---------|------|
| `_poll_job` | 同期実行、sacct 利用可 | submitit の `job.state` |
| `_poll_job_squeue` | 同期実行、sacct 利用不可 | `squeue -j <job_id>` |
| `_async_poll_job` | 非同期実行、sacct 利用可 | `job.state` + `asyncio.sleep` |
| `_async_poll_job_squeue` | 非同期実行、sacct 利用不可 | `squeue` + `asyncio.sleep` |

**終端状態:** `COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`

---

## OOM 検知 (_detect_oom)

以下の条件のいずれかで OOM と判定:

1. SLURM ジョブ状態が `OUT_OF_MEMORY`
2. exit_code が `137` または `-9` + stderr に OOM パターン
3. stderr に `MemoryError` / `OutOfMemoryError`

**OOM stderr パターン:** `"MemoryError"`, `"OutOfMemoryError"`, `"Killed"`, `"Cannot allocate memory"`

---

## ヘルパーメソッド

| メソッド | 説明 |
|---------|------|
| `_build_compute_params(compute_config)` | `ComputeConfig` → submitit パラメータ変換 |
| `_check_sacct_available()` | `sacct --version` で利用可否判定 |
| `_cancel_job(job)` | `scancel` でジョブキャンセル |
| `_collect_submitit_logs(job, slurm_log_dir, ...)` | submitit ログを stdout/stderr にコピー |
| `_parse_time_limit(time_str)` | `HH:MM:SS` / `D-HH:MM:SS` を分に変換 |
