# SERA 要件定義書 — Phase 3: 実験実行

> 本ファイルは TASK.md v13.1 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 7. Phase 3：実験実行（Executor）

### 7.1 実行器の要件（必須）
- ローカル実行（MVP必須）
- SLURM実行（ResourceSpecで利用可能なら切替）
- コンテナ実行（ResourceSpecで必須なら、指定イメージで実行）

### 7.2 実験コード生成（具体）

```python
class ExperimentGenerator:
    """探索ノードの実験条件から実行可能な実験コードを生成する"""

    def generate(self, node: SearchNode, problem_spec: ProblemSpec, agent_llm: AgentLLM) -> str:
        """
        1. ProblemSpec.experiment_template をベースに、
           node.experiment_config の値で変数を埋める
        2. テンプレートが不十分な場合、LLM に以下を生成させる：
           - データ読み込み
           - 前処理
           - モデル構築/学習
           - 評価
           - metrics.json 出力
        3. 生成コードは必ず以下を満たす：
           - metrics.json を stdout ではなくファイルに出力
           - seed を受け取り np.random.seed / torch.manual_seed を設定
           - 例外時は stderr にトレースバックを出力し exit(1)
        """
        pass
```

### 7.3 実行インターフェース（プラグイン抽象クラス）

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunResult:
    node_id: str
    success: bool
    exit_code: int
    stdout_path: Path        # runs/<node_id>/stdout.log
    stderr_path: Path        # runs/<node_id>/stderr.log
    metrics_path: Path | None  # runs/<node_id>/metrics.json（成功時のみ）
    artifacts_dir: Path      # runs/<node_id>/artifacts/
    wall_time_sec: float
    seed: int

class Executor(ABC):
    @abstractmethod
    def run(self, node_id: str, script_path: Path, seed: int, timeout_sec: int) -> RunResult:
        """
        実験スクリプトを実行し、結果を返す。

        - 実行前: runs/<node_id>/ および runs/<node_id>/artifacts/ ディレクトリを作成
        - cwd を artifacts/ に設定（実験スクリプトの出力を隔離）
        - stdout/stderr をファイルにリダイレクト
        - タイムアウト超過は RunResult(success=False, exit_code=-9) を返す
        - OOM は RunResult(success=False, exit_code=-7) を返す
        """
        pass

class LocalExecutor(Executor):
    """subprocess.Popen でローカル実行。
    ProblemSpec.language の設定に応じて interpreter / seed 引数を切り替える。"""
    def __init__(
        self,
        work_dir: Path,
        python_executable: str = "python",
        interpreter_command: str | None = None,   # 例: "Rscript", "julia"
        seed_arg_format: str | None = None,        # 例: "-- --seed {seed}"
        allow_internet: bool = True,               # False → プロキシ環境変数を除去
    ): ...

class SlurmExecutor(Executor):
    """sbatch でジョブ投入、sacct で完了待ち"""
    pass

class DockerExecutor(Executor):
    """docker run で隔離実行"""
    pass
```

#### 7.3.1 多言語サポート（LanguageConfig）

`ProblemSpec.language` に `LanguageConfig` を設定すると、Python 以外の言語で実験を実行できる。Agent 実行時は Agent が言語選択・インタプリタ指定を動的に行うため、本設定は主に `StatisticalEvaluator` が `executor.run()` で反復実行する際に使用される。

```yaml
# ProblemSpec.language — デフォルト（未指定時）は Python
language:
  name: "python"                    # プロンプト生成に使用
  interpreter_command: "python"     # LocalExecutor.interpreter_command に渡る
  file_extension: ".py"             # 実験スクリプトの拡張子
  seed_arg_format: "--seed {seed}"  # LocalExecutor.seed_arg_format に渡る
  code_block_tag: "python"          # Markdown コードブロックタグ
```

**R の例**: `interpreter_command: "Rscript"`, `file_extension: ".R"`, `code_block_tag: "r"`
**Julia の例**: `interpreter_command: "julia"`, `file_extension: ".jl"`, `seed_arg_format: "-- --seed {seed}"`

`metrics.json` の出力スキーマ（§7.4）は言語に依存しない。全言語で同一の契約に準拠する。

### 7.4 metrics.json の規約（必須）
Executorは `runs/<node_id>/metrics.json` を出力する。

```json
{
  "primary": {"name": "score", "value": 0.73, "higher_is_better": true},
  "constraints": [
    {"name": "format_valid", "value": 1, "type": "bool", "satisfied": true},
    {"name": "latency_ms", "value": 45.2, "type": "le", "threshold": 100, "satisfied": true}
  ],
  "secondary": [
    {"name": "cost_gpu_sec", "value": 12.3, "lower_is_better": true},
    {"name": "memory_mb", "value": 2048.0, "lower_is_better": true}
  ],
  "raw": {"epoch": 10, "train_loss": 0.32, "val_loss": 0.41},
  "environment": {
    "python": "3.11.5",
    "torch": "2.3.0",
    "cuda": "12.1",
    "gpu": "NVIDIA A100 80GB"
  },
  "seed": 42,
  "wall_time_sec": 125.3
}
```

### 7.5 ストリーミング実行プロトコル

現在の `LocalExecutor.run()` は同期的な `subprocess.Popen` + `proc.wait()` を使用しており、Agent は実行中の出力をリアルタイムに参照できない。本節では非同期の行単位出力キャプチャを追加する。既存の `run()` は変更しない。

#### 7.5.1 ストリーミング型定義（`src/sera/execution/streaming.py` 新規）

```python
class StreamEventType(enum.Enum):
    # Core types (spec §7.5)
    STDOUT = "stdout"
    STDERR = "stderr"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"
    # Extensions (実装済み: metrics polling + keepalive)
    METRICS_UPDATE = "metrics_update"
    HEARTBEAT = "heartbeat"

@dataclass
class StreamEvent:
    event_type: StreamEventType
    data: str = ""                          # 行内容 or 終端サマリ
    elapsed_sec: float = 0.0
    exit_code: int | None = None            # 終端イベントのみ
    metadata: dict[str, Any] = field(default_factory=dict)
    # 終端イベントの metadata:
    #   stdout_tail, stderr_tail, metrics_path, run_result

StreamIterator = AsyncIterator[StreamEvent]
```

`StreamEvent` と `StreamEventType` を `src/sera/execution/__init__.py` の `__all__` に追加する。

#### 7.5.2 Executor 基底クラスへの `run_stream()` 追加

`Executor` ABC に非抽象のデフォルト `run_stream()` を追加する。`SlurmExecutor` / `DockerExecutor` はこれを自動継承する。

```python
class Executor(ABC):
    # run() は変更なし

    async def run_stream(
        self,
        node_id: str,
        script_path: Path,
        seed: int,
        timeout_sec: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """デフォルト実装: run() を run_in_executor で呼び、
        ファイルから tail を読み取り、COMPLETED/TIMEOUT/ERROR イベントを 1 つ yield する。"""
        ...
```

| ID | 要件 |
|----|------|
| S-5.2.1 | デフォルト実装は `asyncio.get_event_loop().run_in_executor` で既存 `run()` を呼ぶ |
| S-5.2.2 | 完了後 `stdout_path`/`stderr_path` 末尾 50 行を読み `metadata` に格納 |
| S-5.2.3 | `exit_code == -9` → `TIMEOUT`, `success == True` → `COMPLETED`, 他 → `ERROR` |
| S-5.2.4 | `run()` が例外を送出した場合は `ERROR` イベントを yield |

#### 7.5.3 LocalExecutor の `run_stream()` オーバーライド

`LocalExecutor` は `asyncio.create_subprocess_exec` + `PIPE` で真のストリーミングを実装する。

```python
class LocalExecutor(Executor):
    # run() は変更なし

    async def run_stream(self, node_id, script_path, seed, timeout_sec=None):
        """asyncio.create_subprocess_exec で行単位ストリーミング。
        asyncio.Queue + 2 reader タスクで stdout/stderr を時系列順にマージ。"""
        ...
```

| ID | 要件 |
|----|------|
| S-5.3.1 | `asyncio.create_subprocess_exec` + `PIPE` で stdout/stderr を非同期取得 |
| S-5.3.2 | `asyncio.Queue` + 2 reader タスクで stdout/stderr を時系列順にマージして yield |
| S-5.3.3 | メモリ内バッファは `deque(maxlen=1000)` で上限付き |
| S-5.3.4 | プロセス終了後 `stdout.log` / `stderr.log` にフル出力を書き込む（§7.3 RunResult 契約維持） |
| S-5.3.5 | OOM 検出ロジックは §7.3 `run()` と同一（exit_code 137/-9 + stderr パターン → `-7`） |
| S-5.3.6 | 終端イベントの `metadata` に `stdout_tail`, `stderr_tail`, `stdout_line_count`, `stderr_line_count`, `metrics_path`, `run_result` を格納 |
| S-5.3.7 | `run()` は一切変更しない |
| S-5.3.8 | シェル構文を含む `interpreter_command` は非対応（基底クラスのデフォルト実装にフォールバック可） |

#### 7.5.4 テスト（`tests/test_execution/test_streaming.py` 新規、`tests/test_execution/test_local_executor.py` 追記）

| テストケース | 概要 |
|-------------|------|
| `TestStreamEvent.test_event_types` | 7 つの `StreamEventType` 値を検証（core 5 + extension 2） |
| `TestStreamEvent.test_event_defaults` | デフォルト値（`data=""`, `exit_code=None`, `metadata={}` 等） |
| `TestStreamEvent.test_metadata_isolation` | 各イベントが独立した `metadata` dict を持つ |
| `TestLocalExecutorRunStream.test_run_stream_successful` | 成功スクリプト → STDOUT イベント + COMPLETED 終端 |
| `TestLocalExecutorRunStream.test_run_stream_timeout` | タイムアウト → TIMEOUT 終端、`exit_code == -9` |
| `TestLocalExecutorRunStream.test_run_stream_error` | エラースクリプト → ERROR 終端、`exit_code != 0` |
| `TestLocalExecutorRunStream.test_run_stream_preserves_artifacts` | `stdout.log`/`stderr.log` ファイルが書かれている |
| `TestLocalExecutorRunStream.test_run_stream_empty_output` | 出力なし → 終端イベント 1 つのみ |
| `TestLocalExecutorRunStream.test_run_stream_metadata_has_run_result` | 終端 `metadata` に `RunResult` が格納 |
| `TestExecutorBaseRunStream.test_default_wraps_run` | 基底クラスのデフォルト実装が `run()` をラップ |
| `TestExecutorBaseRunStream.test_default_handles_run_exception` | `run()` 例外時に ERROR イベント |
| `TestLocalExecutorRunStream.test_run_stream_oom` | stderr に MemoryError + exit 137 → `exit_code == -7` |

### 7.6 実験失敗時のハンドリング
```text
（§7.5 から番号繰り下げ）

1. exit_code != 0 の場合：
   - metrics.json が存在しない → node.status = "failed"、primary = -inf、feasible = false
   - metrics.json が部分的 → 読めるフィールドのみ使用、不足は NaN
2. タイムアウトの場合：
   - node.status = "timeout"、cost = timeout_sec（全消費扱い）
3. OOM の場合：
   - node.status = "oom"、cost = 最大予算（ペナルティ扱い）
4. リトライ：
   - 失敗ノードのリトライは行わない（別の seed で新ノードとして扱う）
```

---
