# SERA 要件定義書 — Phase 3: 実験実行

> 本ファイルは TASK.md v13.0 を分割したものである。目次は [README.md](./README.md) を参照。

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

        - 実行前: runs/<node_id>/ ディレクトリを作成
        - stdout/stderr をファイルにリダイレクト
        - タイムアウト超過は RunResult(success=False, exit_code=-9) を返す
        - OOM は RunResult(success=False, exit_code=-7) を返す
        """
        pass

class LocalExecutor(Executor):
    """subprocess.Popen でローカル実行"""
    pass

class SlurmExecutor(Executor):
    """sbatch でジョブ投入、sacct で完了待ち"""
    pass

class DockerExecutor(Executor):
    """docker run で隔離実行"""
    pass
```

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

### 7.5 実験失敗時のハンドリング
```text
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
