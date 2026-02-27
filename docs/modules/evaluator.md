# StatisticalEvaluator / Feasibility

Phase 4 の統計的評価を担当するモジュール群のドキュメント。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `Evaluator` (ABC) | `src/sera/evaluation/evaluator.py` |
| `StatisticalEvaluator` | `src/sera/evaluation/statistical_evaluator.py` |
| `check_feasibility` | `src/sera/evaluation/feasibility.py` |

## 依存関係

- `sera.evaluation.evaluator` (`Evaluator` ABC)
- `sera.evaluation.feasibility` (`check_feasibility`)
- `sera.execution.executor` (実験実行バックエンド)
- `sera.execution.experiment_generator` (`ExperimentGenerator`)

---

## Evaluator (ABC)

実験評価の抽象基底クラス。2 フェーズの評価プロトコルを定義する。

```python
class Evaluator(ABC):
    @abstractmethod
    async def evaluate_initial(self, node: Any) -> None: ...

    @abstractmethod
    async def evaluate_full(self, node: Any) -> None: ...
```

- `evaluate_initial`: 全ノードに対して実行される初期（迅速）評価
- `evaluate_full`: top-k ノードに対してのみ実行される完全評価

---

## StatisticalEvaluator

`Evaluator` ABC を実装する具象クラス。複数シードによる実験の平均・標準誤差・下側信頼限界 (LCB) を計算する。

### コンストラクタ

```python
def __init__(
    self,
    executor,                  # 実験実行バックエンド
    experiment_generator=None, # 実験スクリプトジェネレータ
    problem_spec=None,         # ProblemSpec
    execution_spec=None,       # ExecutionSpec
    exec_spec=None,            # ExecutionSpec (execution_spec の別名、互換用)
    base_seed=42,              # シード導出の基底値
    eval_logger=None,          # 評価ログ用 JSONL ロガー
)
```

**注意:** `execution_spec` と `exec_spec` の両方を受け付ける（`self.execution_spec = execution_spec or exec_spec`）。

### 設定値の取得（2 段フォールバック）

各パラメータは以下の優先順位で取得される:

1. `execution_spec.evaluation.*`（evaluation セクション）
2. `execution_spec.search.*`（search セクション、フォールバック）

対象パラメータ: `sequential_eval_initial`, `sequential_eval_topk`, `repeats`, `lcb_coef`

### evaluate_initial(node) -- 初期評価

`sequential_eval_initial`（デフォルト 1）回の実験を実行する。

**処理フロー:**

1. `experiment_generator.generate(node)` でスクリプトパスを取得（未生成の場合のみ生成）
2. 各リピートごとに:
   - シード導出: `SHA-256(f"{base_seed}:{node_id}:{repeat_idx}") % 2^31`
   - `executor.run(node_id, script_path, seed, timeout_sec)` で実験を実行
   - 成功時: `metrics_path` から JSON を読み込み `node.add_metric(metrics)` で追加
   - 失敗時でも `metrics_path` が存在する場合、部分メトリクスの読み込みを試行する。主要メトリクス（`metric_name`）が欠落している場合は `NaN` を設定する
   - `exit_code == -7`（OOM）: `node.status = "oom"` に設定し、`node.total_cost` を `execution_spec.pruning.budget_limit` の `limit` 値（取得可能な場合）に、取得不可の場合は `timeout` に設定して早期リターン
   - `exit_code == -9`（タイムアウト）: `node.status = "timeout"` に設定し、エラーメッセージを記録、`node.total_cost = timeout` に設定して早期リターン
   - その他の失敗: `node.mark_failed(error_msg)` で失敗マークして早期リターン
   - `node.wall_time_sec` に実行時間を加算
3. `update_stats(node, lcb_coef, metric_name)` で統計量を計算
4. `check_feasibility(node, problem_spec)` で実行可能性を判定
5. 評価結果をログ出力（`evaluation_complete` イベントには `mu`, `se`, `lcb`, `ucb`, `n_repeats_done`, `feasible`, `wall_time_sec` 等を含む）

### evaluate_full(node) -- 完全評価

top-k ノードに対して、残りのリピートを実行する。

**処理フロー:**

1. `repeats`（デフォルト 3）から `node.eval_runs` を引いた残回数を計算。0 以下なら早期リターン
2. 既存スクリプトを `experiment.*` glob で検索して再利用
3. 各リピートを実行（シード導出ロジックは `evaluate_initial` と同一）
4. **失敗したリピートは警告ログのみ**（全体の中断はしない）
5. `update_stats()` と `check_feasibility()` を再実行

### _derive_seed(node_id, repeat_idx) -> int

決定論的シード導出。

```python
h = hashlib.sha256(f"{base_seed}:{node_id}:{repeat_idx}".encode()).hexdigest()
return int(h, 16) % (2**31)
```

---

## update_stats (モジュールレベル関数)

ノードの `metrics_raw` から統計量を計算し、ノードを直接更新する。

```python
def update_stats(node, lcb_coef=1.96, metric_name="score") -> None
```

**計算ルール:**

| 条件 | mu | se | lcb | ucb |
|------|-----|-----|------|------|
| `metrics_raw` が空 (n=0) | `None` | `None` | `None` | (設定なし) |
| n=1 | `values[0]` | `inf` | `-inf` | `inf` |
| n>=2 | `mean(values)` | `sqrt(variance / n)` | `mu - lcb_coef * se` | `mu + lcb_coef * se` |

**出力:** `node.mu`, `node.se`, `node.lcb`, `node.ucb` を直接更新する。

**計算式:**

```
mu = sum(values) / n
variance = sum((v - mu)^2 for v in values) / (n - 1)    # 不偏分散
se = sqrt(variance / n)                                   # 標準誤差
lcb = mu - lcb_coef * se                                  # 下側信頼限界
ucb = mu + lcb_coef * se                                  # 上側信頼限界
```

**メトリクス値の抽出:**

`update_stats` はネスト形式とフラット形式の両方をサポートする。各 `metrics_raw` エントリに対して、まず `m["primary"]` キーが dict であり `"value"` を含むかを確認し、該当する場合は `m["primary"]["value"]` を抽出する。該当しない場合は `m[metric_name]` にフォールバックする。dict でない場合やキーが存在しない場合、エントリ自体が数値であればそのまま使用。

**lcb_coef のデフォルト:** 1.96（95% 信頼区間に対応）

---

## check_feasibility (sera.evaluation.feasibility)

ノードの評価結果が ProblemSpec の全制約を満たすか判定する関数。

```python
def check_feasibility(node, problem_spec) -> bool
```

**判定ルール:**

| 条件 | 結果 |
|------|------|
| 制約なし (`constraints` が空) | `True` |
| メトリクスなし (`metrics_raw` が空) | `True`（保守的に feasible と判定） |
| 全制約を満たす | `True` |
| いずれかの制約に違反 | `False` |

**制約タイプ:**

| 制約タイプ | 違反条件 |
|-----------|---------|
| `"bool"` | `not bool(value)` が True |
| `"ge"` | `value < threshold - epsilon` |
| `"le"` | `value > threshold + epsilon` |

**チェック方式:**

- **全メトリクス実行にわたってチェック**: `metrics_raw` の各エントリに対して各制約を検査
- 制約名のメトリクスが報告されていない場合: スキップ（充足と仮定）
- `epsilon` のデフォルト値: `0.0`

### is_topk (StatisticalEvaluator の静的メソッド)

```python
@staticmethod
def is_topk(node, all_nodes, k) -> bool
```

- `node.lcb` が None の場合: `True`（未評価ノードは常に top-k 扱い）
- 評価済み・feasible なノードを LCB 降順でソートし、上位 k 件の ID に含まれるかを返す
