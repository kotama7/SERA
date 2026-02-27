# BootstrapEvaluator / bootstrap_update_stats

ブートストラップリサンプリングによる統計的評価モジュールのドキュメント。`StatisticalEvaluator` の代替として、解析的な平均/SE/LCB の代わりにブートストラップ法で信頼限界を計算する。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `BootstrapEvaluator` | `src/sera/evaluation/bootstrap_evaluator.py` |
| `bootstrap_update_stats` | `src/sera/evaluation/bootstrap_evaluator.py` |
| `_percentile` | `src/sera/evaluation/bootstrap_evaluator.py` |

## 依存関係

- `sera.evaluation.evaluator` (`Evaluator` ABC) -- 評価の抽象基底クラス
- `sera.evaluation.feasibility` (`check_feasibility`) -- 実行可能性制約チェック
- `random.Random` -- ブートストラップリサンプリングの乱数生成
- `math` -- `sqrt`, `floor`, `ceil`

---

## 定数

| 定数 | 値 | 説明 |
|------|-----|------|
| `DEFAULT_B` | `1000` | ブートストラップリサンプル数のデフォルト |
| `DEFAULT_ALPHA` | `0.05` | 有意水準のデフォルト（95% 信頼区間に対応） |

---

## BootstrapEvaluator

`Evaluator` ABC を実装する具象クラス。実験実行部分は `StatisticalEvaluator` と同一であり、統計量の計算方法のみが異なる。

### 有効化条件

`ExecutionSpec` の `evaluation.bootstrap=True` が設定されている場合に `research_cmd.py` で使用される。

### コンストラクタ

```python
def __init__(
    self,
    executor: Any,
    experiment_generator: Any = None,
    problem_spec: Any = None,
    execution_spec: Any = None,
    exec_spec: Any = None,
    base_seed: int = 42,
    eval_logger: Any = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
)
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `executor` | `Executor` | -- | 実験実行バックエンド |
| `experiment_generator` | `ExperimentGenerator` | `None` | 実験スクリプトジェネレータ |
| `problem_spec` | `ProblemSpec` | `None` | 問題仕様（目的関数、制約） |
| `execution_spec` | `ExecutionSpec` | `None` | 実行仕様（評価パラメータ） |
| `exec_spec` | `Any` | `None` | `execution_spec` の別名（互換用） |
| `base_seed` | `int` | `42` | シード導出の基底値 |
| `eval_logger` | `Any` | `None` | 評価ログ用 JSONL ロガー |
| `n_bootstrap` | `int` | `1000` | ブートストラップリサンプル数 |
| `alpha` | `float` | `0.05` | 有意水準（0.05 で 95% CI） |

**注意:** `execution_spec` と `exec_spec` の両方を受け付ける（`self.execution_spec = execution_spec or exec_spec`）。

### 設定値の取得（2 段フォールバック）

`StatisticalEvaluator` と同一パターン:

1. `execution_spec.evaluation.*`（evaluation セクション）
2. `execution_spec.search.*`（search セクション、フォールバック）

対象パラメータ: `sequential_eval_initial`, `repeats`, `timeout_per_run_sec`

---

### evaluate_initial(node) -- 初期評価

`sequential_eval_initial`（デフォルト 1）回の実験を実行する非同期メソッド。

**処理フロー:**

1. `experiment_generator.generate(node)` でスクリプトパスを取得
2. 各リピートごとに:
   - シード導出: `_derive_seed(node_id, eval_runs)`
   - `executor.run(node_id, script_path, seed, timeout_sec)` で実験を実行
   - 成功時: `metrics_path` から JSON を読み込み `node.add_metric(metrics)` で追加
   - 失敗時の部分メトリクス回収: `metrics_path` が存在すれば読み込みを試行。主要メトリクスが欠落している場合は `NaN` を設定
   - `exit_code == -7`（OOM）: `node.status = "oom"` に設定して早期リターン
   - `exit_code == -9`（タイムアウト）: `node.status = "timeout"` に設定して早期リターン
   - その他の失敗: `node.mark_failed(error_msg)` で失敗マークして早期リターン
   - `node.wall_time_sec` に実行時間を加算
3. `bootstrap_update_stats(node, metric_name, n_bootstrap, alpha, rng_seed=base_seed)` でブートストラップ統計量を計算
4. `check_feasibility(node, problem_spec)` で実行可能性を判定
5. 評価結果をログ出力（`evaluator: "bootstrap"` を含む）

### evaluate_full(node) -- 完全評価

top-k ノードに対して、残りのリピートを実行する非同期メソッド。

**処理フロー:**

1. `repeats`（デフォルト 3）から `node.eval_runs` を引いた残回数を計算。0 以下なら早期リターン
2. 既存スクリプトを `runs/<node_id>/experiment.*` glob で検索して再利用。存在しない場合は再生成
3. 各リピートを実行（シード導出ロジックは `evaluate_initial` と同一）
4. **失敗したリピートは警告ログのみ**（全体の中断はしない）
5. `bootstrap_update_stats()` と `check_feasibility()` を再実行
6. 完全評価結果をログ出力

### _derive_seed(node_id, repeat_idx) -> int

`StatisticalEvaluator` と同一の決定論的シード導出。

```python
h = hashlib.sha256(f"{base_seed}:{node_id}:{repeat_idx}".encode()).hexdigest()
return int(h, 16) % (2**31)
```

### is_topk(node, all_nodes, k) -> bool (静的メソッド)

ノードが top-k に含まれるかを判定する。

| 条件 | 結果 |
|------|------|
| `node.lcb` が None | `True`（未評価ノードは常に top-k 扱い） |
| 通常 | 評価済み・feasible ノードを LCB 降順ソートし、上位 k 件に含まれるか |

---

## bootstrap_update_stats (モジュールレベル関数)

ブートストラップリサンプリングにより `mu`, `se`, `lcb`, `ucb` を計算し、ノードを直接更新する。

```python
def bootstrap_update_stats(
    node: Any,
    metric_name: str = "score",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng_seed: int | None = None,
) -> None
```

### 計算アルゴリズム

1. `node.metrics_raw` から `metric_name` キーの値を `float` として抽出
2. 値が空の場合: `mu = se = lcb = None` を設定して返す

**n=1 の場合:**

```
node.mu = values[0]
node.se = inf
node.lcb = -inf
node.ucb = inf
```

**n>=2 の場合:**

```
1. mu = sum(values) / n（元データの平均）
2. B 回のブートストラップリサンプリング:
   各リサンプルで n 個の値を復元抽出し、平均を計算
3. ブートストラップ平均をソート
4. LCB = percentile(alpha/2 * 100) of bootstrap_means
5. UCB = percentile((1 - alpha/2) * 100) of bootstrap_means
6. SE = sqrt(不偏分散 of bootstrap_means)
```

**具体例（alpha=0.05, 95% CI の場合）:**

- LCB = 2.5 パーセンタイル
- UCB = 97.5 パーセンタイル

### SE の計算

```python
bm_mean = sum(bootstrap_means) / len(bootstrap_means)
bm_var = sum((x - bm_mean) ** 2 for x in bootstrap_means) / (len(bootstrap_means) - 1)
node.se = math.sqrt(bm_var)
```

ブートストラップ平均の標準偏差（不偏分散の平方根）を SE として使用する。

### メトリクス値の抽出

`StatisticalEvaluator.update_stats` と同一のロジック:

- `metrics_raw` の各エントリが dict の場合: `metric_name` キーの値を `float` として抽出
- `StatisticalEvaluator` ではネストされた `m["primary"]["value"]` 形式とフラットな `m[metric_name]` 形式の両方をサポートするが、`BootstrapEvaluator` はフラットな `m[metric_name]` 形式のみをサポートする
- エントリ自体が数値（`int` / `float`）の場合: そのまま使用

### 乱数の再現性

`random.Random(rng_seed)` を使用して決定論的なリサンプリングを保証する。同一の `rng_seed` と入力データに対して常に同じ結果を返す。

---

## _percentile (モジュールレベル関数)

ソート済みデータから指定パーセンタイルを線形補間で計算する。numpy のデフォルトメソッドと同等。

```python
def _percentile(sorted_data: list[float], pct: float) -> float
```

### 計算式

```
rank = (pct / 100.0) * (n - 1)
lo = floor(rank)
hi = ceil(rank)
frac = rank - lo
result = sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac
```

### エッジケース

| 条件 | 戻り値 |
|------|--------|
| データが空 | `NaN` |
| データが 1 件 | `sorted_data[0]` |
| `lo == hi` | `sorted_data[lo]`（補間なし） |

---

## StatisticalEvaluator との比較

| 項目 | StatisticalEvaluator | BootstrapEvaluator |
|------|---------------------|-------------------|
| LCB 計算方法 | `mu - lcb_coef * se`（解析的） | パーセンタイル法（リサンプリング） |
| SE 計算方法 | `sqrt(不偏分散 / n)` | ブートストラップ平均の標準偏差 |
| UCB | `mu + lcb_coef * se` | `percentile((1-alpha/2)*100)` |
| 分布仮定 | 正規分布（暗黙的） | ノンパラメトリック |
| 追加パラメータ | `lcb_coef` | `n_bootstrap`, `alpha` |
| 計算コスト | O(n) | O(n * B) |
| 実験実行部分 | 同一 | 同一 |
| シード導出 | 同一 | 同一 |
