# AblationRunner / AblationResult

メイン検索ループ完了後に、ベストノードに対する自動アブレーション実験を実行するモジュール。操作変数を一つずつベースライン値にリセットし、主要メトリクスへの影響を測定する。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `AblationRunner` / `AblationResult` | `src/sera/execution/ablation.py` |

## 依存関係

- `sera.execution.executor` (`Executor`) -- 実験実行バックエンド
- `sera.execution.experiment_generator` (`ExperimentGenerator`) -- 実験スクリプト生成
- `sera.search.search_node` (`SearchNode`) -- 一時ノード生成用

---

## AblationResult (dataclass)

単一のアブレーションバリアントの結果。

```python
@dataclass
class AblationResult:
    variable_name: str              # アブレーションされた変数名
    baseline_value: Any             # ベースライン値（リセット後の値）
    original_value: Any             # ベストノードでの元の値
    ablation_config: dict           # アブレーション実験の設定
    metric_value: float | None      # 取得されたメトリクス値（失敗時 None）
    metric_delta: float | None      # ベストとの差分 (best_mu - ablated)
    success: bool = False           # 実行成功フラグ
    error_message: str | None       # エラーメッセージ
```

`metric_delta` が正の値の場合、その変数がメトリクスに正の貢献をしていることを示す。

---

## _get_baseline_value (モジュールレベル関数)

操作変数のベースライン/デフォルト値を決定する。

```python
def _get_baseline_value(variable: Any) -> Any
```

| 変数型 | ベースライン値 |
|--------|-------------|
| `float` | `range[0]`（下限値）。range なしの場合 `0.0` |
| `int` | `range[0]`（下限値）。range なしの場合 `0` |
| `categorical` | `choices[0]`（最初の選択肢） |

---

## generate_ablation_configs (モジュールレベル関数)

ベストノードの設定からアブレーション実験設定を生成する。

```python
def generate_ablation_configs(
    best_config: dict,
    manipulated_variables: list[Any],
) -> list[dict]
```

**処理フロー:**

1. 各操作変数について:
   - `best_config` に存在しない場合: スキップ
   - 既にベースライン値と同じ場合: スキップ
   - `best_config` をディープコピーし、対象変数のみベースラインにリセット
2. 各結果には `variable_name`, `baseline_value`, `original_value`, `config` を含む

---

## AblationRunner

### コンストラクタ

```python
def __init__(
    self,
    executor: Any,                  # 実験実行バックエンド
    experiment_generator: Any,      # 実験スクリプトジェネレータ
    problem_spec: Any,              # ProblemSpec (objective, manipulated_variables)
    execution_spec: Any,            # ExecutionSpec (evaluation パラメータ)
    base_seed: int = 42,            # シード導出の基底値
)
```

### run_ablation(best_node) -> list[AblationResult] [async]

ベストノードに対するアブレーション実験を実行する。

**処理フロー:**

1. ベストノードのバリデーション:
   - `best_node` が `None` → 空リストを返す
   - `experiment_config` が空 → 空リストを返す
   - `manipulated_variables` が未定義 → 空リストを返す
2. `generate_ablation_configs()` でアブレーション設定を生成
3. 各アブレーション設定に対して `_run_single_ablation()` を実行
4. 結果のサマリをログ出力

### _run_single_ablation(best_node, ablation_info, metric_name, best_mu, timeout) -> AblationResult [async]

単一のアブレーション実験を実行する。

**処理フロー:**

1. 一時的な `SearchNode` を作成（仮説: "Ablation: <variable> set to <baseline>"）
2. `experiment_generator.generate()` でスクリプトを生成
3. `_derive_seed()` でシードを導出
4. `executor.run()` で実験を実行
5. 成功時: `metrics.json` からメトリクスを抽出、`metric_delta` を計算
6. 失敗時: エラーメッセージを記録

### _extract_metric(metrics, metric_name) -> float | None [staticmethod]

メトリクス辞書から主要メトリクス値を抽出する。

**対応フォーマット:**

| フォーマット | 例 |
|------------|-----|
| フラット | `{metric_name: 0.95}` |
| ネスト | `{"primary": {"name": "accuracy", "value": 0.95}}` |

### _derive_seed(node_id) -> int

アブレーション実行用の決定論的シード導出。

```python
SHA-256(f"{base_seed}:ablation:{node_id}:0") % 2^31
```

### format_results(results) -> dict[str, float | None]

アブレーション結果を `variable_name -> metric_delta` のマッピングにフォーマットする。
