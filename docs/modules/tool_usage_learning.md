# ToolCallRecord / ToolUsageStats / compute_reward_tool_aware

ツール使用学習モジュールのドキュメント。ツール呼び出し記録の管理、統計トラッキング、報酬調整を担当する。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `ToolCallRecord` | `src/sera/learning/tool_usage_learning.py` |
| `ToolUsageStats` | `src/sera/learning/tool_usage_learning.py` |
| `compute_reward_tool_aware` | `src/sera/learning/tool_usage_learning.py` |
| `compute_reward_tool_aware_dispatch` | `src/sera/learning/reward.py` |

## 依存関係

- `sera.learning.reward` (`register_reward_method`, `compute_reward_mt_grpo`) -- 報酬メソッドレジストリおよびベース報酬計算
- `collections.defaultdict` -- ツール別統計の内部管理

---

## ToolCallRecord (dataclass)

単一のツール呼び出しの記録を保持するデータクラス。PPO 報酬計算および HiPER アドバンテージ推定のためのメタデータを格納する。

```python
@dataclass
class ToolCallRecord:
    tool_name: str
    phase: str = ""
    node_id: str = ""
    success: bool = True
    latency_sec: float = 0.0
    result_quality: float = 1.0
```

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `tool_name` | `str` | -- | 呼び出されたツール名 |
| `phase` | `str` | `""` | ツールが呼び出されたフェーズ（例: `"phase0"`, `"phase2"`） |
| `node_id` | `str` | `""` | ツール呼び出しをトリガーした SearchNode の ID |
| `success` | `bool` | `True` | ツール実行が成功したか |
| `latency_sec` | `float` | `0.0` | ツール実行の壁時計時間（秒） |
| `result_quality` | `float` | `1.0` | 結果の品質スコア `[0, 1]`。成功時は 1.0、失敗時は 0.0 がデフォルト。ドメイン固有の評価器で洗練可能 |

### to_dict() -> dict[str, Any]

JSON ストレージ用にシリアライズする。全フィールドを dict として返す。

```python
{
    "tool_name": "read_file",
    "phase": "phase2",
    "node_id": "abc123",
    "success": True,
    "latency_sec": 0.15,
    "result_quality": 1.0
}
```

### from_dict(cls, d) -> ToolCallRecord

dict からデシリアライズする。未知のキーは無視される。

```python
valid_fields = {"tool_name", "phase", "node_id", "success", "latency_sec", "result_quality"}
```

---

## ToolUsageStats

ツール別の成功率・平均レイテンシ・品質を追跡するクラス。内部で `defaultdict` を使用して効率的にツール別統計を管理する。

### 用途

- 報酬計算（ツール効率ボーナス/ペナルティ）
- モニタリング（ダッシュボード、ログ）
- 適応的ツール選択（将来: 信頼性/速度の高いツールを優先）

### コンストラクタ

```python
def __init__(self) -> None
```

**内部状態:**

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `_total_calls` | `defaultdict[str, int]` | ツール別の合計呼び出し回数 |
| `_success_calls` | `defaultdict[str, int]` | ツール別の成功呼び出し回数 |
| `_total_latency` | `defaultdict[str, float]` | ツール別の合計レイテンシ |
| `_quality_sum` | `defaultdict[str, float]` | ツール別の品質スコア合計 |

### メソッド

#### record(record: ToolCallRecord) -> None

1 件のツール呼び出しを記録する。

- `_total_calls[name] += 1`
- 成功時: `_success_calls[name] += 1`
- `_total_latency[name] += latency_sec`
- `_quality_sum[name] += result_quality`

#### record_batch(records: list[ToolCallRecord]) -> None

複数のツール呼び出しを一括記録する。内部で `record()` を各レコードに対して呼び出す。

#### success_rate(tool_name: str) -> float

特定ツールの成功率を返す。

| 条件 | 戻り値 |
|------|--------|
| 呼び出し記録なし (total=0) | `1.0` |
| 通常 | `success_calls / total_calls` |

#### average_latency(tool_name: str) -> float

特定ツールの平均レイテンシ（秒）を返す。

| 条件 | 戻り値 |
|------|--------|
| 呼び出し記録なし (total=0) | `0.0` |
| 通常 | `total_latency / total_calls` |

#### average_quality(tool_name: str) -> float

特定ツールの平均品質スコアを返す。

| 条件 | 戻り値 |
|------|--------|
| 呼び出し記録なし (total=0) | `1.0` |
| 通常 | `quality_sum / total_calls` |

#### overall_success_rate() -> float

全ツール横断の成功率を返す。

| 条件 | 戻り値 |
|------|--------|
| 全体の呼び出し記録なし | `1.0` |
| 通常 | `sum(success_calls) / sum(total_calls)` |

#### overall_average_latency() -> float

全ツール横断の平均レイテンシを返す。呼び出し記録がない場合は `0.0`。

### プロパティ

| プロパティ | 型 | 説明 |
|-----------|-----|------|
| `total_calls` | `int` | 全ツールの合計呼び出し回数 |
| `tool_names` | `list[str]` | 記録されたツール名のソート済みリスト |

#### summary() -> dict[str, Any]

全ツール使用統計のサマリー dict を返す。

```python
{
    "total_calls": 42,
    "overall_success_rate": 0.9524,
    "overall_average_latency_sec": 0.2345,
    "per_tool": {
        "read_file": {
            "total_calls": 20,
            "success_rate": 1.0,
            "average_latency_sec": 0.15,
            "average_quality": 1.0,
        },
        "execute_experiment": {
            "total_calls": 22,
            "success_rate": 0.9091,
            "average_latency_sec": 0.31,
            "average_quality": 0.85,
        },
    }
}
```

各数値は `round(..., 4)` で小数点以下 4 桁に丸められる。

#### reset() -> None

全ての記録済み統計をクリアする。4 つの内部 `defaultdict` を全て `.clear()` する。

---

## compute_reward_tool_aware

ベース報酬をツール使用の効率性に基づいて調整するモジュールレベル関数。

```python
def compute_reward_tool_aware(
    base_reward: float,
    tool_records: list[ToolCallRecord],
    tool_call_budget: int = 20,
    efficiency_coef: float = 0.01,
    failure_penalty_coef: float = 0.05,
) -> float
```

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `base_reward` | `float` | -- | `compute_reward_mt_grpo` 等からのベース報酬 |
| `tool_records` | `list[ToolCallRecord]` | -- | ノードのライフタイム中のツール呼び出し記録 |
| `tool_call_budget` | `int` | `20` | 最大許容ツール呼び出し数（効率の正規化に使用） |
| `efficiency_coef` | `float` | `0.01` | 効率ボーナスの係数。大きいほどツール呼び出し削減を強く報酬する |
| `failure_penalty_coef` | `float` | `0.05` | 失敗ペナルティの係数。大きいほどツール失敗を強く罰する |

### 計算式

```
total_tool_calls = len(tool_records)
tool_success_rate = successful / total_tool_calls

# 効率ボーナス: バジェットに対してツール呼び出しが少ないほど高い
efficiency_bonus = efficiency_coef * (1.0 - total_tool_calls / tool_call_budget)

# 失敗ペナルティ: 成功率が低いほど高い
failure_penalty = failure_penalty_coef * (1.0 - tool_success_rate)

adjusted_reward = base_reward + efficiency_bonus - failure_penalty
```

### エッジケース

| 条件 | 動作 |
|------|------|
| `tool_records` が空 | `base_reward` をそのまま返す |
| `tool_call_budget <= 0` | `efficiency_bonus = 0.0` |
| 全ツール成功 (`success_rate=1.0`) | `failure_penalty = 0.0` |
| 全ツール失敗 (`success_rate=0.0`) | `failure_penalty = failure_penalty_coef` |

### 計算例

```python
# 20 バジェットに対して 2 回のツール呼び出し（全て成功）
records = [
    ToolCallRecord(tool_name="read_file", success=True),
    ToolCallRecord(tool_name="read_file", success=True),
]
# efficiency_bonus = 0.01 * (1.0 - 2/20) = 0.01 * 0.9 = 0.009
# failure_penalty = 0.05 * (1.0 - 1.0) = 0.0
# result = 1.0 + 0.009 - 0.0 = 1.009
compute_reward_tool_aware(1.0, records, tool_call_budget=20)  # -> 1.009
```

---

## reward.py との統合

`compute_reward_tool_aware` は `reward.py` に `"tool_aware"` メソッドとして登録される。

```python
@register_reward_method("tool_aware")
def compute_reward_tool_aware_dispatch(
    node, plan_spec, exec_spec, kl_divergence=0.0, **kw
) -> float
```

**処理フロー:**

1. `compute_reward_mt_grpo()` でベース報酬を計算
2. `kw["tool_records"]` からツール呼び出し記録を取得
3. ツール記録がない場合はベース報酬をそのまま返す
4. `plan_spec.reward` から設定パラメータを取得:
   - `tool_call_budget`（デフォルト 20）
   - `efficiency_coef`（デフォルト 0.01）
   - `failure_penalty_coef`（デフォルト 0.05）
5. `compute_reward_tool_aware()` で最終報酬を計算

### PPORolloutV3 との連携

PPO ロールアウトデータ (`PPORolloutV3`) は `tool_trajectory` フィールドを持ち、ノードのライフタイム中のツール呼び出し記録を `ToolCallRecord` のリストとして運搬する。`SearchManager._evaluate_node()` でノード評価後に `tool_records` として `ppo_buffer` に追加される。
