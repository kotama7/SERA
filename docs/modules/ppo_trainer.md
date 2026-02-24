# PPOTrainer / RewardComputer / PPORollout

Phase 5 の PPO 学習（LoRA のみ更新）を担当するモジュール群のドキュメント。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `PPOTrainer` | `src/sera/learning/ppo_trainer.py` |
| `compute_reward` | `src/sera/learning/reward.py` |
| `PPORollout` / `PPORolloutV2` | `src/sera/learning/rollout.py` |
| `TurnRewardEvaluator` | `src/sera/learning/turn_reward.py` |
| `HierarchicalAdvantageEstimator` | `src/sera/learning/hierarchical_ppo.py` |

## 依存関係

- `trl.trainer.utils` (`entropy_from_logits`) -- エントロピー計算
- `accelerate` (`Accelerator`) -- デバイス管理・勾配操作
- `peft` (`get_peft_model_state_dict`) -- アダプタ重み抽出
- `torch` / `numpy` -- テンソル演算
- `sera.learning.rollout` (`PPORollout`)
- `sera.utils.logging` (`JsonlLogger`)

---

## PPOTrainer

LoRA パラメータのみを PPO で更新するトレーナークラス。

### コンストラクタ

```python
def __init__(self, exec_spec, model_spec, lineage_manager, log_path: Path, plan_spec=None)
```

`plan_spec` パラメータ（デフォルト `None`）を受け取り、報酬手法に応じた Advantage 計算のルーティングに使用する。`None` の場合は従来の `outcome_rm` 動作となる。

**抽出されるハイパーパラメータ** (`exec_spec.learning` から取得、デフォルト値付き):

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `clip_range` | 0.2 | PPO クリッピング範囲 |
| `lr` | 1e-4 | AdamW 学習率 |
| `batch_size` | 4 | バッチサイズ |
| `mini_batch_size` | 2 | ミニバッチサイズ |
| `epochs_per_update` | 4 | 1 更新あたりのエポック数 |
| `gamma` | 0.99 | 割引率 |
| `gae_lambda` | 0.95 | GAE のラムダ |
| `kl_control` | True | 適応 KL 制御の有効化 |
| `kl_coef` | 0.01 | KL ペナルティ係数 |
| `kl_target` | 0.01 | KL ターゲット値 |
| `entropy_coef` | 0.01 | エントロピーボーナス係数 |
| `max_grad_norm` | 1.0 | 勾配クリッピングの最大ノルム |
| `value_loss_coef` | 0.5 | 価値関数ロスの係数 |
| `ppo_trigger_interval` | 5 | PPO 更新トリガー間隔 |

### should_update(n_evaluated) -> bool

PPO 更新をトリガーすべきか判定する。

| 条件 | 結果 |
|------|------|
| `n_evaluated < 2` | `False` |
| `n_evaluated % ppo_trigger_interval == 0` | `True` |
| `_steps_since_improvement >= plateau_patience` | `True`（プラトー検知） |

`plateau_patience` は `exec_spec.termination.plateau_patience`（デフォルト 10）から取得。

### notify_step(current_best_lcb)

現在の最良 LCB を通知してプラトー検知を更新する。

- `current_best_lcb > _best_lcb` の場合: `_best_lcb` を更新し `_steps_since_improvement = 0`
- そうでない場合: `_steps_since_improvement += 1`

### update(rollouts, agent_llm, specs) -> dict

PPO 更新サイクルを 1 回実行する非同期メソッド。

- `_mock_fn` が設定されている場合はそれを呼び出し、結果をログに記録して返す
- そうでない場合は `_ppo_update_impl()` を実行

**戻り値:** `mean_reward`, `kl_divergence`, `policy_loss`, `value_loss`, `entropy`, `delta_norm_l2`, `kl_coef_current` を含む dict。

### _compute_advantages_for_method(rollouts, turn_rewards_map)

報酬手法（`plan_spec.reward.method`）に応じた Advantage 計算のルーティングを行う。

| メソッド | 処理 |
|---------|------|
| `hiper` | `HierarchicalAdvantageEstimator.compute_hierarchical_advantages()` を使用。3 層（switch/high/low level）の階層的 Advantage 分解を実行 |
| `mt_grpo` | 従来の `_compute_gae`（報酬値にターン報酬が反映済み） |
| `outcome_rm`（デフォルト） | 従来の `_compute_gae` |

`plan_spec` が `None` の場合は常に `_compute_gae` にフォールバックする。

### _ppo_update_impl (実際の PPO 実装)

**処理フロー:**

1. **vLLM エンジンのスリープ**: `agent_llm._vllm_engine` が存在する場合、`.sleep()` で GPU メモリを解放
2. **価値推定の取得**: 各ロールアウトの `value` が 0.0 の場合、`agent_llm.get_value(prompt, response)` で取得
3. **Advantage 計算**: `_compute_advantages_for_method()` を呼び出し、手法に応じた Advantage を計算
   - `outcome_rm` / `mt_grpo`: `_compute_gae` で単一ステップエピソードとして処理（`advantage = reward - value`, `returns = reward`）
   - `hiper`: `HierarchicalAdvantageEstimator` で 3 層階層的 Advantage を計算
4. **LoRA パラメータの特定**:
   ```python
   lora_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "lora" in n.lower()]
   ```
   LoRA パラメータが見つからない場合はスキップ
5. **Accelerator と AdamW の準備**:
   ```python
   accelerator = Accelerator()
   optimizer = torch.optim.AdamW(lora_params, lr=self.lr)
   model, optimizer = accelerator.prepare(model, optimizer)
   ```
6. **更新前重みのスナップショット**: `peft.get_peft_model_state_dict()` で取得
7. **ミニバッチ PPO ループ** (`epochs_per_update` エポック):
   - インデックスをランダムシャッフル
   - ミニバッチサイズ `mini_batch_size` で分割
   - 各ミニバッチで:
     - フォワードパス: `agent_llm.get_log_probs_with_logits(prompt, response)` で新しい log-prob とロジットを取得
     - アドバンテージの正規化: `std > 1e-8` の場合のみ
     - **PPO クリッピングサロゲート**:
       ```
       ratio = exp(new_lp - old_lp)
       surr1 = ratio * advantages
       surr2 = clamp(ratio, 1-clip_range, 1+clip_range) * advantages
       policy_loss = -min(surr1, surr2).mean()
       ```
     - **価値関数ロス**: `0.5 * MSE(values, returns)`
     - **エントロピー**: `trl.trainer.utils.entropy_from_logits(logits)` を使用
     - **総合ロス**: `policy_loss + value_loss_coef * value_loss - entropy_coef * entropy`
     - 勾配クリッピング: `accelerator.clip_grad_norm_(lora_params, max_grad_norm)`
8. **デルタノルムの計算**: `delta_norm_L2 = sqrt(sum((post[k] - pre[k]).norm()^2 for k in pre))`
9. **適応 KL 係数の調整**:
   - `mean_kl > kl_target * 1.5` の場合: `kl_coef *= 2.0`
   - `mean_kl < kl_target / 1.5` の場合: `kl_coef /= 2.0`
10. **vLLM エンジンのウェイク**: `finally` ブロックで `.wake()` を呼び出す
11. **ログ出力**: `ppo_log.jsonl` に結果を記録

### set_mock(fn)

テスト用のモック関数を注入する。`fn(rollouts) -> dict` 形式のコーラブルを受け付け、設定後は `update()` が実際の PPO を実行せずモックを呼び出す。

---

## PPORollout / PPORolloutV2

1 つの探索ノードの実験サイクルを 1 エピソードとして表す dataclass。

```python
@dataclass
class PPORollout:
    node_id: str           # 探索ノード ID
    prompt: str            # LLM 入力（仮説生成プロンプト）
    response: str          # LLM 出力（仮説 + 実験設計 JSON）
    log_prob: float        # 出力トークン列の対数確率
    reward: float          # 計算された報酬
    value: float           # 価値関数の推定値
    advantage: float = 0.0 # GAE で計算されたアドバンテージ（後から設定）
    returns: float = 0.0   # 割引リターン（後から設定）
```

### PPORolloutV2（拡張版）

MT-GRPO/HiPER 用に `turn_rewards` フィールドを追加したサブクラス。

```python
@dataclass
class PPORolloutV2(PPORollout):
    turn_rewards: dict[str, float] = field(default_factory=dict)
    # 例: {"phase0": 0.9, "phase3": 1.0, "phase4": 0.6}
```

`SearchManager` は `turn_reward_evaluator` が有効な場合に `PPORolloutV2` を使用し、そうでない場合は `PPORollout` を使用する。

---

## compute_reward — 報酬ディスパッチ (sera.learning.reward)

探索木ノードに対するスカラー報酬を計算する関数。**レジストリパターン**により、`plan_spec.reward.method` に応じて適切な報酬計算関数にディスパッチする。

```python
def compute_reward(node, plan_spec, exec_spec, kl_divergence=0.0, **kw) -> float
```

**注意:** 関数シグネチャの引数順序は `(node, plan_spec, exec_spec, kl_divergence)` であり、`exec_spec` は第 3 引数。`**kw` により `turn_rewards` 等の追加引数を受け取る。

### 報酬手法レジストリ

```python
_REWARD_METHODS: dict[str, Callable[..., float]] = {}

def register_reward_method(name: str) -> Callable:
    """報酬手法をレジストリに登録するデコレータ"""
    ...
```

3 つの手法が登録されている:

| メソッド名 | 関数 | 説明 |
|-----------|------|------|
| `outcome_rm` | `compute_reward_outcome_rm` | 従来の報酬計算（デフォルト） |
| `mt_grpo` | `compute_reward_mt_grpo` | ターンレベル報酬の重み付き和 |
| `hiper` | `compute_reward_hiper` | HiPER（報酬値は `mt_grpo` に委譲） |

### ディスパッチロジック

```python
def compute_reward(node, plan_spec, exec_spec, kl_divergence=0.0, **kw) -> float:
    method = getattr(getattr(plan_spec, "reward", None), "method", "outcome_rm")
    fn = _REWARD_METHODS.get(method, compute_reward_outcome_rm)
    return fn(node, plan_spec, exec_spec, kl_divergence, **kw)
```

未知のメソッド名は `outcome_rm` にフォールバックする。

### Outcome RM（outcome_rm）

```
R = primary_value
    - constraint_penalty * num_violated_constraints
    - lambda_cost * normalized_cost
    - kl_coef * kl_divergence
```

### MT-GRPO（mt_grpo）

```
R = Σ(weight_t * turn_reward_t)
    - constraint_penalty * num_violated_constraints
    - lambda_cost * normalized_cost
    - kl_coef * kl_divergence
```

`turn_rewards` が `**kw` で渡されない場合、`outcome_rm` にフォールバックする。

### HiPER（hiper）

報酬値自体は `mt_grpo` と同一。HiPER 固有の処理は `HierarchicalAdvantageEstimator` での Advantage 分解側で行われる。

### 異常時の報酬

`status` が `"failed"` / `"timeout"` / `"oom"` の場合、`metrics_raw` が空の場合、`mu` が None の場合: `_FAILURE_REWARD = -100.0` を返す（全手法共通）。

### 各項の計算

**primary_value** (`_extract_primary_value`):
- `metrics_raw` 内で `primary == True` または `name == "primary"` のエントリの `value` を取得
- `direction == "minimize"` の場合は値を負にする
- 該当エントリがない場合は `node.mu` にフォールバック

**constraint_penalty**:
- 係数: `plan_spec.reward.constraint_penalty`（デフォルト 10.0）
- 違反数: `metrics_raw` 内の `constraint_violated == True` のエントリ数。`node.feasible == False` の場合は最低 1

**normalized_cost** (`_normalize_cost`):
- `min(cost / budget_limit, 1.0)` で [0, 1] に正規化
- `budget_limit`: `termination.max_wall_time_hours * 3600`（フォールバック: `max_wallclock_hours * 3600`、デフォルト 14400 秒）

**lambda_cost**: `exec_spec.search.lambda_cost`（デフォルト 0.1）

**kl_coef**: `plan_spec.reward.kl_coef_in_reward`（デフォルト 0.01）

---

## TurnRewardEvaluator (sera.learning.turn_reward)

Phase 毎のターンレベル報酬を評価するクラス。MT-GRPO / HiPER で使用される。

```python
class TurnRewardEvaluator:
    def __init__(self, turn_reward_spec):
        ...

    def evaluate_all(self, node, parent, all_nodes) -> dict[str, float]:
        """全 Phase の報酬を計算して返す。"""
        ...
```

### 登録済み Phase 評価器

| 評価器名 | 説明 | 出力範囲 |
|---------|------|---------|
| `citation_relevance` | 仮説内の先行研究参照キーワードの存在 | 0.0 - 1.0 |
| `hypothesis_novelty` | 既存ノード仮説との単語レベル類似度の逆数 | 0.0 - 1.0 |
| `code_executability` | ノードが `"evaluated"` ステータスか | 0.0 or 1.0 |
| `metric_improvement` | 親ノード比のメトリクス改善率 | 0.0 - 1.0 |
| `paper_score_delta` | PaperScore 改善幅（プレースホルダ、常に 0.0） | 0.0 |

未知の評価器名は 0.0 を返す。

---

## HierarchicalAdvantageEstimator (sera.learning.hierarchical_ppo)

HiPER の 3 層階層的 Advantage 分解を実装するクラス。`reward.method == "hiper"` の場合に `PPOTrainer` から使用される。

```python
class HierarchicalAdvantageEstimator:
    def __init__(self, hiper_config):
        ...

    def compute_hierarchical_advantages(self, rollouts, turn_rewards_map=None):
        """各ロールアウトの advantage と returns を設定する。"""
        ...
```

### 3 層 Advantage 分解

| 層 | 計算 | 意味 |
|----|------|------|
| High Level | `reward - value` | 全体的なパフォーマンス推定 |
| Switch Level | `-variance(turn_rewards)` | Phase 間バランスのペナルティ。ターン報酬のばらつきが大きいほど負 |
| Low Level | `mean(turn_rewards) - value` | Phase 平均報酬に基づくパフォーマンス |

**最終 Advantage**:
```
A = high_weight * A_high + switch_weight * A_switch + low_weight * A_low
```

ターン報酬が存在しない場合、switch/low level は 0 にフォールバックし、high level のみで計算される。
