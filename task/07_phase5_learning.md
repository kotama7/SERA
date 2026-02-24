# SERA 要件定義書 — Phase 5: 学習（PPO + LoRA）

> 本ファイルは TASK.md v12.4 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 9. Phase 5：学習（PPOのみ、LoRA-only、差分継承）

### 9.1 学習の位置づけ（必須）
- 学習はテーマではなく **実行エージェント**の専門化
- 学習対象は LoRA パラメータのみ（ベースモデル凍結）
- 具体的には、エージェントLLMの「仮説生成」「実験設計」「コード生成」能力を研究ドメインに特化させる

### 9.1.1 外部探索木とLoRA系譜木の同期ルール（必須）

二重木構造（設計原則2）の同期は以下のルールに従う：

```text
外部探索木（SearchNode）          LoRA系譜木（adapter_node）
━━━━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━━━━━
root_child_1 (depth=0)  ───────► adapter_root (zero-init LoRA)
  ├─ child_A (depth=1)  ───────► adapter_root（同一。PPO未実行）
  │    └─ child_A1      ───────► adapter_A1（PPO更新でΔ生成）
  ├─ child_B (depth=1)  ───────► adapter_root（同一。PPO未実行）
  └─ child_C (depth=1)  ───────► adapter_C（PPO更新でΔ生成）
       └─ child_C1      ───────► adapter_C（同一。PPO未実行）
            └─ child_C1a ───────► adapter_C1a（PPO更新でΔ生成）
```

```python
def sync_adapter_assignment(search_node: SearchNode, ppo_updated: bool,
                            new_adapter_node_id: str | None):
    """
    PPO更新の有無に応じて adapter_node_id を設定する。

    ルール:
    1. PPO更新なし → 親の adapter_node_id を継承
    2. PPO更新あり → 新しい adapter_node_id を設定
    3. ルートノード（parent=None） → adapter_root（zero-init）を設定
    """
    if search_node.parent_id is None:
        search_node.adapter_node_id = "adapter_root"
    elif ppo_updated:
        search_node.adapter_node_id = new_adapter_node_id
    else:
        parent = get_node(search_node.parent_id)
        search_node.adapter_node_id = parent.adapter_node_id
```

> **重要**: 外部探索木のノード数 ≥ LoRA系譜木のノード数。PPO更新が走らなければ、多数の探索ノードが同一のアダプタを共有する。これは正常な動作であり、LoRAノードの「1対多」関係を許容する。

### 9.2 PPOのロールアウト収集（具体）

> **実装状況**: 以下は実装済み。ソースコードは `src/sera/learning/rollout.py`, `src/sera/learning/reward.py`, `src/sera/specs/plan_spec.py` を参照。

```python
@dataclass
class PPORollout:
    """1つの探索ノードの実験サイクルを1エピソードとするロールアウト"""
    node_id: str
    prompt: str              # LLMへの入力（仮説生成プロンプト）
    response: str            # LLMの出力（仮説+実験設計JSON）
    log_prob: float          # 出力トークン列の対数確率
    reward: float            # 計算された報酬
    value: float             # 価値関数の推定値
    advantage: float = 0.0   # GAE/HiPER で計算されたアドバンテージ
    returns: float = 0.0     # 割引リターン

@dataclass
class PPORolloutV2(PPORollout):
    """拡張: MT-GRPOターンレベル報酬を含むロールアウト（§26.4.2）"""
    turn_rewards: dict[str, float] = field(default_factory=dict)
    # 例: {"phase0": 0.9, "phase3": 1.0, "phase4": 0.6}
    # 注: 要件定義の turn_log_probs は現在の実装に含まれない（将来拡張）
```

**報酬手法の選択（実装済み）**:

`plan_spec.reward.method` フィールドで報酬計算手法を選択する。レジストリパターン（`_REWARD_METHODS` dict + `register_reward_method` デコレータ）でディスパッチされる:

| method | 関数 | 説明 |
|--------|------|------|
| `"outcome_rm"` | `compute_reward_outcome_rm` | 従来の報酬計算（デフォルト） |
| `"mt_grpo"` | `compute_reward_mt_grpo` | ターンレベル報酬の重み付き和 |
| `"hiper"` | `compute_reward_hiper` | HiPER（報酬値は `mt_grpo` に委譲） |

```python
# ディスパッチャ（実装済み: src/sera/learning/reward.py）
def compute_reward(node, plan_spec, exec_spec, kl_divergence=0.0, **kw) -> float:
    """
    plan_spec.reward.method に応じて適切な報酬計算関数にディスパッチする。

    Outcome RM:
      R = primary_value - constraint_penalty * num_violated
          - lambda_cost * normalized_cost - kl_coef * kl_divergence

    MT-GRPO (turn_rewards が存在する場合):
      R = Σ_t (w_t * r_turn_t) - constraint_penalty * num_violated
          - lambda_cost * normalized_cost - kl_coef * kl_divergence

    HiPER:
      報酬値は MT-GRPO と同一。HiPER固有の処理はAdvantage分解側で行う。

    失敗/タイムアウト/OOM → -100.0 (全手法共通)
    """
    method = getattr(getattr(plan_spec, "reward", None), "method", "outcome_rm")
    fn = _REWARD_METHODS.get(method, compute_reward_outcome_rm)
    return fn(node, plan_spec, exec_spec, kl_divergence, **kw)
```

**ターンレベル報酬評価器（実装済み: `src/sera/learning/turn_reward.py`）**:

```python
class TurnRewardEvaluator:
    """Phase毎のターンレベル報酬を評価する。"""
    def evaluate_all(self, node, parent, all_nodes) -> dict[str, float]:
        """全Phase報酬を計算して返す。"""
        ...

# 登録済み評価器: citation_relevance, hypothesis_novelty,
# code_executability, metric_improvement, paper_score_delta
```

**設定（PlanSpec）**:

```python
class TurnRewardSpec(BaseModel):
    """各Phaseの出力品質を評価するターンレベル報酬定義。PlanSpec に含まれる。"""
    phase_rewards: dict[str, PhaseRewardConfig] = {}

class PhaseRewardConfig(BaseModel):
    evaluator: str   # 評価器名
    weight: float = 0.0  # MT-GRPOの重み付き和における重み
```

### 9.3 PPO更新ループ（具体）

> **実装状況**: 以下は実装済み。ソースコードは `src/sera/learning/ppo_trainer.py`, `src/sera/learning/hierarchical_ppo.py` を参照。

```python
class PPOTrainer:
    """
    trl.PPOTrainer をラップし、LoRA-only 更新を実装する。

    依存: transformers, peft, trl

    コンストラクタ:
      __init__(self, exec_spec, model_spec, lineage_manager, log_path, plan_spec=None)
      plan_spec パラメータにより報酬手法に応じたAdvantage計算をルーティング。
    """

    def _compute_advantages_for_method(self, rollouts, turn_rewards_map=None):
        """
        plan_spec.reward.method に応じたAdvantage計算のルーティング:
          - "hiper"    → HierarchicalAdvantageEstimator.compute_hierarchical_advantages()
          - "mt_grpo"  → _compute_gae（ターン報酬反映済み）
          - "outcome_rm" / その他 → _compute_gae（従来のGAE）
        """
        pass

    def update(self, rollouts: list["PPORollout | PPORolloutV2"], agent_llm: AgentLLM, specs: AllSpecs):
        """
        1. 現在のノードの LoRA アダプタをロード（materialize済み）
        2. rollouts から batch を構成（batch_size=16, mini_batch_size=4）
        3. _compute_advantages_for_method() でメソッド別のAdvantage計算:
           - outcome_rm/mt_grpo: _compute_gae (advantage = reward - value, returns = reward)
           - hiper: HierarchicalAdvantageEstimator で3層分解
             - High Level: reward - value
             - Switch Level: -variance(turn_rewards)
             - Low Level: mean(turn_rewards) - value
             - A = Σ(weight_i * A_i)
        4. epochs_per_update（既定4）回のミニバッチ更新：
           a. old_log_prob と new_log_prob の比率 r(θ) を計算
           b. クリッピング: min(r*A, clip(r, 1-ε, 1+ε)*A)
           c. 価値関数損失: MSE(V_pred, returns)
           d. エントロピーボーナス: entropy_coef * H(π)
           e. 合計損失 = -policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
           f. 勾配クリッピング: max_grad_norm=0.5
           g. optimizer.step()（LoRA パラメータのみ更新される）
        5. KL divergence を計算：
           - kl > kl_target * 1.5 → kl_coef *= 2（自動増加）
           - kl < kl_target / 1.5 → kl_coef /= 2（自動減少）
        6. 更新後のLoRAアダプタと親アダプタの差分Δを計算：
           Δ = updated_lora_weights - parent_lora_weights
        7. Δを adapter_delta.safetensors として保存
        8. ppo_log.jsonl にログ出力（turn_rewards フィールドを含む）
        """
        pass
```

**HiPER 3層Advantage分解（実装済み: `src/sera/learning/hierarchical_ppo.py`）**:

```python
class HierarchicalAdvantageEstimator:
    """HiPER 3層の階層的Advantage分解。"""
    def __init__(self, hiper_config):
        self.switch_weight = hiper_config.switch_level_weight  # 0.3
        self.high_weight = hiper_config.high_level_weight      # 0.4
        self.low_weight = hiper_config.low_level_weight        # 0.3

    def compute_hierarchical_advantages(self, rollouts, turn_rewards_map=None):
        """各ロールアウトの advantage と returns を設定。"""
        ...
```

### 9.4 差分継承（delta inheritance：必須）
- 子のLoRA = 親のLoRA + Δ
- 保存するのは Δのみ（A/B差分）

保存形式：
```text
lineage/nodes/<adapter_node_id>/
  meta.json:
    {
      "adapter_node_id": "...",
      "parent_adapter_node_id": "...",  # null for root
      "search_node_id": "...",
      "adapter_spec_hash": "sha256:...",
      "depth": 3,
      "created_at": "2026-02-21T12:00:00Z",
      "delta_norm_l2": 0.0023,          # Δの L2 ノルム（品質監視用）
      "is_snapshot": false
    }
  adapter_delta.safetensors:
    - model.layers.0.self_attn.q_proj.lora_A.delta  (shape: [rank, in_features])
    - model.layers.0.self_attn.q_proj.lora_B.delta  (shape: [out_features, rank])
    - model.layers.0.self_attn.v_proj.lora_A.delta
    - model.layers.0.self_attn.v_proj.lora_B.delta
    - ...（全 target_layers × target_modules）
```

### 9.5 PPO更新トリガー条件
```text
PPO更新は以下のいずれかで発火：
1. 評価済みノードが ppo_trigger_interval（既定5）個溜まった
2. 現在の最良LCBが plateau_patience ステップ改善していない（学習促進のため）

更新しない条件：
1. 評価済みノードが2個未満（統計的に意味がない）
2. 全ノードが制約違反（有効な報酬信号がない）
```

---
