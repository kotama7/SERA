# SERA 要件定義書 — Tool-Using Agent拡張（HiPER + ECHO）

> 本ファイルは TASK.md v12.4 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 26. Tool-Using Agent拡張（HiPER + ECHO統合による信用割当）

### 26.1 旧設計から新設計への移行

SERAのAgentLLM（§0.1, §21）は v12.3 までテキスト生成器として設計されていた。v12.4 で tool-calling対応の自律エージェントに移行する。

```text
■ 旧設計（v12.3以前）: テキスト生成器
  SearchManager ──(戦略決定)──► どのノードを展開するか
       │
       ▼
  AgentLLM ──(テキスト生成)──► コード・仮説のテキスト出力
       │
       ▼
  Executor ──(実験実行)──► metrics.json
       │
       ▼
  RewardComputer ──(報酬計算)──► R = f(metrics) のみ

■ 新設計（v12.4）: Tool-Calling Agent + MT-GRPO + ECHO軽量版
  SearchManager ──(戦略決定)──► どのノードを展開するか
       │
       ▼
  AgentLLM ──(テキスト生成 + ツール呼び出し)──► GenerationOutput(text, tool_calls)
       │                                            │
       │                                    ToolRegistry ──► ツール実行結果
       ▼
  Executor ──(実験実行)──► metrics.json
       │                      │
       │               FailureKnowledgeExtractor ──► 兄弟ノードcontext注入
       ▼
  RewardComputer ──(MT-GRPO報酬)──► R = Σ(w_t * r_turn_t) - penalties
```

**旧設計の制約**: 報酬はメトリクスのみから計算される。LLMの中間的な意思決定品質（適切なライブラリ選択、効率的なアルゴリズム設計、デバッグ戦略等）は報酬に反映されない。

**新設計の改善**: ターンレベル報酬（MT-GRPO）により各Phaseの出力品質を個別に評価。失敗知識の再利用（ECHO軽量版）により探索効率が向上。Phase Cでtool-calling対応。

### 26.2 Tool-Using Agent化の障壁

SERAにtool-calling agentを導入する際の障壁は以下の5点：

| 障壁 | 詳細 | 深刻度 |
|------|------|--------|
| **B1: 信用割当問題** | ツール呼び出し列の中でどの行動が成功/失敗に寄与したか不明 | 致命的 |
| **B2: 行動空間爆発** | テキスト生成に加えツール選択・引数生成が加わり方策空間が膨張 | 高 |
| **B3: vLLM sleep/wake制約** | tool-calling中にvLLMを維持する必要があり、PPO学習とのGPU共存が困難 | 高 |
| **B4: ツール実行の非決定性** | 同一ツール呼び出しでもネットワーク/環境状態で結果が変わる | 中 |
| **B5: ExecutionSpec固定との互換** | ツール種類・パラメータの追加が固定層原則と矛盾する可能性 | 中 |

### 26.3 既存手法の評価とSERAへの適合性

LLM Agentの信用割当問題に対する既存手法を評価した：

| 手法 | 原理 | SERAとの互換性 | 採用判定 |
|------|------|---------------|---------|
| **Outcome RM** (SWE-Agent型) | 最終結果のみで報酬 | ◎ 現行SERAと同一 | 現状維持 |
| **Process RM** (AgentPRM/TRM) | 中間ステップを別モデルで評価 | △ 追加モデルの学習コスト | 将来検討 |
| **MT-GRPO** (Turn-level MDP) | ターンごとにGRPO適用 | **◎ Phase構造と自然対応** | **推奨** |
| **HiPER** (階層型RL) | 3層の階層的信用割当 | △ 階層はシステムレベルで既存 | 長期目標 |
| **ECHO** (Hindsight Rewriting) | 失敗軌道を事後書き換え | △ 木構造と構造的重複 | 軽量版のみ |

### 26.4 採用アーキテクチャ：MT-GRPO + ECHO軽量版

#### 26.4.1 設計方針

```text
採用アーキテクチャ（v12.4で正式採用）:

  Phase 0-8 の各Phase完了時:
    r_turn = TurnRewardModel(phase_output)    ← MT-GRPO（ターンレベル報酬）

  実験失敗時:
    summary = LLM.summarize(failed_trajectory)  ← ECHO軽量版（知識抽出）
    sibling_node.context += summary              ← 兄弟ノードへ注入

  PPO更新時:
    advantage = Σ γ^t * r_turn  （ターンレベル報酬の重み付き和）
```

SERAの既存8フェーズパイプラインが**自然にターン構造**を形成しているため、MT-GRPOが最も低コストで統合可能である。

#### 26.4.2 MT-GRPO統合（Phase報酬の定義）

> **実装状況**: 実装済み。ソースコードは `src/sera/specs/plan_spec.py`, `src/sera/learning/turn_reward.py`, `src/sera/learning/rollout.py`, `src/sera/learning/reward.py` を参照。

```python
# 設定モデル（実装済み: src/sera/specs/plan_spec.py）
class PhaseRewardConfig(BaseModel):
    evaluator: str
    weight: float = 0.0

class TurnRewardSpec(BaseModel):
    """各Phaseの出力品質を評価するターンレベル報酬定義"""
    phase_rewards: dict[str, PhaseRewardConfig] = {}

# plan_spec.yaml での設定例:
# turn_rewards:
#   phase_rewards:
#     phase0: { evaluator: "citation_relevance", weight: 0.10 }
#     phase2: { evaluator: "hypothesis_novelty", weight: 0.15 }
#     phase3: { evaluator: "code_executability", weight: 0.25 }
#     phase4: { evaluator: "metric_improvement", weight: 0.35 }
#     phase7: { evaluator: "paper_score_delta",  weight: 0.15 }
```

```python
# ロールアウト（実装済み: src/sera/learning/rollout.py）
@dataclass
class PPORolloutV2(PPORollout):
    """拡張: ターンレベル報酬を含むロールアウト"""
    turn_rewards: dict[str, float] = field(default_factory=dict)
    # 例: {"phase0": 0.9, "phase3": 1.0, "phase4": 0.6}

# 報酬計算（実装済み: src/sera/learning/reward.py）
# レジストリパターンで3手法をディスパッチ:
#   _REWARD_METHODS = {"outcome_rm": ..., "mt_grpo": ..., "hiper": ...}

@register_reward_method("mt_grpo")
def compute_reward_mt_grpo(node, plan_spec, exec_spec, kl_divergence=0.0, **kw) -> float:
    """
    MT-GRPO報酬計算:
      R = Σ_t (w_t * r_turn_t) - penalties
    turn_rewards が未提供の場合は outcome_rm にフォールバック。
    """
    ...

def compute_reward(node, plan_spec, exec_spec, kl_divergence=0.0, **kw) -> float:
    """plan_spec.reward.method でディスパッチ。未知のメソッド名は outcome_rm にフォールバック。"""
    method = getattr(getattr(plan_spec, "reward", None), "method", "outcome_rm")
    fn = _REWARD_METHODS.get(method, compute_reward_outcome_rm)
    return fn(node, plan_spec, exec_spec, kl_divergence, **kw)
```

#### 26.4.3 ECHO軽量版（失敗知識注入）

> **実装状況**: 実装済み。ソースコードは `src/sera/search/failure_extractor.py`, `src/sera/search/search_manager.py`, `src/sera/search/tree_ops.py` を参照。

完全なECHOパイプライン（軌道要約→目標再推測→軌道書き換え→MDL更新）は木構造と構造的に重複するため、**知識抽出と注入のみ**を採用する：

```python
# 実装済み: src/sera/search/failure_extractor.py

@dataclass
class FailureSummary:
    """構造化された失敗要約"""
    node_id: str
    hypothesis: str
    error_category: str  # "runtime", "oom", "timeout", "logical", "unknown"
    error_message: str   # summary_max_tokens で切り詰め
    lesson: str          # ヒューリスティック生成の教訓

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "FailureSummary": ...

class FailureKnowledgeExtractor:
    """ECHO軽量版: 失敗ノードから知識を抽出し兄弟ノードに注入"""

    def __init__(self, echo_config, agent_llm=None):
        self.max_summaries = echo_config.max_summaries_per_node  # デフォルト: 3
        self.summary_max_tokens = echo_config.summary_max_tokens  # デフォルト: 256

    def extract(self, failed_node) -> FailureSummary:
        """
        失敗ノードの実験結果をヒューリスティックに分類・要約する。
        - エラーカテゴリ分類: status とエラーメッセージのキーワードマッチ
        - 教訓生成: カテゴリに応じたテンプレートベースの教訓
        - エラーメッセージは summary_max_tokens で切り詰め

        注意: 現在はヒューリスティック実装。agent_llm による高品質抽出は将来拡張。
        """
        ...

    def inject(self, summary: FailureSummary, siblings: list):
        """
        失敗知識を兄弟ノードの failure_context に注入する。
        - 重複防止: 同一 node_id の FailureSummary は追加しない
        - 上限制御: max_summaries_per_node を超えると追加をスキップ
        - failure_context 属性がないノードはスキップ
        """
        ...
```

**設定（PlanSpec）**:

```python
class EchoConfig(BaseModel):
    enabled: bool = False           # ECHO機能の有効化
    max_summaries_per_node: int = 3 # 1ノードに注入する最大失敗サマリ数
    summary_max_tokens: int = 256   # エラーメッセージ・教訓の最大トークン長
```

**プロンプト注入（実装済み）**:

`IMPROVE_PROMPT` に `{failure_context}` プレースホルダを配置。`TreeOps._build_failure_context(parent)` が `failure_context` リストをテキスト化:

```text
Failed approaches to avoid:
- [runtime] My hypothesis: Approach 'My hypothesis' raised a runtime error: index out of range
- [oom] Another approach: Approach 'Another approach' caused OOM. Consider reducing model/batch size.
```

> **ECHOとの差分**: 完全なECHOは「失敗軌道を書き換えて成功軌道として再利用」するが、SERAの木構造では兄弟ノードの展開が自然に代替軌道を生成するため、書き換えは冗長である。失敗からの**知識抽出のみ**をECHOから借用する。

#### 26.4.4 木構造との互換性

> **実装状況**: 後方互換性は全て確認済み。

```text
MT-GRPO + ECHO軽量版 + HiPER と既存アーキテクチャの互換性:

                        互換性   実装済み変更
外部探索木（LCB選択）     ◎      変更なし
LoRA系譜（delta継承）     ◎      PPORolloutV2に拡張（PPORollout後方互換）
SearchManager             ◎      failure_extractor/turn_reward_evaluator追加（デフォルトNone）
SearchNode                ◎      failure_context: list[dict]追加（デフォルト[]、from_dict互換）
TreeOps                   ◎      IMPROVE_PROMPTに{failure_context}追加、_build_failure_context追加
ExecutionSpec固定          ◎      全新フィールドはPlanSpecに追加（ロックハッシュ不変）
vLLM sleep/wake           ◎      変更なし
PPOTrainer                ◎      plan_spec追加（デフォルトNone）、メソッド別Advantageルーティング
reward.py                 ◎      レジストリパターンで3手法ディスパッチ（**kw追加で後方互換）
AgentLLM                  ◎      変更なし
```

### 26.5 長期目標：HiPER統合（Phase C）

MT-GRPO + ECHO軽量版の検証後、SERAを完全なtool-using agentに進化させる場合はHiPERの階層的信用割当が有効である。

#### 26.5.1 HiPER概要

HiPER（Hierarchical Policy for Explicit Reward decomposition）は3層の階層的RLフレームワーク：

```text
HiPER 3層構造:
  Switch Level:  タスク切替判断（continue / switch）
       │
  High Level:    サブゴール設定（ツール選択、戦略決定）
       │
  Low Level:     具体的行動（ツール引数生成、テキスト出力）
```

各層に独立したAdvantage推定を持ち、境界でのBootstrappingにより分散を低減する（Theorem 4.3: $\text{Var}[\hat{A}^{\text{HiPER}}] \leq \text{Var}[\hat{A}^{\text{flat}}]$）。

#### 26.5.2 SERAへのHiPERマッピング

```text
HiPER層             SERA現行の対応            拡張後の対応
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Switch Level    SearchManager（ハードコード）   学習可能なノード選択方策
High Level      Phase選択（固定パイプライン）   学習可能なPhase戦略
Low Level       AgentLLM（テキスト生成）        Tool-calling AgentLLM
```

> **注意**: HiPER統合はSERAの根本的再設計を伴う。SearchManagerのハードコードされたLCBスコアリングを学習可能な方策に置き換える必要があり、設計原則の「統計的妥当性必須」との整合性を慎重に検討する必要がある。

### 26.6 段階的実装ロードマップ

> **実装状況更新**: Phase A + Phase B + Phase C の報酬/Advantage部分は**全て実装済み**。設定ファイル（`plan_spec.yaml`）から全手法を切り替え可能。残りは Phase C の Agent LLM tool-calling 対応のみ。

```text
Phase A（短期・低コスト）: MT-GRPO統合 ✅ 実装済み
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  対象: PPORollout → PPORolloutV2, compute_reward → レジストリパターンで3手法ディスパッチ
  実装ファイル:
    - src/sera/learning/ppo_trainer.py（plan_specパラメータ追加、メソッド別Advantageルーティング）
    - src/sera/learning/turn_reward.py（新規: TurnRewardEvaluator — 5つのPhase評価器）
    - src/sera/learning/reward.py（レジストリパターン: outcome_rm / mt_grpo / hiper）
    - src/sera/learning/rollout.py（PPORolloutV2 追加）
    - src/sera/specs/plan_spec.py（RewardConfig.method, TurnRewardSpec, PhaseRewardConfig 追加）
    - src/sera/search/search_manager.py（turn_reward_evaluator統合、PPORolloutV2使い分け）
    - src/sera/commands/research_cmd.py（TurnRewardEvaluator条件付き初期化）
  テスト:
    - tests/test_learning/test_turn_reward.py（11テスト）
    - tests/test_evaluation/test_reward.py（ディスパッチテスト8件追加）
  検証: A1アブレーション（PPO有無）でターン報酬の効果を測定

Phase B（中期・中コスト）: ECHO軽量版統合 ✅ 実装済み
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  対象: 失敗ノードからの知識抽出・兄弟注入
  実装ファイル:
    - src/sera/search/failure_extractor.py（新規: FailureKnowledgeExtractor + FailureSummary）
    - src/sera/search/search_manager.py（failure_extractor統合、debugオペレータ後の抽出・注入）
    - src/sera/search/search_node.py（failure_context: list[dict] フィールド追加）
    - src/sera/search/tree_ops.py（IMPROVE_PROMPTに{failure_context}追加、_build_failure_context）
    - src/sera/specs/plan_spec.py（EchoConfig追加）
    - src/sera/commands/research_cmd.py（FailureKnowledgeExtractor条件付き初期化）
  テスト:
    - tests/test_search/test_failure_extractor.py（12テスト）
  検証: A3アブレーション（コンテキスト戦略）で失敗知識注入の効果を測定

Phase C: HiPER Advantage分解 ✅ 実装済み / Tool-Using Agent化 🔲 未着手
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  実装済み部分:
    - src/sera/learning/hierarchical_ppo.py（新規: HierarchicalAdvantageEstimator — 3層Advantage分解）
    - src/sera/learning/ppo_trainer.py（method=="hiper"時にHierarchicalAdvantageEstimator使用）
    - src/sera/specs/plan_spec.py（HiperConfig追加）
  テスト:
    - tests/test_learning/test_hierarchical_ppo.py（8テスト）

  未着手部分（Tool-Using Agent化）:
    - src/sera/agent/agent_llm.py（tool-calling対応）
    - src/sera/agent/tool_registry.py（新規: ツール定義・実行管理）
    - src/sera/search/search_manager.py（学習可能な選択方策へ拡張）
  前提条件: HiPER/ECHOの有効性がアブレーションで確認済み
  検証: 全アブレーション再実行 + tool-calling特有の評価指標追加
```

### 26.7 ECHO-2（分散RL基盤）との関連

ECHO-2（Xiao et al., 2026; arxiv:2602.02192）はGradient社による分散RLフレームワークであり、Hindsight RewritingのECHOとは別系統の研究である。ECHO-2は以下の点でSERAの**スケーラビリティ**に関連する：

| ECHO-2の概念 | SERAへの適用 |
|-------------|-------------|
| Rollout Plane（分散軌道生成） | 実験ノードの並列評価をワーカープールで実行 |
| Learning Plane（中央集権学習） | PPO更新を単一ノードに集約 |
| Data Plane（タスクアダプタ） | ドメイン別（ML/HPC/TSP）の設定を分離 |
| Bounded Staleness S | 非同期ツリー探索でのLoRA staleness制御 |
| Peer-Assisted Broadcast | LoRAスナップショットの効率的配信 |

> **注意**: ECHO-2の統合はPhase A-Cとは独立したスケーリング課題であり、単一ノード上での動作検証完了後に検討する。

### 26.8 新規ファイル一覧

| ファイル | Phase | 状態 | 役割 |
|---------|-------|------|------|
| `src/sera/learning/turn_reward.py` | A | ✅ 実装済み | TurnRewardEvaluator — 5つのPhase評価器 |
| `src/sera/learning/rollout.py` (PPORolloutV2) | A | ✅ 実装済み | MT-GRPO用の拡張ロールアウト |
| `src/sera/search/failure_extractor.py` | B | ✅ 実装済み | FailureKnowledgeExtractor + FailureSummary (ECHO軽量版) |
| `src/sera/learning/hierarchical_ppo.py` | C | ✅ 実装済み | HierarchicalAdvantageEstimator — 3層Advantage分解 |
| `src/sera/agent/tool_registry.py` | C | 🔲 未着手 | ツール定義・実行管理 |
| `tests/test_learning/test_turn_reward.py` | A | ✅ 実装済み | ターン報酬のユニットテスト（11テスト） |
| `tests/test_learning/test_hierarchical_ppo.py` | C | ✅ 実装済み | HiPER Advantage分解のユニットテスト（8テスト） |
| `tests/test_search/test_failure_extractor.py` | B | ✅ 実装済み | 失敗知識抽出のユニットテスト（12テスト） |
| `tests/test_evaluation/test_reward.py` (追記) | A | ✅ 実装済み | ディスパッチテスト（8テスト追加） |

### 26.8.1 修正ファイル一覧

| ファイル | Phase | 変更内容 |
|---------|-------|---------|
| `src/sera/specs/plan_spec.py` | A+B+C | RewardConfig.method, TurnRewardSpec, EchoConfig, HiperConfig 追加 |
| `src/sera/learning/reward.py` | A | レジストリパターン導入、3手法ディスパッチ |
| `src/sera/learning/ppo_trainer.py` | A+C | plan_spec パラメータ追加、メソッド別Advantageルーティング |
| `src/sera/search/search_node.py` | B | failure_context フィールド追加 |
| `src/sera/search/tree_ops.py` | B | IMPROVE_PROMPT に {failure_context}、_build_failure_context 追加 |
| `src/sera/search/search_manager.py` | A+B | failure_extractor/turn_reward_evaluator 統合 |
| `src/sera/commands/research_cmd.py` | A+B | 条件付きコンポーネント初期化 |

### 26.9 先行研究参照

| 手法 | 論文 | 年 | 核心 |
|------|------|----|------|
| MT-GRPO | Multi-Turn GRPO | 2025 | ターンレベルMDPでのGRPO適用 |
| HiPER | Hierarchical Policy for Explicit Reward decomposition (arxiv:2602.16165) | 2026 | 3層階層RL、境界Bootstrapping、分散低減証明 |
| ECHO | Sample-Efficient Online Learning via Hindsight Trajectory Rewriting (arxiv:2510.10304) | 2025 | 失敗軌道の事後書き換え、MDL更新則 |
| ECHO-2 | Large-Scale Distributed Rollout Framework (arxiv:2602.02192) | 2026 | 分散RL基盤、Bounded Staleness、コスト最適化 |
| AgentPRM | Process Reward Model for LLM Agents | 2025 | 中間ステップ評価モデル |

---

（このTASK.mdは完全最終版 v12.4（全面Agent対応更新版）である。v12.3 からの変更：§0-§22を tool-calling + MT-GRPO + ECHO軽量版前提に改訂。§0.1 エージェント定義をtool-calling対応に書き換え、§1 設計原則に信用割当原則追加、§2.1 データフロー図にターン報酬・失敗知識注入追加、§6 にECHO失敗知識抽出統合、§9 にPPORolloutV2/compute_reward_v2/TurnRewardSpec追加、§11-12 をエージェント的反復改善/マルチエージェントレビューとして再フレーミング、§19-20 受け入れ基準・MVP・モジュール構成にAgent機能追加、付録C AgentLLMインターフェースをtool-calling対応に全面書き換え、§22 実装手順書にStep 13-15追加。v12.2 からの追加：§25 Agentモデル拡張、§26 Tool-Using Agent拡張。v12.1 からの追加：§24 SLURM実行パイプライン。v12 からの追加：§23 多言語実験サポート）
