# SERA 要件定義書 — 付録

> 本ファイルは TASK.md v13.1 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 付録A：先行研究（docs/related_work.md の最低要件）

> **注意**：実装時は Phase 0 で収集した文献を基礎に、ここで規定した枠組みに従って整理文書を生成すること（テンプレ＋自動挿入で可）。

### A-1 自律研究・論文生成（AI-Scientist系）
- AI-Scientist：アイデア→実験→論文→査読風評価の一気通貫
- AI-Scientist-v2：agentic tree search / 管理エージェントの強化
- SERA対応：Phase 7–8、及び Phase 2（探索木）

### A-2 自律コード実験（CodeScientist等）
- コード実験中心の研究支援
- SERA対応：Phase 3–4（実験と評価）

### A-3 探索（tree search / agentic search）
- SERA対応：Phase 2（Best-First + 統計LCB + ε-constraint）

### A-4 LoRA / PEFT / 継続学習・アダプタ管理
- LoRA（PEFTの基盤）
- Delta-LoRA（delta概念の参照）
- アダプタ継続学習（複数タスク/分岐の背景）
- SERA対応：Phase 5–6（PPO+LoRA差分継承+系譜）

### A-5 Scholar API / 文献収集基盤
- Scholar公式APIがない可能性→第三者API/代替API前提
- SERA対応：Phase 0（優先順位＋再現ログ）

### A-6 LLM Agent信用割当・階層RL（§25関連）
- HiPER：3層階層RL（Switch/High/Low）、境界Bootstrapping、明示的報酬分解（arxiv:2602.16165）
- ECHO：Hindsight Trajectory Rewriting、失敗軌道からの事後学習、MDL更新則（arxiv:2510.10304）
- MT-GRPO：ターンレベルMDPでのGRPO適用、マルチターンエージェント学習
- AgentPRM/TRM：プロセス報酬モデルによる中間ステップ評価
- ECHO-2：分散RL基盤、Bounded Staleness、コスト最適ワーカープロビジョニング（arxiv:2602.02192）
- SERA対応：§25 Tool-Using Agent拡張（Phase A: MT-GRPO、Phase B: ECHO軽量版、Phase C: HiPER）

---

## 付録B：実装上の注意（禁止事項）

### B-1 ExecutionSpec固定
- Phase 2以降に探索/評価/学習の規定値を暗黙変更してはならない（ExecutionSpec固定違反）
- PlanSpec.branching.ops の重みを探索中に動的変更してはならない（固定原則違反）

### B-2 変数可変性の境界
- 分岐生成の experiment_config に ProblemSpec.manipulated_variables に存在しないキーを含めてはならない（ホワイトリスト違反）
- ExecutionSpec に属するパラメータ（PPOのlr, repeats, lcb_coef 等）を experiment_config 経由で変更してはならない（層境界違反）
- 分岐生成時にバリデーション（validate_experiment_config）をスキップしてはならない

### B-3 LoRA系譜の整合性
- adapter_spec_hash が異なる delta を合成してはならない（互換性違反）
- PPO更新なしのノードに新規 adapter_node_id を割り当ててはならない（系譜整合性違反）
- 分岐生成ロジック内で adapter_node_id を直接操作してはならない（責務分離違反。adapter_node_id は Phase 5 でのみ設定される）

### B-4 再現性
- API検索の結果を保存せずに進めてはならない（再現性違反）
- LLM呼び出しをログに記録せずに進めてはならない（再現性違反）
- 実験スクリプト内で seed を設定せずに実行してはならない（再現性違反）
- metrics.json を標準出力のみに頼ってはならない（ファイル出力必須）

### B-5 評価一貫性
- PaperSpec/PaperScoreSpec を Phase 7以降に恣意的に差し替えてはならない（評価一貫性違反）

---

## 付録C：AgentLLM インターフェース（tool-calling対応）

> **v13.1 更新**: 以下のインターフェース定義は初期設計時のもの。実装では `ToolRegistry` は `ToolExecutor`（§28）に、`load_tools()` は `AgentLoop` 統合に置き換えられている。`call_function()`（§27）が統一エントリポイントとして単発生成と AgentLoop を自動切替する。ツール・関数の有効化は PlanSpec §5.8 `agent_commands` で Phase 1 に凍結される。実際の実装は `src/sera/agent/agent_llm.py` を参照。

```python
from pydantic import BaseModel
from typing import Any

class ToolCall(BaseModel):
    """エージェントが発行するツール呼び出しの構造化表現"""
    tool_name: str                    # 呼び出すツール名（ToolRegistryに登録済み）
    arguments: dict[str, Any]         # ツールへの引数
    reasoning: str                    # ツール呼び出しの根拠（ログ・デバッグ用）

class GenerationOutput(BaseModel):
    """AgentLLMの統一戻り値型。テキスト生成とツール呼び出しの両方を扱う"""
    text: str | None                  # テキスト出力（テキスト生成時）
    tool_calls: list[ToolCall] | None # ツール呼び出し（Phase C有効時）
    purpose: str                      # 呼び出し目的

class AgentLLM:
    """
    SERAの全LLM呼び出しを統一管理するクラス。

    責務:
    1. ベースモデルのロード（HuggingFace transformers）
    2. LoRAアダプタの動的切り替え（peft）
    3. 推論（generate）— 統一戻り値型 GenerationOutput
    4. ツール付き推論（generate_with_tools）— Phase C で有効化
    5. ツール定義のロード（load_tools）
    6. 外部APIプロバイダへの転送（OpenAI/Anthropic）
    7. 全呼び出しのログ記録（agent_llm_log.jsonl）
    """

    def __init__(self, model_spec: ModelSpec, resource_spec: ResourceSpec):
        if model_spec.agent_llm.provider == "local":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_spec.base_model.id,
                revision=model_spec.base_model.revision,
                torch_dtype=getattr(torch, model_spec.base_model.dtype),
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_spec.base_model.id,
                revision=model_spec.compatibility.tokenizer_revision,
            )
            # 初期LoRA（zero init → ベースモデルと同一出力）
            self.model = get_peft_model(self.model, LoraConfig(
                r=model_spec.adapter_spec.rank,
                lora_alpha=model_spec.adapter_spec.alpha,
                target_modules=model_spec.adapter_spec.target_modules,
                lora_dropout=model_spec.adapter_spec.dropout,
                init_lora_weights=model_spec.adapter_spec.init == "zero",
            ))
        else:
            # OpenAI/Anthropic API クライアント
            self.client = create_api_client(model_spec.agent_llm)

        self._tool_registry: ToolRegistry | None = None

    def generate(self, prompt: str, purpose: str, adapter_node_id: str | None = None,
                 temperature: float | None = None, max_tokens: int | None = None) -> GenerationOutput:
        """
        LLM推論を実行し、結果をログに記録する。

        注: 実装では後方互換性のため generate() は str を返す。
        GenerationOutput が必要な場合は generate_full() を使用する。

        Args:
            prompt: 入力テキスト
            purpose: 呼び出し目的（"branch_generation", "experiment_code", "paper_section",
                     "failure_analysis", etc.）
            adapter_node_id: 使用するLoRAアダプタのID（local時のみ）
            temperature: 温度パラメータ（None=ModelSpec既定値）
            max_tokens: 最大生成トークン数（None=ModelSpec既定値）

        Returns:
            GenerationOutput(text=..., tool_calls=None, purpose=purpose)
            ※ Phase A/B では tool_calls は常に None
        """
        pass

    def generate_with_tools(self, prompt: str, available_tools: list[dict],
                            purpose: str, adapter_node_id: str | None = None,
                            temperature: float | None = None,
                            max_tokens: int | None = None) -> GenerationOutput:
        """
        ツール定義を含むLLM推論を実行する（Phase Cで有効化）。

        Args:
            prompt: 入力テキスト
            available_tools: ツール定義のリスト（ToolRegistryから取得）
            purpose: 呼び出し目的
            adapter_node_id: 使用するLoRAアダプタのID
            temperature: 温度パラメータ
            max_tokens: 最大生成トークン数

        Returns:
            GenerationOutput(text=..., tool_calls=[ToolCall(...)], purpose=purpose)
        """
        pass

    def load_tools(self, tool_registry: "ToolRegistry") -> None:
        """ToolRegistry をロードし、generate_with_tools で利用可能にする"""
        self._tool_registry = tool_registry

    def load_adapter(self, adapter_node_id: str):
        """指定のLoRAアダプタをロード（materialize してから set_adapter）"""
        pass

    def get_log_probs(self, prompt: str, response: str) -> float:
        """PPO用：response の対数確率を計算（local時のみ）"""
        pass

    def get_turn_log_probs(self, prompt: str,
                           responses_per_phase: dict[str, str]) -> dict[str, float]:
        """
        MT-GRPO用：Phase毎の応答に対する対数確率を計算（local時のみ）。

        Args:
            prompt: 各Phaseへの入力プロンプト
            responses_per_phase: {"phase2": "...", "phase3": "...", ...}

        Returns:
            {"phase2": -1.23, "phase3": -0.98, ...}
        """
        pass
```

---
