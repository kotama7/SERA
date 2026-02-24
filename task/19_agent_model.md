# SERA 要件定義書 — Agentモデル拡張

> 本ファイルは TASK.md v12.4 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 25. Agentモデル拡張（Multi-Agent Model Support）

### 25.1 現状と課題

現行SERAは **Qwen2.5-Coder-7B-Instruct** を唯一のベースモデルとして使用している。選定理由は以下の通り：

| 要件 | Qwen2.5-Coder-7B の適合性 |
|------|--------------------------|
| LoRA+PPO互換性 | ○ HuggingFace transformers + peft + trl で完全対応 |
| コード生成能力 | ○ Coder系列で実験スクリプト生成に強い |
| vLLM推論サポート | ○ vLLM公式対応、LoRA Hot-Swap可能 |
| GPUメモリ制約 | ○ 7Bパラメータは単一A100 80GBで推論+学習が共存可能 |
| sleep/wake対応 | ○ vLLMのsleep(level=2)に対応 |

しかし、単一モデルへの依存は以下の**限界**をもたらす：

1. **モデル固有のバイアス**: Qwenの学習データ・アーキテクチャに由来する生成傾向が探索空間を制限
2. **汎化性の未検証**: 他モデルでの動作保証がない（LoRA rank/alpha、プロンプト形式、トークナイザ等の差異）
3. **ベンチマーク比較不能**: SERA自体の貢献とモデル選択の効果が分離できない

### 25.2 対応候補モデル

| モデル | パラメータ数 | 特徴 | vLLM対応 | LoRA+PPO | Tool-Calling互換性 | 備考 |
|--------|------------|------|---------|----------|-------------------|------|
| Qwen2.5-Coder-7B | 7B | コード特化 | ○ | ○ | △（構造化出力で代替可） | 現行ベースライン |
| Qwen3-8B | 8B | 汎用+思考モード | ○ | ○ | ○（ネイティブtool-calling対応） | thinking_budget制御可能 |
| DeepSeek-Coder-V2-Lite | 16B (2.4B active) | MoEコード特化 | ○ | △（MoE+LoRA要検証） | △（Function Calling対応だがMoE+LoRA未検証） | active paramが少なく効率的 |
| CodeLlama-7B | 7B | Meta製コード特化 | ○ | ○ | △（構造化出力で代替可） | Llama系との比較用 |
| Llama-3.1-8B-Instruct | 8B | 汎用 | ○ | ○ | ○（ネイティブtool-calling対応） | 非コード特化モデルの対照群 |

### 25.3 実装要件

#### 25.3.1 ModelSpec の拡張

```yaml
# model_spec.yaml - 拡張版
base_model:
  name: "Qwen/Qwen2.5-Coder-7B-Instruct"   # HuggingFace モデルID
  revision: "abc123"                          # 固定リビジョン
  family: "qwen2"                             # モデルファミリ識別子（新規）

# モデルファミリごとの設定
model_families:
  qwen2:
    chat_template: "qwen2"
    prompt_format: "chatml"                   # <|im_start|>system\n...
    supports_system_prompt: true
    tokenizer_kwargs: {}
  llama3:
    chat_template: "llama3"
    prompt_format: "llama3"                   # <|begin_of_text|><|start_header_id|>...
    supports_system_prompt: true
    tokenizer_kwargs: {}
  deepseek:
    chat_template: "deepseek"
    prompt_format: "deepseek"
    supports_system_prompt: true
    tokenizer_kwargs: {}
```

#### 25.3.2 AgentLLM のモデル抽象化

```python
class AgentLLM:
    """拡張: モデルファミリに応じたプロンプト整形を追加"""

    def _format_prompt(self, prompt: str, purpose: str) -> str:
        """
        model_family に基づいてプロンプトを整形する。
        - qwen2: ChatML形式
        - llama3: Llama3形式
        - deepseek: DeepSeek形式
        プロンプトテンプレート（tree_ops.py内）はモデル非依存の
        プレーンテキストで記述し、ここで最終整形する。
        """
        formatter = PROMPT_FORMATTERS[self.model_family]
        return formatter.format(prompt, purpose)
```

#### 25.3.3 LoRA互換性の保証

```python
def validate_lora_compatibility(model_config: dict, lora_config: dict) -> bool:
    """
    モデル変更時にLoRA互換性を検証する。
    以下が一致しない場合、delta inheritanceが破壊される：
    - hidden_size
    - num_attention_heads
    - num_hidden_layers
    - target_modules（モデルファミリで異なる: q_proj/k_proj/v_proj等）
    """
    pass
```

> **重要**: `adapter_spec_hash`（§10.1）はモデルアーキテクチャの情報を含むため、異なるモデルファミリ間でのdelta継承は**不可能**である。モデル比較実験は独立した探索木として実行する。

### 25.4 アブレーション計画（A7: ベースモデル比較）

| 実験ID | モデル | 評価タスク | 測定指標 |
|--------|--------|----------|---------|
| A7-a | Qwen2.5-Coder-7B | ML/HPC/TSP全域 | primary_metric, tokens/sec, cost |
| A7-b | Qwen3-8B | 同上 | 同上 |
| A7-c | CodeLlama-7B | 同上 | 同上 |
| A7-d | Llama-3.1-8B | 同上 | 同上 |
| A7-e | DeepSeek-Coder-V2-Lite | 同上 | 同上（MoE+LoRA動作検証含む） |
| A7-f | Qwen3-8B (tool-calling有効) | 同上 | 同上 + tool_call_success_rate |
| A7-g | Qwen3-8B (tool-calling無効) | 同上 | 同上（tool-calling有無の比較） |

各モデルについて同一の `ProblemSpec` と `ExecutionSpec`（LoRA target_modules のみモデルに応じて変更）で探索を実行し、以下を比較する：

1. **探索効率**: 同一ノード数での最良メトリクス到達値
2. **PPO学習効果**: LoRA更新前後のメトリクス改善幅
3. **コード生成品質**: 実験スクリプトの実行成功率
4. **コスト効率**: tokens/sec × メトリクス改善の費用対効果
5. **Tool-Calling効果**（A7-f vs A7-g）: tool-calling有無によるメトリクス改善・実行成功率の差分

---
