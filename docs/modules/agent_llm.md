# AgentLLM / PromptFormatters / ToolCall / GenerationOutput

SERA の全 LLM 呼び出しを管理する統合インターフェースのドキュメント。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `AgentLLM` | `src/sera/agent/agent_llm.py` |

## 依存関係

- `transformers` -- ローカルモデルのロード・推論
- `peft` -- LoRA アダプタ管理
- `trl` -- エントロピー・対数確率計算
- `accelerate` -- デバイス管理
- `openai` / `anthropic` -- 外部 API プロバイダ（任意）
- `sera.agent.vllm_engine` (`VLLMInferenceEngine`) -- vLLM 推論エンジン（任意）
- `sera.specs.model_spec` (`ModelSpecModel`, `infer_model_family`) -- モデル設定

---

## ToolCall (dataclass)

LLM の応答から抽出された単一のツール呼び出しを表す。

```python
@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reasoning: str = ""
```

- `reasoning`: ツール呼び出しの前にLLMが生成した推論テキスト（ローカルプロバイダでのパース時に抽出）

## GenerationOutput (dataclass)

`generate()` / `generate_with_tools()` / `generate_full()` の構造化された出力。

```python
@dataclass
class GenerationOutput:
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    purpose: str = ""
    text_log_prob: float | None = None
    tool_call_log_probs: list[float] | None = None
```

- `text_log_prob`: テキスト出力の対数確率（PPO用、ローカルプロバイダのみ）
- `tool_call_log_probs`: 各ツール呼び出しの対数確率リスト（将来の tool_aware 報酬用）

---

## PROMPT_FORMATTERS

モデルファミリごとのプロンプトフォーマットレジストリ。`_PromptFormatter` 基底クラスのサブクラスを格納する。

```python
PROMPT_FORMATTERS: dict[str, _PromptFormatter] = {
    "chatml": _ChatMLFormatter(),
    "llama3": _Llama3Formatter(),
    "llama2": _Llama3Formatter(),
    "deepseek": _DeepSeekFormatter(),
    "default": _PromptFormatter(),
}
```

### フォーマッタの動作

| キー | 対象モデル | フォーマット |
|------|----------|------------|
| `chatml` | Qwen2 系 | `<\|im_start\|>system\n...<\|im_end\|>\n<\|im_start\|>user\n...<\|im_end\|>\n<\|im_start\|>assistant\n` |
| `llama3` | Llama 3 Instruct | `<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n...<\|eot_id\|>...` |
| `llama2` | Llama 2 系 | `llama3` と同一のフォーマッタを使用 |
| `deepseek` | DeepSeek 系 | DeepSeek 固有の特殊トークンを使用 |
| `default` | その他 | プロンプトをそのまま返す（パススルー） |

フォーマッタは `AgentLLM._format_prompt()` から呼び出される。トークナイザに `apply_chat_template` がある場合はそちらが優先される。

---

## AgentLLM

SERA の統合 LLM 呼び出しマネージャ。以下の責務を持つ:

1. ローカルモデルのロード（HuggingFace transformers）
2. 動的 LoRA アダプタ切り替え（peft）
3. テキスト生成（generate）
4. 外部 API への転送（OpenAI / Anthropic）
5. 全呼び出しの JSONL ログ記録

### コンストラクタ

```python
def __init__(self, model_spec: Any, resource_spec: Any, log_path: Path)
```

- `model_spec`: `ModelSpecModel` またはダック型オブジェクト。`base_model`, `adapter_spec`, `agent_llm` 等のフィールドを持つ
- `resource_spec`: `ResourceSpecModel` またはダック型オブジェクト。`api_keys`, `compute` 等のフィールドを持つ
- `log_path`: `agent_llm_log.jsonl` の保存先パス

初期化時にモデルファミリを自動検出し、対応するプロンプトフォーマッタを設定:

```python
self._model_family = getattr(getattr(model_spec, "base_model", None), "family", "")
self._prompt_format = "default"
if self._model_family:
    family_cfg = getattr(model_spec, "get_family_config", lambda: None)()
    if family_cfg:
        self._prompt_format = getattr(family_cfg, "prompt_format", "default")
```

### call_function(function_name, prompt, ...) -> Any

**構造化 LLM 呼び出しのプライマリエントリポイント。** `REGISTRY` に登録された `AgentFunction` のスキーマに基づいてLLMを呼び出し、出力を検証する非同期メソッド。

```python
async def call_function(
    self,
    function_name: str,
    prompt: str,
    purpose: str | None = None,
    adapter_node_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any
```

**処理フロー:**

1. `REGISTRY.get(function_name)` で `AgentFunction` を取得
2. `allowed_tools` が定義されている場合: `AgentLoop` を使用してReActループで実行（ツール呼び出し付き）
3. API プロバイダ（OpenAI/Anthropic）で `return_schema` が定義されている場合: ネイティブ tool calling を使用
4. ローカルプロバイダの場合: スキーマをプロンプトに注入してテキスト生成
5. `handler` が定義されている場合: `handler(raw_response)` で後処理
6. `validate_against_schema()` で出力を検証
7. 失敗時: `max_retries` 回までリトライ（温度を +0.1 ずつインクリメント）

**`generate()` との使い分け:**
- `call_function()`: スキーマ検証が必要な構造化呼び出し（JSON/CODE出力）に使用
- `generate()`: フリーテキスト生成に使用

### generate(prompt, purpose, ...) -> str

LLM からテキストを生成する非同期メソッド。

```python
async def generate(
    self,
    prompt: str,
    purpose: str,
    adapter_node_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str
```

**プロバイダ別の処理:**

| プロバイダ | 処理 |
|-----------|------|
| `local` | vLLM エンジン（利用可能な場合）または transformers で推論。チャットテンプレートまたは `_format_prompt` でフォーマット |
| `openai` | OpenAI Chat Completions API を呼び出し |
| `anthropic` | Anthropic Messages API を呼び出し |

全呼び出しは `agent_llm_log.jsonl` にログ記録される（`_log_call`）。

### generate_full(prompt, purpose, ...) -> GenerationOutput

`generate()` と同様だが、`GenerationOutput` を返す非同期メソッド。`text_log_prob` 等のメタデータも含む。

```python
async def generate_full(
    self,
    prompt: str,
    purpose: str,
    adapter_node_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> GenerationOutput
```

### _format_prompt(prompt, purpose) -> str

モデルファミリに応じたプロンプトフォーマットを適用する内部メソッド。

- ローカルプロバイダでトークナイザに `apply_chat_template` がない場合に使用
- `PROMPT_FORMATTERS[self._prompt_format]` から適切なフォーマッタを取得して適用
- API プロバイダではパススルー

### generate_with_tools(prompt, available_tools, purpose, ...) -> GenerationOutput

ツール呼び出し付きのテキスト生成を行う非同期メソッド。

```python
async def generate_with_tools(
    self,
    prompt: str,
    available_tools: list[dict],
    purpose: str,
    adapter_node_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> GenerationOutput
```

**プロバイダ別の処理:**

| プロバイダ | 処理 |
|-----------|------|
| `openai` | ネイティブ tool calling を使用。レスポンスから `tool_calls` を抽出 |
| `anthropic` | ネイティブ tool calling を使用。レスポンスから `tool_use` ブロックを抽出 |
| `local` | `_generate_local_with_tools()` でネイティブチャットテンプレートのツール呼び出しまたはプロンプトベースでツール定義を注入し、`_parse_local_tool_calls()` でパース |

### _generate_local_with_tools(prompt, available_tools, ...) -> str

ローカルプロバイダでのツール呼び出し付きテキスト生成の内部非同期メソッド。

- トークナイザのチャットテンプレートが `tools` パラメータをサポートしている場合はネイティブのツール呼び出しフォーマットを使用
- サポートしていない場合はツール定義をJSONとしてプロンプトに埋め込む

### _parse_local_tool_calls(text) -> tuple[list[ToolCall] | None, str] [staticmethod]

ローカルモデルの出力からツール呼び出しを抽出する静的メソッド。

- `<tool_call>` タグ内のJSON、またはプレーンJSONとしてパース
- 返り値: `(tool_calls, remaining_text)` のタプル。ツール呼び出しが見つからない場合は `(None, text)`

### load_adapter(adapter_node_id, lineage_manager)

LoRA アダプタの重みをロードする。

- `lineage_manager` が指定された場合: lineage ツリーから重みをマテリアライズ
- `peft.set_peft_model_state_dict` で重みを設定
- `AutoModelForCausalLMWithValueHead` のラッピングにも対応
- ローカルプロバイダのみ

### get_log_probs(prompt, response) -> float

応答の対数確率を計算する（PPO 用）。

- `trl.trainer.utils.selective_log_softmax` を使用（メモリ効率が高い）
- ローカルプロバイダのみ

### get_log_probs_with_logits(prompt, response) -> tuple[float, torch.Tensor]

対数確率とロジットを同時に返す（エントロピー計算用）。

- 戻り値: `(summed_log_prob, shift_logits)` where `shift_logits.shape = (1, response_len, vocab_size)`
- ローカルプロバイダのみ

### get_value(prompt, response) -> float

バリューヘッドによる価値推定を返す（PPO GAE 用）。

- レスポンストークンに対するバリューヘッド出力の平均値
- バリューヘッドがない場合またはローカル以外では `0.0` を返す

### get_turn_log_probs(prompt, responses_per_phase) -> dict[str, float]

フェーズ別の対数確率を計算する（MT-GRPO / HiPER 用）。

```python
def get_turn_log_probs(
    self,
    prompt: str,
    responses_per_phase: dict[str, str],
) -> dict[str, float]
```

- 各フェーズの応答テキストに対して個別に対数確率を計算
- API プロバイダでは均一なプレースホルダ（各フェーズ `0.0`）を返す

### set_mock(mock_fn)

テスト用モック関数を設定する。

```python
def set_mock(self, mock_fn: Callable[[str, str], str]) -> None
```

- `mock_fn(prompt, purpose) -> str`
- 設定後は全 `generate` 呼び出しがモック関数を経由する
- テストでは実 I/O をバイパスするために使用

---

## ログフォーマット（agent_llm_log.jsonl）

```json
{
  "event": "llm_call",
  "call_id": "<UUID>",
  "purpose": "draft",
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "adapter_node_id": "abc123",
  "prompt_tokens": 1024,
  "completion_tokens": 256,
  "temperature": 0.7,
  "prompt_hash": "<SHA-256 先頭 16 文字>",
  "response_hash": "<SHA-256 先頭 16 文字>",
  "latency_ms": 1234
}
```

---

## vLLM エンジン統合

- vLLM エンジンは遅延初期化される（最初の `generate` 呼び出し時）
- PPO 学習中は `sleep(level=2)` で GPU メモリを解放
- 学習完了後に `wake()` で復帰（成功・失敗に関わらず）
- vLLM と PyTorch トレーニングは同一 GPU 上で共存できないため、この切り替えが必要
