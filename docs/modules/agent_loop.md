# AgentLoop / AgentLoopConfig / AgentLoopResult

ReAct (Reason + Act) スタイルのエージェント反復ループ。LLM がツール呼び出しを生成しなくなるか、終了条件に達するまで Think → Act → Observe を繰り返す。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `AgentLoop` | `src/sera/agent/agent_loop.py` |

## 依存関係

- `sera.agent.agent_llm` (`AgentLLM`, `GenerationOutput`) -- LLM インターフェース
- `sera.agent.tool_executor` (`ToolExecutor`, `ToolResult`, `get_tool_schemas`) -- ツールディスパッチ
- `sera.utils.logging` (`JsonlLogger`) -- JSONL ログ記録

---

## AgentTurn (dataclass)

エージェントループの単一ステップの記録。

```python
@dataclass
class AgentTurn:
    step: int                           # ステップ番号
    prompt: str                         # 入力プロンプト（末尾 500 文字に制限）
    generation: GenerationOutput        # LLM 生成結果
    tool_results: list[ToolResult]      # ツール実行結果
    wall_time_sec: float = 0.0          # ステップの実行時間
```

---

## AgentLoopResult (dataclass)

エージェントループ全体の実行結果。

```python
@dataclass
class AgentLoopResult:
    final_output: Any                   # 最終出力（最後のステップのテキスト）
    turns: list[AgentTurn]              # 全ステップの記録
    total_steps: int = 0                # 実行されたステップ数
    total_tool_calls: int = 0           # 合計ツール呼び出し数
    total_wall_time_sec: float = 0.0    # 合計実行時間
    exit_reason: str = "completed"      # 終了理由
```

### 終了理由

| `exit_reason` | 説明 |
|---------------|------|
| `"completed"` | LLM がツール呼び出しなしで応答（正常完了） |
| `"max_steps"` | `max_steps` に到達 |
| `"budget_exhausted"` | `tool_call_budget` を超過 |
| `"timeout"` | `timeout_sec` を超過 |

---

## AgentLoopConfig (dataclass)

エージェントループの設定。

```python
@dataclass
class AgentLoopConfig:
    max_steps: int = 10                              # 最大ステップ数
    tool_call_budget: int = 20                       # ツール呼び出し予算
    observation_max_tokens: int = 2000               # observation トランケーション上限
    timeout_sec: float = 300.0                       # タイムアウト（秒）
    allowed_tools: list[str] | None = None           # 許可ツールリスト
    on_tool_output: Callable[[ToolResult], None] | None = None  # コールバック
```

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `max_steps` | `10` | ループの最大反復回数 |
| `tool_call_budget` | `20` | ループ全体でのツール呼び出し上限 |
| `observation_max_tokens` | `2000` | 各ツール出力のトークン上限 |
| `timeout_sec` | `300.0` | ループ全体の壁時計タイムアウト |

---

## AgentLoop

### コンストラクタ

```python
def __init__(
    self,
    agent_llm: AgentLLM,
    tool_executor: ToolExecutor,
    config: AgentLoopConfig | None = None,
    log_path: Any = None,
)
```

- `agent_llm`: ツール呼び出し付き生成を行う LLM インターフェース
- `tool_executor`: ツール呼び出しをディスパッチするエグゼキュータ
- `config`: ループ設定（`None` の場合デフォルト `AgentLoopConfig()` を使用）
- `log_path`: `agent_loop_log.jsonl` の保存先

### run(task_prompt, purpose, available_tools, adapter_node_id, node_id) -> AgentLoopResult [async]

エージェントループを完了または終了条件まで実行する。

**パラメータ:**

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `task_prompt` | `str` | タスクを記述する初期プロンプト |
| `purpose` | `str` | ログ用のパーパスタグ |
| `available_tools` | `list[str] \| None` | 使用可能ツールの制限 |
| `adapter_node_id` | `str \| None` | 生成時にロードする LoRA アダプタ |
| `node_id` | `str \| None` | ログに記録するノード ID |

**処理フロー:**

1. 各ステップで以下を繰り返す:
   - タイムアウトチェック（`timeout_sec` 超過で `"timeout"` 終了）
   - `agent_llm.generate_with_tools()` でツール呼び出し付き生成
   - ツール呼び出しがある場合:
     - 各ツール呼び出しに対して予算チェック（`tool_call_budget` 超過で `"budget_exhausted"` 終了）
     - 許可ツールフィルタリング
     - `tool_executor.execute()` でディスパッチ
     - `on_tool_output` コールバック呼び出し（設定されている場合）
     - observation をフォーマットしてコンテキストに追加
   - ツール呼び出しがない場合: `"completed"` で終了
2. `max_steps` に到達: `"max_steps"` で終了

**ツールスキーマの解決:** `available_tools`（ツール名のリスト）は `get_tool_schemas()` で OpenAI 形式の辞書リストに変換され、`generate_with_tools()` に渡される。

### _format_observations(tool_results) -> str

ツール実行結果を LLM コンテキスト用のテキストにフォーマットする。

**出力フォーマット:**

```
Tool Results:
[1] tool_name (call_id: abc12345)
  Status: success
  Output:
    {...}
```

EXECUTION ツールの場合は `stdout`（末尾 20 行）と `stderr`（末尾 10 行）のプレビューが追加される。

---

## ログフォーマット（agent_loop_log.jsonl）

```json
{
  "event": "agent_loop_complete",
  "purpose": "search_draft_step0",
  "total_steps": 3,
  "total_tool_calls": 5,
  "total_wall_time_sec": 12.5,
  "exit_reason": "completed",
  "tools_used": {"read_file": 2, "execute_code_snippet": 3},
  "node_id": "abc123"
}
```
