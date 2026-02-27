# ToolExecutor / ToolResult

`ToolCall` オブジェクトを適切なハンドラ関数にディスパッチし、ポリシーチェック・レート制限・ログ記録・ノード使用量トラッキングを行うモジュール。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `ToolExecutor` / `ToolResult` | `src/sera/agent/tool_executor.py` |
| ツールハンドラ（4 カテゴリ） | `src/sera/agent/tools/` |

## 依存関係

- `sera.agent.agent_llm` (`ToolCall`) -- ツール呼び出しデータ構造
- `sera.agent.tool_policy` (`ToolPolicy`) -- 実行ポリシー
- `sera.utils.logging` (`JsonlLogger`) -- JSONL ログ記録
- `sera.agent.tools.search_tools` -- 検索ツールハンドラ
- `sera.agent.tools.execution_tools` -- 実行ツールハンドラ
- `sera.agent.tools.file_tools` -- ファイルツールハンドラ
- `sera.agent.tools.state_tools` -- 状態ツールハンドラ

---

## ToolResult (dataclass)

単一のツール実行結果を表す。

```python
@dataclass
class ToolResult:
    tool_name: str
    call_id: str
    success: bool
    output: Any
    error: str | None = None
    wall_time_sec: float = 0.0
    truncated: bool = False
    stdout_preview: str | None = None    # 実行ツール用（末尾 20 行）
    stderr_preview: str | None = None    # 実行ツール用（末尾 10 行）
    is_execution: bool = False           # 実行ツールフラグ
```

---

## 18 ツール一覧（4 カテゴリ）

### SEARCH ツール（6）

| ツール名 | 説明 |
|---------|------|
| `semantic_scholar_search` | Semantic Scholar 論文検索 |
| `semantic_scholar_references` | 論文の参照文献取得 |
| `semantic_scholar_citations` | 論文の被引用文献取得 |
| `crossref_search` | CrossRef 論文検索 |
| `arxiv_search` | arXiv プレプリント検索 |
| `web_search` | 一般 Web 検索 |

### EXECUTION ツール（3）

| ツール名 | 説明 |
|---------|------|
| `execute_experiment` | 実験スクリプトの実行 |
| `execute_code_snippet` | コードスニペットの実行 |
| `run_shell_command` | ホワイトリストされたシェルコマンドの実行 |

### FILE ツール（5）

| ツール名 | 説明 |
|---------|------|
| `read_file` | ワークスペース内ファイルの読み取り |
| `write_file` | ワークスペース内ファイルへの書き込み |
| `read_metrics` | ノードの `metrics.json` 読み取り |
| `read_experiment_log` | ノードの `stdout.log` / `stderr.log` 読み取り |
| `list_directory` | ディレクトリ内容の一覧 |

### STATE ツール（4）

| ツール名 | 説明 |
|---------|------|
| `get_node_info` | 検索ツリーノードの詳細取得 |
| `list_nodes` | ノード一覧（ステータス/ソートフィルタ付き） |
| `get_best_node` | 現在のベストノード取得 |
| `get_search_stats` | 検索統計の集約情報取得 |

---

## TOOL_SCHEMAS (モジュールレベル辞書)

全 18 ツールの OpenAI function-calling 形式スキーマ。`AgentLoop` が LLM に利用可能なツールを伝えるために使用する。

### get_tool_schemas(tool_names) -> list[dict]

ツール名のリストを受け取り、対応するスキーマを解決する。未知のツール名は警告ログを出力してスキップする。

---

## ToolExecutor

### コンストラクタ

```python
def __init__(
    self,
    workspace_dir: Path,
    policy: ToolPolicy | None = None,
    executor: Executor | None = None,
    scholar_clients: list[BaseScholarClient] | None = None,
    search_manager: SearchManager | None = None,
    log_path: Path | None = None,
)
```

- `workspace_dir`: SERA ワークスペースのルート
- `policy`: 安全ポリシー（`None` の場合デフォルト `ToolPolicy()` を使用）
- `executor`: `execute_experiment` 用の実験実行バックエンド
- `scholar_clients`: 検索ツール用の API クライアント
- `search_manager`: 状態ツール用の検索マネージャ参照
- `log_path`: `tool_execution_log.jsonl` の保存先

初期化時に `_register_handlers()` で全 18 ツールのハンドラを登録する。

### execute(tool_call) -> ToolResult [async]

`ToolCall` をディスパッチし `ToolResult` を返す。

**処理フロー:**

1. ビルトインハンドラを検索
2. 見つからない場合: MCP プロバイダからハンドラを検索
3. ハンドラなし: エラー結果を返す
4. `ToolPolicy.check_tool_allowed()` でグローバル/個別ツールチェック
5. `ToolPolicy.check_network_allowed()` で API ツールのネットワークチェック
6. SEARCH ツールの場合: `check_api_rate_limit()` + `record_api_call()`
7. ハンドラを実行（`await handler(tool_call.arguments)`）
8. 出力をトランケーション（`_truncate_output`）
9. EXECUTION ツールの場合: `stdout_preview`（末尾 20 行）、`stderr_preview`（末尾 10 行）を設定
10. `_update_node_tool_usage()` でノードの使用量を更新
11. `_log_execution()` で JSONL ログを記録

**エラーハンドリング:**

| 例外 | ToolResult |
|------|-----------|
| `PermissionError` | `success=False`, `error="Permission denied: ..."` |
| その他 `Exception` | `success=False`, `error=str(exc)` |

### available_tools(phase) -> list[str]

現在実行可能なツール名を返す。ビルトインツールと MCP プロバイダのツールを含む。`phase` 指定時はフェーズ別フィルタリングを適用。

### total_tool_calls -> int [property]

累計ツールコール数を返す。

---

## MCP 統合

### add_mcp_provider(provider) -> None

MCP ツールプロバイダを登録する。プロバイダのツールはビルトインツールと並行してディスパッチ可能になる。

プロバイダは以下のインターフェースを実装する必要がある:

- `tool_names() -> list[str]` -- 提供するツール名のリスト
- `execute(tool_name, args) -> result` -- ツール実行（`result.success`, `result.output`, `result.error`）

---

## ノード tool_usage トラッキング

### set_current_node_id(node_id) -> None

現在のノード ID を設定する。以降のツール実行がこのノードの `tool_usage` 辞書に記録される。

### _update_node_tool_usage(tool_name, success, wall_time_sec) -> None

`SearchNode.tool_usage` 辞書を更新する。

**トラッキングされるフィールド:**

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `total_tool_calls` | `int` | 全ツールコール数 |
| `total_successes` | `int` | 成功コール数 |
| `tool_success_rate` | `float` | 成功率 |
| `tools_used` | `dict[str, dict]` | ツール別の `calls`, `successes`, `total_latency_sec` |

---

## ログフォーマット（tool_execution_log.jsonl）

```json
{
  "event": "tool_execution",
  "call_id": "<UUID>",
  "tool_name": "read_file",
  "arguments": {"path": "runs/abc/metrics.json"},
  "success": true,
  "output_size_bytes": 256,
  "truncated": false,
  "wall_time_sec": 0.05,
  "error": null
}
```
