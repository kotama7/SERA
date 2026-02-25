# MCPToolProvider / MCPConfig / MCPToolSchema / MCPToolResult

MCP (Model Context Protocol) クライアントモジュールのドキュメント。外部 MCP サーバへの接続、ツール検出、ツール実行を担当する。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `MCPConfig` | `src/sera/agent/mcp_client.py` |
| `MCPToolSchema` | `src/sera/agent/mcp_client.py` |
| `MCPToolResult` | `src/sera/agent/mcp_client.py` |
| `MCPToolProvider` | `src/sera/agent/mcp_client.py` |

## 依存関係

- `httpx` -- MCP サーバとの HTTP 通信（オプション。未インストール時はスタブモードで動作）
- `sera.agent.tool_executor` (`ToolExecutor.add_mcp_provider()`) -- SERA ツールフレームワークへのブリッジ

---

## MCPConfig (frozen dataclass)

MCP サーバへの接続設定を保持するイミュータブルなデータクラス。

```python
@dataclass(frozen=True)
class MCPConfig:
    server_url: str = ""
    auth_token: str | None = None
    timeout_sec: float = 30.0
    name: str = "default"
    allowed_tools: list[str] = field(default_factory=list)
```

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `server_url` | `str` | `""` | MCP サーバの URL（例: `http://localhost:8080`） |
| `auth_token` | `str \| None` | `None` | 認証トークン（Bearer トークンとして HTTP ヘッダに付与） |
| `timeout_sec` | `float` | `30.0` | MCP サーバへのリクエストタイムアウト（秒） |
| `name` | `str` | `"default"` | サーバ接続の識別名 |
| `allowed_tools` | `list[str]` | `[]` | 使用許可するツール名のリスト。空の場合は全ツールを許可 |

---

## MCPToolSchema

MCP サーバから検出されたツールのスキーマを表すデータクラス。

```python
@dataclass
class MCPToolSchema:
    name: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    server_name: str = ""
```

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `name` | `str` | -- | ツール名（サーバ内で一意） |
| `description` | `str` | `""` | ツールの説明文 |
| `parameters` | `dict` | `{"type": "object", "properties": {}}` | ツール入力パラメータの JSON Schema |
| `server_name` | `str` | `""` | ツールを提供する MCP サーバ名 |

### to_openai_schema() -> dict

OpenAI function-calling スキーマ形式に変換する。

```python
{
    "name": self.name,
    "description": self.description,
    "parameters": self.parameters,
}
```

---

## MCPToolResult

MCP 経由でのツール実行結果を表すデータクラス。

```python
@dataclass
class MCPToolResult:
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    wall_time_sec: float = 0.0
```

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `tool_name` | `str` | -- | 実行されたツール名 |
| `success` | `bool` | -- | 実行が成功したか |
| `output` | `Any` | `None` | ツールの出力（dict, str, list 等） |
| `error` | `str \| None` | `None` | 失敗時のエラーメッセージ |
| `wall_time_sec` | `float` | `0.0` | 実行にかかった時間（秒） |

---

## MCPToolProvider

MCP サーバに接続してツールの検出・実行を行うクラス。外部 MCP ツールを SERA の `ToolExecutor` フレームワークにブリッジする。

httpx の `AsyncClient` を使用して MCP サーバと通信する。テスト用のモックハンドラ登録機能も提供する。

### コンストラクタ

```python
def __init__(self, config: MCPConfig) -> None
```

**内部状態:**

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `_config` | `MCPConfig` | サーバ接続設定 |
| `_connected` | `bool` | 接続済みかどうか |
| `_discovered_tools` | `dict[str, MCPToolSchema]` | 検出済みツールのマップ |
| `_mock_handlers` | `dict[str, Any]` | テスト用モックハンドラ（ツール名 -> callable） |
| `_http_client` | `httpx.AsyncClient \| None` | HTTP クライアント |

### プロパティ

| プロパティ | 型 | 説明 |
|-----------|-----|------|
| `config` | `MCPConfig` | サーバ接続設定を返す |
| `is_connected` | `bool` | 接続済みかどうかを返す |
| `server_name` | `str` | `config.name` を返す |

---

### connect() -> bool

MCP サーバへの接続を試行する非同期メソッド。

**処理フロー:**

1. `server_url` が空の場合: 警告ログを出力し `False` を返す
2. `httpx.AsyncClient` を初期化（`base_url`, `timeout`, `headers` を設定）
3. HTTP GET `/health` でヘルスチェックを実行
4. ステータスコード < 400: 接続成功
5. ステータスコード >= 400: 警告ログを出力するが、**接続済みとしてマーク**する（サーバが `/health` を持たない場合に備える）

**フォールバック動作:**

| 状況 | 動作 |
|------|------|
| ヘルスチェック成功 (< 400) | `connected = True`, `return True` |
| ヘルスチェック失敗 (>= 400) | `connected = True`, `return True`（警告ログ） |
| `httpx` 未インストール | `connected = True`, `return True`（スタブモード） |
| 接続例外（ネットワークエラー等） | `connected = True`, `return True`（実際のエラーはツール検出/実行時に発覚） |
| `server_url` が空 | `return False` |

---

### disconnect() -> None

MCP サーバから切断する非同期メソッド。

- `httpx.AsyncClient` を `aclose()` で閉じる
- `_connected = False` に設定
- `_discovered_tools` をクリア

---

### discover_tools() -> list[MCPToolSchema]

MCP サーバから利用可能なツールを検出する非同期メソッド。

**処理フロー:**

1. 未接続かつモックハンドラがない場合: 空リストを返す
2. **HTTP 検出**: `POST /tools/list` でサーバからツール一覧を取得
   - レスポンス形式: `{"tools": [...]}` または直接リスト
   - 各ツールの `inputSchema` または `parameters` フィールドをパラメータスキーマとして使用
3. **モックハンドラの補完**: HTTP で検出されなかったモックツールをリストに追加（モックは HTTP 検出結果をオーバーライドしない）
4. **`allowed_tools` によるフィルタリング**: `config.allowed_tools` が非空の場合、許可されたツールのみを返す
5. 検出結果を `_discovered_tools` に保存

**エラー時の動作:** HTTP リクエスト失敗時はデバッグログを出力し、モックハンドラのみの結果を返す。

---

### execute(tool_name, params=None) -> MCPToolResult

ツール呼び出しを MCP サーバに転送して実行する非同期メソッド。

**実行優先順位:**

1. **`allowed_tools` チェック**: ツールが許可リストに含まれない場合は `success=False` で即座に返す
2. **モックハンドラ**（最優先）: `_mock_handlers` にハンドラが登録されている場合はそちらを使用。同期/非同期の両方に対応
3. **HTTP 実行**: `POST /tools/call` にリクエスト送信

**HTTP 実行の詳細:**

リクエストボディ:
```json
{"name": "tool_name", "arguments": {"param1": "value1"}}
```

**MCP プロトコルのレスポンスパース:**

- レスポンスの `content` フィールドはブロックの配列（text/image 型）
- 最初のブロックの `text` フィールドを出力として取得
- `isError` フラグで成功/失敗を判定

```python
content = data.get("content", data)
if isinstance(content, list) and content:
    first = content[0]
    output = first.get("text", first) if isinstance(first, dict) else first
is_error = data.get("isError", False)
```

**エラーハンドリング:**

| 状況 | 結果 |
|------|------|
| `allowed_tools` 外のツール | `success=False`, エラーメッセージ |
| モックハンドラが例外を発生 | `success=False`, 例外メッセージ |
| HTTP ステータス >= 400 | `success=False`, `"HTTP {status}: {body[:200]}"` |
| HTTP リクエスト失敗 | `success=False`, `"HTTP request failed: {exc}"` |
| MCP レスポンスの `isError=True` | `success=False`, 出力をエラーとして返す |
| 未接続かつハンドラなし | `success=False`, `"Not connected to MCP server"` |
| 接続済みだがハンドラなし | `success=False`, `"No handler available"` |

全てのケースで `wall_time_sec` は `time.monotonic()` で正確に計測される。

---

### register_mock_tool(name, handler, schema=None) -> None

テスト用のモックツールハンドラを登録する。

```python
def register_mock_tool(
    self,
    name: str,
    handler: Any,
    schema: MCPToolSchema | None = None,
) -> None
```

- `handler`: `(params: dict) -> Any` 形式の callable。同期/非同期の両方をサポート
- `schema`: 省略時は `"Mock MCP tool: {name}"` の説明を持つデフォルトスキーマを自動生成
- モックハンドラは `execute()` で HTTP よりも優先される

---

### tool_names() -> list[str]

検出済みツールの名前リストを返す。

---

## ToolExecutor との統合

`ToolExecutor.add_mcp_provider()` を使用して MCP プロバイダを SERA のツールフレームワークに登録する。

```python
# 使用例
from sera.agent.mcp_client import MCPConfig, MCPToolProvider

config = MCPConfig(server_url="http://localhost:8080", name="custom_server")
provider = MCPToolProvider(config)
await provider.connect()
await provider.discover_tools()

tool_executor.add_mcp_provider(provider)
```

登録後、MCP ツールはビルトインツールと同じインターフェースで `ToolExecutor` からディスパッチ可能になる。ツール名の衝突がある場合、ビルトインツールが優先される。

---

## テスト方法

```python
# モックハンドラを使用したテスト例
config = MCPConfig(name="test_server")
provider = MCPToolProvider(config)

provider.register_mock_tool("my_tool", lambda params: {"result": "ok"})
tools = await provider.discover_tools()
result = await provider.execute("my_tool", {"input": "test"})
assert result.success is True
assert result.output == {"result": "ok"}
```
