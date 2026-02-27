# AgentFunction / AgentFunctionRegistry / REGISTRY

SERA の全構造化 LLM 呼び出しを定義・管理するレジストリシステム。各 LLM 関数のスキーマ、パースユーティリティ、プロバイダ別フォーマット変換を提供する。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `AgentFunction` / `AgentFunctionRegistry` / `REGISTRY` | `src/sera/agent/agent_functions.py` |
| 関数定義（6 サブモジュール） | `src/sera/agent/functions/` |

## 依存関係

- `json`, `re` -- JSON パース・正規表現
- `sera.agent.functions` -- 19 個の登録関数（サイドエフェクトインポート）

---

## OutputMode (Enum)

LLM 応答の解釈方法を定義する列挙型。

| 値 | 説明 |
|----|------|
| `JSON` | JSON としてパース |
| `CODE` | コードブロックとして抽出 |
| `FREE_TEXT` | そのままテキストとして使用 |

---

## AgentFunction (frozen dataclass)

単一の LLM 関数呼び出しのスキーマ定義。

```python
@dataclass(frozen=True)
class AgentFunction:
    name: str                              # 一意な snake_case 識別子
    description: str                       # LLM に提示される説明
    parameters: dict[str, Any]             # JSON Schema (OpenAI function-calling 形式)
    return_schema: dict[str, Any] | None   # 戻り値の JSON Schema
    output_mode: OutputMode                # 応答のパースモード
    phase: str                             # 論理グループ: "search", "execution", etc.
    default_temperature: float             # デフォルト温度 (0.7)
    max_retries: int                       # パース/検証失敗時の最大リトライ回数 (3)
    handler: Callable | None               # 後処理ハンドラ
    allowed_tools: list[str] | None        # AgentLoop 用のツールリスト
    loop_config: dict[str, Any] | None     # AgentLoop のカスタム設定
```

---

## AgentFunctionRegistry

全 `AgentFunction` 定義の中央レジストリ。

### コンストラクタ

```python
def __init__(self) -> None
```

内部辞書 `_functions: dict[str, AgentFunction]` を初期化する。

### register(func) -> None

関数を登録する。名前の重複時は `ValueError` を送出する。

### get(name) -> AgentFunction

名前で関数を取得する。未登録の場合は `KeyError` を送出する。

### list_all() -> list[AgentFunction]

全登録関数をリストで返す。

### list_by_phase(phase) -> list[AgentFunction]

指定 `phase` に属する関数を返す。

### list_by_mode(mode) -> list[AgentFunction]

指定 `output_mode` 値にマッチする関数を返す。

### to_openai_tools(names) -> list[dict]

選択（または全）関数を OpenAI tool-calling 形式に変換する。

```python
{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
```

### to_anthropic_tools(names) -> list[dict]

選択（または全）関数を Anthropic tool 形式に変換する。

```python
{"name": ..., "description": ..., "input_schema": ...}
```

### to_prompt_schema(names) -> str

関数スキーマのテキスト記述を生成する（ローカルプロバイダのプロンプト注入用）。

---

## REGISTRY (モジュールレベルシングルトン)

```python
REGISTRY = AgentFunctionRegistry()
```

`sera.agent.functions` のインポート時にサイドエフェクトで全関数が登録される。

---

## register_function (デコレータ)

関数ハンドラを `REGISTRY` に登録するデコレータ。

```python
@register_function("search_draft", "Draft new hypotheses", ...)
def _handle_search_draft(response: str) -> list[dict]:
    return parse_json_response(response) or []
```

デコレートされた callable は `AgentFunction.handler` として設定される。

---

## 登録関数一覧（19 関数）

| サブモジュール | 関数名 | phase | output_mode |
|--------------|--------|-------|-------------|
| `phase0_functions` (2) | `query_generation` | phase0 | JSON |
| | `paper_clustering` | phase0 | JSON |
| `search_functions` (3) | `search_draft` | search | JSON |
| | `search_debug` | search | JSON |
| | `search_improve` | search | JSON |
| `execution_functions` (1) | `experiment_code_gen` | execution | CODE |
| `spec_functions` (2) | `spec_generation_problem` | spec | JSON |
| | `spec_generation_plan` | spec | JSON |
| `paper_functions` (8) | `paper_outline` | paper | JSON |
| | `paper_draft` | paper | FREE_TEXT |
| | `paper_reflection` | paper | JSON |
| | `aggregate_plot_generation` | paper | CODE |
| | `aggregate_plot_fix` | paper | CODE |
| | `citation_identify` | paper | JSON |
| | `citation_select` | paper | JSON |
| | `citation_bibtex` | paper | FREE_TEXT |
| `evaluation_functions` (3) | `paper_review` | evaluation | JSON |
| | `paper_review_reflection` | evaluation | JSON |
| | `meta_review` | evaluation | FREE_TEXT |

---

## パースユーティリティ

### parse_json_response(response) -> dict | list | None

LLM 応答から JSON を 3 段階フォールバックで抽出する。

**処理フロー:**

1. `` ```json ... ``` `` フェンスブロックを検索
2. 応答全体を `json.loads` でパース
3. 正規表現で `[...]` または `{...}` を検索

いずれも失敗した場合は `None` を返す。

### extract_code_block(response, language="python") -> str

LLM 応答からフェンスコードブロックを抽出する。

**優先順位:**

1. 言語指定フェンス（例: `` ```python ... ``` ``）
2. 汎用フェンス（`` ``` ... ``` ``）
3. 応答テキストそのまま

### validate_against_schema(data, schema) -> tuple[bool, list[str]]

軽量な JSON Schema 検証。`jsonschema` 依存なし。

**サポートする型:** `object`, `array`, `string`, `number`, `integer`, `boolean`

**検証内容:**
- トップレベルの `type` チェック
- `object` の `required` キーチェック
- `properties` の再帰的な型検証
- `array` の `items` の再帰的な型検証

**戻り値:** `(is_valid, errors)` タプル。`errors` はヒューマンリーダブルなエラーメッセージのリスト。
