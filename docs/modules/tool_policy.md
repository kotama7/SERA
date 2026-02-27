# ToolPolicy

ツール実行の安全性制御を担当するポリシーモジュール。フェーズ別許可リスト、書き込みパス制限、シェルコマンドホワイトリスト、API レート制限、パストラバーサル防止を提供する。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `ToolPolicy` | `src/sera/agent/tool_policy.py` |

## 依存関係

- `fnmatch` -- ファイルパターンマッチング
- `time` -- レート制限用タイムスタンプ
- `sera.specs` -- `PlanSpec`, `ResourceSpec`（ファクトリメソッド経由）

---

## ToolPolicy (dataclass)

ツール実行の安全性とリソース制限を制御するポリシークラス。

### コンストラクタ

```python
@dataclass
class ToolPolicy:
    tools_enabled: bool = True
    disabled_tools: set[str] = field(default_factory=set)
    phase_allowed_tools: dict[str, list[str]] = field(default_factory=dict)
    max_file_read_bytes: int = 1_000_000       # 1 MB
    max_file_write_bytes: int = 500_000         # 500 KB
    max_output_tokens: int = 2000               # observation truncation
    allowed_write_dirs: list[str]               # デフォルト: ["runs/", "paper/", "outputs/"]
    blocked_write_patterns: list[str]           # デフォルト: ["specs/*.yaml", "*.lock", "*.jsonl"]
    allowed_shell_commands: list[str]           # デフォルト: ["pip", "python", "ls", "cat", "wc"]
    allowed_build_commands: list[str]           # デフォルト: ["g++", "gcc", "clang++", "clang", "cargo", "rustc", "go", "make", "cmake"]
    compiled_language: bool = False
    api_rate_limit_per_minute: int = 30
    api_rate_limit_burst: int = 5
    allow_network: bool = True
    allow_api_calls: bool = True
    allowed_domains: list[str] = field(default_factory=list)
```

### デフォルト値一覧

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| `tools_enabled` | `True` | ツールシステム全体の有効/無効 |
| `max_file_read_bytes` | `1,000,000` | ファイル読み取り上限 (1 MB) |
| `max_file_write_bytes` | `500,000` | ファイル書き込み上限 (500 KB) |
| `max_output_tokens` | `2000` | observation トランケーション上限 |
| `api_rate_limit_per_minute` | `30` | 外部 API コール制限（スライディングウィンドウ） |
| `compiled_language` | `False` | ビルドコマンドの有効化フラグ |

---

### from_specs(plan_spec, resource_spec) -> ToolPolicy [classmethod]

Spec オブジェクトから `ToolPolicy` を構築するファクトリメソッド。

**読み取る設定:**

| ソース | フィールド | ToolPolicy フィールド |
|--------|---------|---------------------|
| `plan_spec.tools` | `enabled` | `tools_enabled` |
| `plan_spec.tools` | `api_rate_limit_per_minute` | `api_rate_limit_per_minute` |
| `plan_spec.agent_commands.tools` | `phase_tool_map` | `phase_allowed_tools` |
| `resource_spec.network` | `allow_internet` | `allow_network` |
| `resource_spec.network` | `allow_api_calls` | `allow_api_calls` |

### from_specs_with_problem(plan_spec, resource_spec, problem_spec) -> ToolPolicy [classmethod]

`from_specs` に加えて `ProblemSpec.language.compiled` フラグを読み取り、`compiled_language` を設定する。

---

### check_tool_allowed(tool_name, phase) -> tuple[bool, str]

指定されたツールが現在のフェーズで許可されているか判定する。

**チェック順序:**

1. `tools_enabled` が `False` → 拒否
2. `tool_name` が `disabled_tools` に含まれる → 拒否
3. `phase` が `phase_allowed_tools` に存在し、`tool_name` がリストに含まれない → 拒否
4. それ以外 → 許可

### check_network_allowed(tool_name) -> tuple[bool, str]

ネットワークアクセスが必要なツールに対するネットワークポリシーを適用する。

**対象ツール（API ツール）:**

- `semantic_scholar_search`, `semantic_scholar_references`, `semantic_scholar_citations`
- `crossref_search`, `arxiv_search`, `web_search`

`allow_network=False` または `allow_api_calls=False` の場合、これらのツールはブロックされる。

### check_write_path(relative_path) -> tuple[bool, str]

書き込みパスの許可判定を行う。

**チェック順序:**

1. `blocked_write_patterns` にマッチ → 拒否（`fnmatch` 使用）
2. `allowed_write_dirs` のいずれかで始まる → 許可
3. それ以外 → 拒否

### check_shell_command(command) -> tuple[bool, str]

シェルコマンドのホワイトリストチェックを行う。

**チェック順序:**

1. 実行ファイル名が `allowed_shell_commands` に含まれる → 許可
2. `compiled_language=True` かつ `allowed_build_commands` に含まれる → 許可
3. それ以外 → 拒否

### check_api_rate_limit() -> tuple[bool, str]

スライディングウィンドウ方式の API レート制限チェック。60 秒以内のコール数が `api_rate_limit_per_minute` 以上の場合に拒否。

### record_api_call() -> None

API コールのタイムスタンプを記録する（レート制限カウント用）。

### resolve_safe_path(workspace, relative_path) -> Path

パストラバーサル防止付きでパスを解決する。解決後のパスがワークスペース外の場合は `PermissionError` を送出する。
