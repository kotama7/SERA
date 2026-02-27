# ExperimentGenerator / GeneratedExperiment / GeneratedFile

LLM 駆動の実験コード生成モジュール。検索ノードから実験スクリプトを生成し、マルチファイルプロジェクトのバリデーションとディスク書き込みを行う。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `ExperimentGenerator` | `src/sera/execution/experiment_generator.py` |

## 依存関係

- `sera.prompts` (`get_prompt`) -- プロンプトテンプレート
- `sera.agent.agent_llm` (`AgentLLM`) -- LLM インターフェース（`call_function` / `generate`）

---

## GeneratedFile (dataclass)

LLM が生成した単一ファイル。

```python
@dataclass
class GeneratedFile:
    relative_path: str   # runs/{node_id}/ からの相対パス (例: "experiment.py", "src/utils.py")
    content: str         # ファイル内容
```

---

## GeneratedExperiment (dataclass)

LLM が生成した完全な実験プロジェクト。

```python
@dataclass
class GeneratedExperiment:
    entry_point: str                       # メインスクリプトの相対パス (例: "experiment.py")
    files: list[GeneratedFile]             # 生成された全ファイル
```

---

## ExperimentGenerator

### コンストラクタ

```python
def __init__(
    self,
    agent_llm: Any,          # generate() または call_function() を持つ LLM クライアント
    problem_spec: Any,        # ProblemSpec
    work_dir: str | Path = "./sera_workspace",
)
```

### generate(node) -> GeneratedExperiment [async]

検索ノードの実験コードを生成する。

**処理フロー:**

1. `problem_spec.language` から言語設定を取得（デフォルト: Python）
2. `runs/<node_id>/` ディレクトリを作成
3. コード生成パスの決定:
   - `node.experiment_code` が既存（debug 操作等）: そのコードを使用
   - `lang_config.multi_file=True`: マルチファイル生成（`_generate_multi_file`）
   - それ以外: 単一ファイル生成（`_generate_code`）
4. `_write_files()` で全ファイルをディスクに書き込み

**戻り値:** エントリポイントとファイルリストを含む `GeneratedExperiment`

### _generate_multi_file(node, lang_config, default_entry) -> GeneratedExperiment [async]

マルチファイル実験を LLM 経由で生成する。

**処理フロー:**

1. `_generate_code()` で LLM にコード生成を依頼
2. `_parse_multi_file_json()` で JSON 構造としてパースを試行
3. パース失敗時: 単一ファイルにフォールバック

**期待する JSON フォーマット:**

```json
{
  "entry_point": "experiment.py",
  "files": [
    {"path": "experiment.py", "content": "..."},
    {"path": "src/utils.py", "content": "..."}
  ]
}
```

### _write_files(run_dir, experiment, lang_config) -> None

生成ファイルをバリデーションしてディスクに書き込む。

**バリデーション:**

| チェック | 条件 | エラー |
|---------|------|--------|
| entry_point 存在確認 | `entry_point` がファイルリストに含まれるか | `ValueError` |
| ファイル数上限 | `max_files`（デフォルト: 10） | `ValueError` |
| 合計サイズ上限 | `max_total_size_bytes`（デフォルト: 1 MB） | `ValueError` |
| パストラバーサル | `..` を含むパス | `ValueError` |

### _generate_code(node) -> str [async]

LLM を呼び出して実験コードを生成する。

**処理フロー:**

1. 言語設定、問題記述、目的関数、テンプレート等からプロンプトを構築
2. `call_function("experiment_code_gen", ...)` で構造化呼び出し（推奨パス）
3. `call_function` がない場合: `generate()` + `_extract_code()` のレガシーパス

### _extract_code(response, code_block_tag) -> str [staticmethod]

LLM 応答からコードを抽出する。

**優先順位:**

1. 言語指定フェンスブロック（例: `` ```python ... ``` ``）
2. 汎用フェンスブロック（`` ``` ... ``` ``）
3. 応答テキストそのまま

---

## 多言語サポート

`ProblemSpec.language`（`LanguageConfig`）から以下を取得:

| フィールド | 説明 | デフォルト |
|-----------|------|----------|
| `name` | 言語名 | `"python"` |
| `interpreter_command` | インタープリタコマンド | `"python"` |
| `file_extension` | ファイル拡張子 | `".py"` |
| `seed_arg_format` | シード引数フォーマット | `"--seed {seed}"` |
| `code_block_tag` | コードブロックタグ | `"python"` |
| `multi_file` | マルチファイル生成の有効化 | `True` |
