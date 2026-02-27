# FailureKnowledgeExtractor / FailureSummary

ECHO lightweight: 失敗ノードから構造化された失敗知識を抽出し、兄弟/子孫ノードに注入することで同一ミスの繰り返しを防止するモジュール。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `FailureKnowledgeExtractor` / `FailureSummary` | `src/sera/search/failure_extractor.py` |

## 依存関係

- なし（外部パッケージ不要。`agent_llm` は将来の LLM ベース抽出用にオプション）

---

## FailureSummary (dataclass)

単一の失敗の構造化サマリ。

```python
@dataclass
class FailureSummary:
    node_id: str              # 失敗ノードの ID
    hypothesis: str           # 失敗した仮説
    error_category: str       # エラーカテゴリ
    error_message: str        # エラーメッセージ（トランケーション済み）
    lesson: str               # 教訓（避けるべきこと / 何が問題だったか）
```

### エラーカテゴリ

| カテゴリ | 説明 |
|---------|------|
| `"runtime"` | ランタイムエラー（例外、トレースバック） |
| `"oom"` | メモリ不足 |
| `"timeout"` | タイムアウト |
| `"logical"` | 論理エラー（NaN、Inf、ダイバージェンス） |
| `"unknown"` | その他 |

### to_dict() -> dict

辞書に変換する。`SearchNode.failure_context` への注入に使用。

### from_dict(d) -> FailureSummary [classmethod]

辞書から復元する。

---

## FailureKnowledgeExtractor

### コンストラクタ

```python
def __init__(
    self,
    echo_config: Any,          # EchoConfig (max_summaries_per_node, summary_max_tokens)
    agent_llm: Any = None,     # オプション: LLM ベースの教訓生成用
)
```

**設定パラメータ:**

| パラメータ | ソース | デフォルト | 説明 |
|-----------|--------|----------|------|
| `max_summaries` | `echo_config.max_summaries_per_node` | `3` | ノードあたりの最大サマリ数 |
| `summary_max_tokens` | `echo_config.summary_max_tokens` | `256` | サマリのトークン上限 |

### extract(failed_node) -> FailureSummary

失敗ノードから構造化された失敗サマリを抽出する。

**処理フロー:**

1. ノードの `error_message` と `status` を取得
2. `_categorise_error()` でエラーカテゴリを判定
3. `_generate_lesson()` で簡潔な教訓を生成
4. `error_message` と `lesson` を `summary_max_tokens` でトランケーション

### inject(summary, siblings) -> None

失敗知識を兄弟ノードに注入する。

**処理フロー:**

1. 各兄弟ノードの `failure_context` リストに対して:
   - `failure_context` 属性がなければスキップ
   - 同一 `node_id` の重複がある場合はスキップ
   - `max_summaries` に達している場合はスキップ
   - `summary.to_dict()` を `failure_context` に追加

---

### _categorise_error(status, error_message) -> str [staticmethod]

ステータスとエラーメッセージからエラーカテゴリを判定するヒューリスティック。

**判定ルール（優先順位順）:**

| 条件 | カテゴリ |
|------|---------|
| `status == "oom"` | `"oom"` |
| `status == "timeout"` | `"timeout"` |
| メッセージに `memory`, `cuda out of memory`, `oom`, `alloc` | `"oom"` |
| メッセージに `timeout`, `timed out`, `deadline` | `"timeout"` |
| メッセージに `runtime`, `exception`, `traceback`, `error` | `"runtime"` |
| メッセージに `nan`, `inf`, `diverge`, `negative loss` | `"logical"` |
| その他 | `"unknown"` |

### _generate_lesson(node, error_category, error_message) -> str [staticmethod]

エラーカテゴリに基づく定型教訓を生成する。

**教訓テンプレート:**

| カテゴリ | テンプレート |
|---------|------------|
| `oom` | "Approach '...' caused OOM. Consider reducing model/batch size. Config: ..." |
| `timeout` | "Approach '...' exceeded time limit. Consider simpler methods or fewer iterations." |
| `runtime` | "Approach '...' raised a runtime error: ..." |
| `logical` | "Approach '...' produced invalid numerical output (NaN/Inf). Check numerical stability." |
| `unknown` | "Approach '...' failed for unknown reasons: ..." |
