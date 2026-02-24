# EvidenceStore

Phase 7 (論文生成) に向けて Phase 2-6 の結果を集約するモジュールのドキュメント。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `EvidenceStore` | `src/sera/paper/evidence_store.py` |

## 依存関係

- `sera.search.search_node` (`SearchNode`) -- ノード型（型アノテーションは `Any`）

---

## EvidenceStore (dataclass)

Phase 2-6 の結果を論文生成に適した形式で集約するデータクラス。

**注意:** 名前に "Store" とあるが、データベースではない。Python のプレーン dataclass であり、メモリ上にデータを保持する。

### フィールド

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `best_node` | `Any` (SearchNode) | `None` | 最良ノード（LCB 最高） |
| `top_nodes` | `list` | `[]` | 上位ノードのリスト |
| `all_evaluated_nodes` | `list` | `[]` | 全評価済みノードのリスト |
| `search_log` | `list[dict]` | `[]` | 探索ログ（search_log.jsonl から読み込み） |
| `eval_log` | `list[dict]` | `[]` | 評価ログ（eval_log.jsonl から読み込み） |
| `ppo_log` | `list[dict]` | `[]` | PPO ログ（ppo_log.jsonl から読み込み） |
| `problem_spec` | `Any` | `None` | ProblemSpec |
| `related_work` | `Any` | `None` | RelatedWorkSpec |
| `execution_spec` | `Any` | `None` | ExecutionSpec |

---

## メソッド

### get_main_results_table() -> str

全評価済みノードをメイン結果テーブル（Markdown 形式）として返す。

**テーブル構造:**

| Method | Metric (mu +/- SE) | LCB | Feasible |
|--------|-------------------|-----|----------|

- LCB 降順でソート
- Method: `experiment_config.get("method")` またはhypothesis の先頭 50 文字
- SE が `inf` の場合は `"N/A"`
- mu / lcb が `None` の場合は `"N/A"`
- Feasible: `"Yes"` / `"No"`

### get_ablation_data() -> dict

最良ノードの子ノード（`branching_op == "improve"` かつ `parent_id == best_node.node_id`）からアブレーションデータを構築する。

**処理:**

1. `best_node` が None の場合は空辞書を返す
2. 各改善子ノードについて、親（best_node）との設定差分を計算
3. 差分がある場合、最初の異なるキーをグループキーとする
4. 各グループのデータ: `{mu, se, lcb, config}` の辞書

**戻り値:** `dict[str, dict]` -- 変数名をキー、統計情報を値とする辞書。

### get_convergence_data() -> list[tuple[int, float]]

探索ログから収束データ（ステップ vs best_lcb の時系列）を抽出する。

**処理:**

- `search_log` を順に走査し、各エントリの `"lcb"` キーの値を取得
- これまでの最良 LCB を追跡し、改善があるたびに更新
- `best_lcb > -inf` の場合にのみデータポイントを追加

**戻り値:** `list[tuple[int, float]]` -- `(step_index, best_lcb_so_far)` のリスト。

### get_experiment_summaries() -> dict[str, list[dict]]

全評価済みノードをステージ別に分類した実験サマリーを返す。

**カテゴリ分類:**

| 条件 | カテゴリ |
|------|---------|
| `branching_op == "draft"` かつ `depth == 0` | `"baseline"` |
| `branching_op == "improve"` | `"research"` |
| その他 | `"research"` |

**各エントリの内容:**

```python
{
    "node_id": str,
    "hypothesis": str,
    "config": dict,       # experiment_config
    "mu": float | None,
    "se": float | None,
    "lcb": float | None,
    "feasible": bool,
    "op": str,            # branching_op
}
```

**戻り値:** `{"baseline": [...], "research": [...], "ablation": []}` -- `"ablation"` カテゴリは常に空リスト（アブレーションデータは `get_ablation_data()` で取得）。

### from_workspace(cls, work_dir) -> EvidenceStore

ワークスペースディレクトリから `EvidenceStore` を構築するクラスメソッド。

**読み込むファイル:**

| ファイルパス | フィールド |
|------------|----------|
| `{work_dir}/logs/search_log.jsonl` | `search_log` |
| `{work_dir}/logs/eval_log.jsonl` | `eval_log` |
| `{work_dir}/logs/ppo_log.jsonl` | `ppo_log` |

各ファイルは JSONL 形式（1 行 1 JSON オブジェクト）で、行ごとに `json.loads()` でパースされる。ファイルが存在しない場合はスキップ。

**注意:** このメソッドはログファイルからの読み込みのみ行い、`SearchNode` オブジェクトの再構築は行わない。そのため `best_node`, `top_nodes`, `all_evaluated_nodes` は空のまま。

### to_json() -> dict

デバッグ用のシリアライズ。以下のサマリー情報を dict として返す:

- `num_evaluated_nodes`: 評価済みノード数
- `search_log_len`: 探索ログのエントリ数
- `eval_log_len`: 評価ログのエントリ数
- `ppo_log_len`: PPO ログのエントリ数
- `best_node_id`: 最良ノードの ID（存在する場合）

---

## 使用箇所

`EvidenceStore` は Phase 7 (`PaperComposer`) から使用される。`PaperComposer.compose()` メソッドが `evidence` 引数として受け取り、以下の情報を論文生成に利用する:

- メイン結果テーブル（`get_main_results_table()`）
- 収束データ（`get_convergence_data()`）
- アブレーションデータ（`get_ablation_data()`）
- 実験サマリー（`get_experiment_summaries()`）
- 最良ノードの詳細（`best_node`）
- 全評価済みノード（`all_evaluated_nodes`）-- 図の生成に使用
