# SearchManager / SearchNode / TreeOps / Priority / Validation

Phase 2 のベストファースト木探索を担当するモジュール群のドキュメント。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `SearchManager` | `src/sera/search/search_manager.py` |
| `SearchNode` | `src/sera/search/search_node.py` |
| `TreeOps` | `src/sera/search/tree_ops.py` |
| `compute_priority` | `src/sera/search/priority.py` |
| `validate_experiment_config` | `src/sera/search/validation.py` |

---

## SearchManager

研究ループ全体を統括するベストファースト探索マネージャ。仮説の起草、実験の評価、デバッグ、改善、PPO 更新、枝刈りを調整する。

### コンストラクタ

```python
def __init__(
    self,
    specs,            # AllSpecs -- execution_spec, problem_spec 等を含む
    agent_llm,        # LLM クライアント
    executor,         # 実験実行バックエンド (local/slurm/docker)
    evaluator,        # 統計的評価エンジン
    ppo_trainer,      # PPO トレーナー (None 可)
    lineage_manager,  # LoRA リネージマネージャ (None 可)
    tree_ops,         # TreeOps (draft/debug/improve)
    pruner,           # Pruner (None 可)
    logger_obj=None,  # JSONL ロガー
    checkpoint_dir="./checkpoints",
    failure_extractor=None,       # FailureKnowledgeExtractor (None 可、ECHO 用)
    turn_reward_evaluator=None,   # TurnRewardEvaluator (None 可、MT-GRPO/HiPER 用)
)
```

**新パラメータ**:

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `failure_extractor` | `FailureKnowledgeExtractor \| None` | `None` | ECHO 軽量版。失敗ノードの知識抽出・兄弟注入を行う。`echo.enabled=True` の場合に `research_cmd.py` で初期化される |
| `turn_reward_evaluator` | `TurnRewardEvaluator \| None` | `None` | Phase 毎のターンレベル報酬評価器。`method` が `mt_grpo` / `hiper` かつ `turn_rewards` が設定されている場合に初期化される |

### データ構造

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `open_list` | `list[tuple[float, str]]` | heapq。`(neg_priority, node_id)` のタプル。最小ヒープなので優先度を負にして最大優先度が先に取り出される |
| `closed_set` | `set[str]` | 処理済みノード ID の集合 |
| `all_nodes` | `dict[str, SearchNode]` | 全ノードのマップ |
| `best_node` | `SearchNode \| None` | 現在の最良ノード |
| `step` | `int` | 現在のステップ番号 |
| `ppo_buffer` | `list[dict]` | PPO 更新用のバッファ |

### run() メインループ（非同期）

`SearchNode | None` を返す。

**フロー:**

1. 初期ノード生成: `tree_ops.draft(n_initial, all_nodes)` で `initial_root_children`（デフォルト 5）個のルートノードを生成
2. `_should_terminate()` が `False` の間ループ:
   - `step` をインクリメント
   - `select_next_node()` で `(node, operator)` を取得
   - operator に応じた処理:
     - `"evaluate"` -> `_evaluate_node(node)`
     - `"debug"` -> `tree_ops.debug(node)` で子ノード生成、親の `children_ids` に追加。**ECHO**: `failure_extractor` が存在する場合、`extract(failed_node)` で失敗知識を抽出し、`inject(summary, siblings)` で兄弟ノードに注入
     - `"draft"` -> `tree_ops.draft(branch_factor, all_nodes)` で新規ノード群を生成
     - `"improve"` -> `tree_ops.improve(node, all_nodes, n_children)` で改善ノード群を生成、親の status を `"expanded"` に変更
   - ノード状態をログ出力（step, node_id, operator, status, mu, se, lcb, priority 等）
   - PPO 更新: `ppo_buffer` が空でなく `ppo_trainer.should_update(n_evaluated)` が `True` の場合に `ppo_trainer.update()` を実行。バッファをクリア
   - 枝刈り: `step % prune_interval（デフォルト 10）== 0` の場合に `pruner.prune()` を実行
   - チェックポイント: `step % 10 == 0` の場合に `save_checkpoint()` を実行
3. `best_node` を返す

### select_next_node() -> tuple[SearchNode | None, str]

ノード状態に基づいてオペレータを自動選択する。

**優先順位:**

1. **失敗ノードのデバッグ**: `status == "failed"` かつ `debug_depth < max_debug_depth`（デフォルト 3）のノード -> `"debug"`
2. **多様性チェックによる再起草**: 評価済みノード中のユニークな method 数 < `min_diverse_methods`（デフォルト 3）かつ評価済み数 >= `draft_trigger_after`（デフォルト 10）-> `"draft"`
3. **ヒープからポップ**: `status == "pending"` -> `"evaluate"`、`status == "evaluated"` -> `"improve"`
4. **フォールバック**: ノード数 < `max_nodes` の場合 -> `"draft"`

ノードが取得できない場合は `(None, "")` を返す。

### _should_terminate() -> bool

終了条件の判定。以下のいずれかが成立すると `True` を返す:

1. `len(all_nodes) < min_nodes_before_stop`（デフォルト 10）の場合は常に `False`（早期終了を防止）
2. `step >= max_steps`（デフォルトは `max_nodes`、100）
3. `len(all_nodes) >= max_nodes`（デフォルト 100）
4. `max_wall_time_hours` 経過（経過時間をモノトニッククロックで計測）
5. プラトー検出: `stop_on_plateau`（デフォルト `True`）が有効かつ `plateau_patience`（デフォルト 10）ステップ間に `plateau_min_improvement`（デフォルト 0.001）以上の改善がない場合
6. 予算超過: `pruning.budget_limit.limit` が設定されており、全ノードの `total_cost` 合計が上限を超えた場合（`_budget_exceeded` フラグを設定）
7. `open_list` が空で、pending ノードもデバッグ可能ノードも拡張可能ノードもない場合

### _evaluate_node(node)

ノードの評価処理。

1. `node.status = "running"` に設定
2. `evaluator.evaluate_initial(node)` を実行
3. 失敗/OOM の場合: `closed_set` から除外して早期リターン（debug オペレータが拾えるようにする）
4. top-k 判定（`_is_topk`）: True なら `evaluator.evaluate_full(node)` を実行
5. `node.mark_evaluated()` で status を `"evaluated"` に変更
6. `_update_best(node)` で最良ノードを更新
7. 優先度を再計算してヒープに再投入
8. **ターン報酬計算**: `turn_reward_evaluator` が存在する場合、`evaluate_all(node, parent, all_nodes)` で Phase 毎のターン報酬を計算
9. `ppo_buffer` に追加（`node.mu` が None でない場合）。ターン報酬が計算された場合は `PPORolloutV2` を使用し、`turn_rewards` を含める

### _update_best(node)

LCB が最高のノードを `best_node` として更新。

- `lcb` が None または `feasible` が False のノードは無視
- LCB が同値の場合は `mu` で比較
- 更新時に `ppo_trainer.notify_step(best_lcb)` を呼び出す（プラトー検知用）

### SIGINT ハンドリング

`signal.SIGINT` を捕捉し、チェックポイントを保存して `sys.exit(20)` で終了する。

### sync_adapter_assignment (モジュールレベル関数)

デュアルツリー同期を行う関数。`_add_node()` とPPO更新後に呼び出される。

```python
def sync_adapter_assignment(
    search_node: SearchNode,
    ppo_updated: bool,
    new_adapter_node_id: str | None,
    all_nodes: dict[str, SearchNode],
) -> None
```

**ルール:**
1. ルートノード (`parent_id is None`) → `adapter_node_id = "adapter_root"`
2. PPO更新あり (`ppo_updated=True`, `new_adapter_node_id` 非 None) → 新しい ID を割り当て
3. それ以外 → 親の `adapter_node_id` を継承

### _build_tool_trajectory(loop_result) -> list[dict]

`AgentLoopResult` からツール使用軌跡を構築するメソッド。`PPORolloutV3` の `tool_trajectory` フィールドに格納される。各ツール呼び出しの `tool_name`, `success`, `wall_time_sec` を記録する。

### _run_post_step_tasks(exec_spec)

ステップ後のPPO更新・プルーニング・チェックポイントを実行する非同期メソッド。メインループの各ステップ終了時に呼び出される。

### _run_batched_pipeline(batch_size, max_concurrent, poll_interval_sec)

SLURM 非同期バッチパイプライン。`SlurmExecutor` 使用時に複数の pending ノードを一括処理する非同期メソッド。

1. `open_list` から `batch_size` 個の pending ノードを収集
2. 実験コード生成（`ExperimentGenerator`）
3. ジョブ一括投入 → 非同期ポーリング → 結果収集
4. 各ノードの評価・PPOバッファへの追加

### _needs_diversity_draft(exec_spec) -> bool

多様性ドラフトが必要かどうかを判定するメソッド。評価済みノード中のユニークメソッド数が `min_diverse_methods` 未満かつ評価済みノード数が `draft_trigger_after` 以上の場合に `True` を返す。

### save_state() / load_state(state)

チェックポイントのシリアライズ/デシリアライズ。保存される情報: `step`, `all_nodes`（各ノードの `to_dict()`）, `closed_set`, `best_node_id`, `open_list`, `ppo_buffer`。

---

## SearchNode

探索木の 1 ノードを表す dataclass。`to_dict()` / `from_dict()` でシリアライズ/デシリアライズ可能。

### フィールド

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `node_id` | `str` | UUID4 | ノードの一意識別子 |
| `parent_id` | `str \| None` | `None` | 親ノード ID |
| `depth` | `int` | `0` | 木の深さ |
| `created_at` | `str` | ISO8601 UTC | 作成日時 |
| `hypothesis` | `str` | `""` | 仮説 |
| `experiment_config` | `dict` | `{}` | 実験設定（操作変数の値） |
| `experiment_code` | `str \| None` | `None` | 生成された実験コード |
| `branching_op` | `str` | `"draft"` | 生成オペレータ (`"draft"` / `"debug"` / `"improve"`) |
| `rationale` | `str` | `""` | アプローチの理由 |
| `adapter_node_id` | `str \| None` | `None` | リネージツリーへのリンク |
| `eval_runs` | `int` | `0` | 評価実行回数 |
| `metrics_raw` | `list[dict]` | `[]` | 各実行の生メトリクス |
| `mu` | `float \| None` | `None` | 平均値 |
| `se` | `float \| None` | `None` | 標準誤差 |
| `lcb` | `float \| None` | `None` | 下側信頼限界 |
| `total_cost` | `float` | `0.0` | 累積コスト |
| `wall_time_sec` | `float` | `0.0` | 累積実行時間（秒） |
| `priority` | `float \| None` | `None` | 探索優先度 |
| `status` | `str` | `"pending"` | 状態 (`pending`/`running`/`evaluated`/`failed`/`pruned`/`expanded`/`timeout`/`oom`) |
| `children_ids` | `list[str]` | `[]` | 子ノード ID リスト |
| `feasible` | `bool` | `True` | 実行可能性制約の充足 |
| `debug_depth` | `int` | `0` | デバッグ深度 |
| `error_message` | `str \| None` | `None` | エラーメッセージ |
| `failure_context` | `list[dict]` | `[]` | ECHO 軽量版で注入された失敗知識のリスト。各要素は `FailureSummary.to_dict()` 形式（`node_id`, `hypothesis`, `error_category`, `error_message`, `lesson`） |

### メソッド

- `add_metric(metric: dict)`: `metrics_raw` に追加し `eval_runs` を更新
- `mark_failed(error_message: str)`: status を `"failed"` に変更
- `mark_evaluated()`: status を `"evaluated"` に変更
- `to_dict() -> dict`: 全フィールドを dict にシリアライズ
- `from_dict(cls, d: dict) -> SearchNode`: 未知のキーは無視してデシリアライズ

---

## TreeOps

AIDE 着想の分岐オペレータ（draft, debug, improve）を実装するクラス。**独自のインラインプロンプトテンプレート**（`DRAFT_PROMPT`, `DRAFT_CATEGORY_PROMPT`, `DEBUG_PROMPT`, `IMPROVE_PROMPT`）を持ち、`prompt_templates.py` とは別のテンプレートを使用する。

### コンストラクタ

```python
def __init__(self, specs, agent_llm, rng=None, agent_loop=None)
```

- `agent_loop`: `AgentLoop` インスタンス（`None` 可）。ツールバインディングが設定された関数（`function_tool_bindings` に登録済み）の場合に ReAct ループで実行される

### draft(n, all_nodes=None) -> list[SearchNode]

新規アプローチを起草する非同期メソッド。

**ルート初期化時**（`all_nodes` が空/None）:

- n を 3 カテゴリに分割: `baseline`（n//3）, `open_problem`（n//3）, `novel`（n//3 + 余り）
- 各カテゴリに `DRAFT_CATEGORY_PROMPT` を使用
- `DRAFT_CATEGORY_PROMPTS` 辞書から各カテゴリの指示文を取得
- related_work からのコンテキスト注入: baseline カテゴリには `baseline_candidates`、open_problem カテゴリには `open_problems` を追加
- 各ノードの `rationale` には `[category]` プレフィックスを付与
- 全ノードの depth は 0

**再起草時**（`all_nodes` が存在）:

- 既存の仮説リスト（最大 10 件、pruned を除く）をコンテキストとして提示
- `DRAFT_PROMPT` テンプレートを使用
- LLM に「既存とは異なるアプローチ」を求める

**JSON パースとリトライ:**

- `_generate_proposals()` 内で最大 3 回リトライ（温度: 0.7, 0.8, 0.9）
- `_parse_json_response()`: `` ```json ``` `` ブロック -> 全体パース -> 正規表現で `[...]` / `{...}` を検索
- 全試行失敗時: フォールバックとしてデフォルト設定のノードを 1 つ生成

### debug(failed_node) -> SearchNode

失敗した実験コードを修正する非同期メソッド。

- `DEBUG_PROMPT` テンプレートに失敗ノードの情報（hypothesis, experiment_config, error_message, experiment_code）を埋め込む
- 温度は `specs._debug_temperature`（デフォルト 0.5）
- 返されるノード: `parent_id=failed_node.node_id`, `depth=failed_node.depth+1`, `branching_op="debug"`, `debug_depth=failed_node.debug_depth+1`
- JSON パース失敗時: 親ノードをクローンして `debug_depth` のみインクリメント

### improve(parent, all_nodes, n_children) -> list[SearchNode]

既存の成功実験をアトミックに改善する非同期メソッド。

- `IMPROVE_PROMPT` テンプレートに親ノードの統計情報（mu, SE, LCB, feasible）と兄弟コンテキスト、および失敗コンテキスト（`{failure_context}`）を埋め込む
- `_build_failure_context(parent)` で親ノードの `failure_context` をテキスト化し、プロンプトに注入する
- 温度: `specs._improve_temperature`（デフォルト 0.7）、リトライごとに +0.1
- 各子ノードの `experiment_config` を `validate_experiment_config()` でバリデーション
- 親との設定差分が 1 キーを超える場合は警告ログを出力
- バリデーション失敗時はそのノードをスキップ

**失敗コンテキスト構築** (`_build_failure_context`):

- `node.failure_context` が空でなければ、「Failed approaches to avoid:」ヘッダに続いて各失敗サマリを整形出力
- 各サマリ: `[{error_category}] {hypothesis}: {lesson}`
- 空の場合は空文字列を返す

**兄弟コンテキスト構築** (`_build_sibling_context`):

- 同じ `parent_id` で `status == "evaluated"` かつ `lcb` が None でない兄弟を抽出
- LCB 降順ソート、上位 `sibling_context_k`（デフォルト 5）件を取得
- グローバル最良ノード（全ノード中の最高 LCB）が兄弟に含まれない場合は追加（`[BEST]` ラベル付き）
- 各兄弟の表示: 仮説、親との設定差分、mu +/- SE、LCB、feasible、eval_runs

---

## compute_priority (sera.search.priority)

ノードの探索優先度を計算する関数。値が大きいほど先に展開される。

```python
def compute_priority(node, exec_spec) -> float
```

**ルール:**

| 条件 | 返り値 |
|------|--------|
| `node.feasible == False` | `-inf`（展開しない） |
| `node.lcb is None`（未評価） | `+inf`（優先的に探索） |
| 通常 | `lcb - lambda_cost * total_cost + beta_exploration * (1 / sqrt(eval_runs + 1))` |

**ハイパーパラメータ:**

- `lambda_cost`: `exec_spec.search.lambda_cost`（デフォルト 0.1）-- コストペナルティ
- `beta_exploration`: `exec_spec.search.beta_exploration`（デフォルト 0.05）-- 探索ボーナス

**探索ボーナス** (`compute_exploration_bonus`):

```
1.0 / sqrt(eval_runs + 1)
```

UCB1 スタイル。評価回数が少ないノードほど高いボーナスを得る。

---

## validate_experiment_config (sera.search.validation)

実験設定を ProblemSpec のホワイトリストに対してバリデーションする関数。

```python
def validate_experiment_config(config: dict, problem_spec) -> tuple[bool, list[str]]
```

**チェック内容:**

1. **未知キーの検出**: `config` のキーが `manipulated_variables` に存在しない場合はエラー
2. **型・範囲チェック**:
   - `float`: 数値型であること、`range[0]` から `range[1]` の範囲内であること
   - `int`: 整数型であること（`bool` は除外）、`range[0]` から `range[1]` の範囲内であること
   - `categorical`: `choices` リストに含まれること

**戻り値:** `(is_valid, error_messages)` のタプル。`is_valid` は `errors` が空のとき `True`。
