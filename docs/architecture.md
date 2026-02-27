# SERA システムアーキテクチャ

## 概要

SERA (Self-Evolving Research Agent) は、ツリーベースの解探索とパラメータ効率的なモデル適応 (LoRA + PPO) を組み合わせた、自律的な科学研究を行うPythonシステムである。本文書は、実装に基づいたアーキテクチャの詳細を記述する。

---

## 1. デュアルツリー設計

SERAは2つの同期されたツリー構造を管理する。

### 1.1 外部探索ツリー (sera.search)

外部探索ツリーは、仮説と実験設計の探索空間を管理する。

**SearchNode データクラス** (`sera.search.search_node.SearchNode`)

各ノードは以下のフィールドを持つ:

| カテゴリ | フィールド | 型 | 説明 |
|----------|-----------|-----|------|
| ID | `node_id` | `str` (UUID4) | ノードの一意識別子 |
| 構造 | `parent_id` | `str \| None` | 親ノードID |
| 構造 | `depth` | `int` | ツリー内の深さ |
| メタ | `created_at` | `str` | UTC ISO形式のタイムスタンプ |
| 外部状態 | `hypothesis` | `str` | テスト可能な仮説 |
| 外部状態 | `experiment_config` | `dict` | 操作変数の割り当て |
| 外部状態 | `experiment_code` | `str \| None` | 生成された実験コード |
| 外部状態 | `branching_op` | `str` | `"draft"` / `"debug"` / `"improve"` |
| 外部状態 | `rationale` | `str` | このアプローチの根拠 |
| 内部参照 | `adapter_node_id` | `str \| None` | LoRA系譜ツリーへの参照 |
| 評価統計 | `eval_runs` | `int` | 完了した評価回数 |
| 評価統計 | `metrics_raw` | `list[dict]` | 各シード実行の生メトリクス |
| 評価統計 | `mu` | `float \| None` | 主メトリクスの平均値 |
| 評価統計 | `se` | `float \| None` | 標準誤差 |
| 評価統計 | `lcb` | `float \| None` | 下側信頼限界 (mu - lcb_coef * se) |
| コスト | `total_cost` | `float` | 累計コスト |
| コスト | `wall_time_sec` | `float` | 累計実行時間(秒) |
| 探索制御 | `priority` | `float \| None` | ノード展開優先度 |
| 探索制御 | `status` | `str` | 下記参照 |
| 探索制御 | `children_ids` | `list[str]` | 子ノードIDリスト |
| 探索制御 | `feasible` | `bool` | 制約を全て満たすか |
| 探索制御 | `debug_depth` | `int` | デバッグ試行回数 |
| 探索制御 | `error_message` | `str \| None` | 失敗時のエラーメッセージ |
| ECHO | `failure_context` | `list[dict]` | 失敗兄弟から注入された知識（`FailureSummary.to_dict()` 形式） |

**ノードステータス遷移:**

```
pending → running → evaluated → expanded
                  ↘ failed (debug_depth < max → debug子ノード生成)
                  ↘ oom
                  ↘ timeout
         pruned (Prunerにより設定)
```

有効なステータス値: `"pending"`, `"running"`, `"evaluated"`, `"failed"`, `"pruned"`, `"expanded"`, `"timeout"`, `"oom"`

**SearchManager** (`sera.search.search_manager.SearchManager`)

- `open_list`: `list[tuple[float, str]]` -- `heapq` ミニヒープ (優先度を負値にして最大優先度を先頭にする)
- `all_nodes`: `dict[str, SearchNode]` -- 全ノードのマップ
- `closed_set`: `set[str]` -- 処理済みノードID
- `best_node`: 最高LCBを持つ実行可能ノード (LCB同値の場合、muが高い方を選択)
- `ppo_buffer`: `list[dict]` -- 評価済みノードからのPPO学習バッファ
- `failure_extractor`: ECHO軽量版の失敗知識抽出器 (None可)
- `turn_reward_evaluator`: Phase毎のターンレベル報酬評価器 (None可)

**ECHO統合**: `failure_extractor` が存在する場合、debugオペレータ実行後に `extract(failed_node)` で失敗知識を抽出し、`inject(summary, siblings)` で兄弟ノードの `failure_context` に注入する。

**ターン報酬統合**: `turn_reward_evaluator` が存在する場合、`_evaluate_node()` 内で `evaluate_all(node, parent, all_nodes)` を呼び出し、Phase毎のターン報酬を計算する。結果はPPOバッファに `PPORolloutV2` として追加される。

**ノード選択ロジック** (`select_next_node`):

1. `status == "failed"` かつ `debug_depth < max_debug_depth` かつ未処理 → `debug` オペレータ
2. 多様性チェック: 評価済みノード数 >= `draft_trigger_after` かつ固有メソッド数 < `min_diverse_methods` → `draft` オペレータ
3. `open_list` からポップ:
   - `status == "pending"` → `evaluate` オペレータ
   - `status == "evaluated"` → `improve` オペレータ
4. `open_list` が空 かつ ノード数 < `max_nodes` → `draft` オペレータ

**優先度計算** (`sera.search.priority.compute_priority`):

```
infeasible         → -inf (展開しない)
lcb is None        → +inf (未評価ノードを優先探索)
otherwise          → lcb - lambda_cost * total_cost + beta * exploration_bonus
exploration_bonus  = 1.0 / sqrt(eval_runs + 1)    # UCB1スタイル
```

**3つのオペレータ** (`sera.search.tree_ops.TreeOps`):

| オペレータ | 説明 | 入力 | 出力 |
|-----------|------|------|------|
| `draft` | 新しいアプローチを提案。ルート初期化時は baseline / open_problem / novel の3カテゴリに分割 (各 n//3)。再起動時は既存の仮説を提示して差別化を要求 | `n`, `all_nodes` | `list[SearchNode]` (depth=0) |
| `debug` | 失敗した実験コードを修正。エラーメッセージと実験コードをLLMに提示 | `failed_node` | `SearchNode` (debug_depth+1) |
| `improve` | 評価済みノードに対する原子的改善を提案。兄弟ノードのコンテキスト (sibling_context_k個) を含む。1-2変数の変更を推奨し、差分が1キーを超える場合は警告をログ出力 | `parent`, `all_nodes`, `n_children` | `list[SearchNode]` |

**実験設定バリデーション** (`sera.search.validation.validate_experiment_config`):

ホワイトリスト方式で操作変数のみを許可。型チェック (`float`, `int`, `categorical`) と範囲チェックを実施。

### 1.2 内部LoRA系譜ツリー (sera.lineage)

**LineageManager** (`sera.lineage.lineage_manager.LineageManager`)

ファイルシステムベースのディレクトリ構造でノードを管理:

```
lineage/
  nodes/
    <adapter_node_id>/
      meta.json                     # parent_id, depth, adapter_spec_hash, is_snapshot, tensor_names, tensor_shapes
      adapter_delta.safetensors     # 親からの重み差分 (delta)
      adapter_snapshot.safetensors  # 完全な重み (squashにより作成、任意)
```

**デルタ継承**: 子ノードは親からの重み差分 (delta) のみを保存する。これにより、ディスク使用量を大幅に削減できる。

**マテリアライゼーション** (`materialize`): ルートからノードまでのパスに沿ってデルタを合算し、完全な重みを再構築する。パス上にスナップショットがある場合は、最も深いスナップショットから開始する。

**スカッシュ** (`maybe_squash`): `squash_depth` (デフォルト: `max_depth // 2`) 以上の深さのノードに対して完全な重みのスナップショットを作成する。これにより、深いノードの再構築コストを制限する。`meta.json` の `is_snapshot` フラグを `True` に更新する。

**LRUキャッシュ** (`sera.lineage.cache.LRUCache`): `OrderedDict` ベース、デフォルト最大10エントリ。マテリアライズされたアダプタ重みをメモリにキャッシュする。

**vLLMエクスポート** (`export_for_vllm`): アダプタ重みを `adapter_model.safetensors` と `adapter_config.json` としてpeft形式で出力。`vllm.lora.request.LoRARequest` 互換。

**デルタ抽出** (`extract_delta_from_model`): `peft.get_peft_model_state_dict` APIを使用して、手動パラメータ反復ではなくpeftのAPIでアダプタデルタを抽出。`AutoModelForCausalLMWithValueHead` の場合は `.pretrained_model` を経由する。

### 1.3 ツリー同期ルール

`sync_adapter_assignment()` 関数（`search_manager.py`）が以下のルールで `adapter_node_id` を設定する:

- **ルートノード**: `adapter_node_id = "adapter_root"`
- **PPO更新あり**: 新しい `adapter_node_id` を割り当て（`PPOTrainer.update()` が返す `new_adapter_node_id`）
- **PPO更新なし**: 親の `adapter_node_id` を継承 (1:多の関係が通常)
- 探索ノード数 >= 系譜ノード数 (常に成立)

PPO更新後、`PPOTrainer` は `lineage_manager.save_delta()` でアダプタデルタをディスクに永続化し、`new_adapter_node_id` を結果辞書で返す。`SearchManager` は PPO バッファ内の全ノードに対して `sync_adapter_assignment()` を呼び出して `adapter_node_id` を更新する。

### 1.4 プルーニング (sera.lineage.pruner.Pruner)

プルーニングはステートレスで、実行時にExecutionSpecから設定を読み取る。以下の順序で実行:

1. **保護リスト構築**: 最良ノード + 祖先、top-k ノード + 祖先、実行中ノード
2. **LCB閾値プルーニング**: `reward_threshold` が非ゼロならその値、そうでなければ `best_lcb * 0.5` を閾値として使用
3. **パレートプルーニング**: (LCB, コスト) 空間で支配されるノードを除去。ノードBがノードAを支配する条件: BのLCB >= AのLCB かつ BのTotal Cost <= AのTotal Cost (少なくとも一方が厳密)
4. **予算プルーニング**: 総コストが予算上限 (`max_wall_time_hours * 3600`) を超過した場合、LCBの低いノードから除去

---

## 2. モジュール依存関係

実際のパッケージ構造 (全て `src/sera/` 配下):

```
sera.cli                          → Typer エントリポイント (app)
sera.commands/                    → CLIコマンドハンドラ
  init_cmd                        → sera init
  phase0_cmd                      → sera phase0-related-work
  phase1_cmd                      → sera freeze-specs
  research_cmd                    → sera research (Phase 2-6ループのオーケストレーション)
  paper_cmd                       → sera generate-paper / sera evaluate-paper
  export_cmd                      → sera export-best
  status_cmd                      → sera status / sera show-node
  replay_cmd                      → sera replay
  validate_cmd                    → sera validate-specs
  setup_cmd                       → sera setup (対話型セットアップウィザード)
  visualize_cmd                   → sera visualize (探索木のインタラクティブHTML可視化)
sera.visualization/
  tree_visualizer                 → TreeVisualizer (D3.jsベースの探索木可視化)
  node_formatter                  → format_node() (ノードデータの整形)
  stats_calculator                → compute_stats() (統計情報の計算)
  html_renderer                   → render_html() (HTML出力生成)
sera.agent/
  agent_llm                       → 統合LLMインターフェース (local/OpenAI/Anthropic)
                                    PROMPT_FORMATTERS レジストリ (モデルファミリ別プロンプト整形)
                                    ToolCall / GenerationOutput データクラス
                                    generate_with_tools() (ネイティブtool calling対応)
                                    get_turn_log_probs() (MT-GRPO用Phase別ログ確率)
  agent_functions                 → AgentFunctionRegistry (19個の登録済み関数)
                                    AgentFunction: frozen dataclass (name, description, parameters,
                                      return_schema, output_mode, phase, default_temperature,
                                      max_retries, handler, allowed_tools, loop_config)
                                    call_function() はプライマリエントリポイント
  agent_loop                      → AgentLoop (ReActループ、max_steps=10, budget=20, timeout=300s)
  tool_executor                   → ToolExecutor (18ツール: SEARCH×6, EXECUTION×3, FILE×5, STATE×4)
  tool_policy                     → ToolPolicy (Phase別ツール許可、書き込みホワイトリスト、レート制限)
  mcp_client                      → MCPToolProvider (httpxベースのMCPプロトコル実装)
  prompt_templates                → 21個のプロンプトテンプレート (TEMPLATE_REGISTRY)
  vllm_engine                     → vLLMオフラインモード推論エンジン (LoRA hot-swap + sleep/wake)
sera.prompts/                     → YAMLベースのプロンプトテンプレート (Phase別)
  prompt_loader                   → テンプレートのロード・管理
sera.specs/
  __init__ (AllSpecs)             → 10個のPydantic specモデルを集約するデータクラス
  input1                          → Input1Model (ユーザー入力)
  problem_spec                    → ProblemSpecModel (最適化問題定義)
  model_spec                      → ModelSpecModel (ベースモデル、アダプタ、VLM設定)
  resource_spec                   → ResourceSpecModel (計算資源設定)
  plan_spec                       → PlanSpecModel (探索戦略、報酬、ロギング、agent_commands)
  execution_spec                  → ExecutionSpecModel (7つのサブ設定を集約)
  paper_spec                      → PaperSpecModel (論文フォーマット設定)
  paper_score_spec                → PaperScoreSpecModel (評価基準)
  related_work_spec               → RelatedWorkSpecModel
  teacher_paper_set               → TeacherPaperSetModel
sera.phase0/
  related_work_engine             → Phase 0 オーケストレーション (6ステップ)
  api_clients/
    base                          → BaseScholarClient ABC + PaperResult データクラス
    semantic_scholar               → Semantic Scholar API クライアント
    crossref                      → CrossRef API クライアント
    arxiv                         → arXiv API クライアント
    web_search                    → SerpAPI ウェブ検索クライアント
  ranking                         → citation_norm, compute_ranking_score, rank_papers
  clustering                      → cluster_papers (LLMベースのクラスタリング、キーワード抽出)
sera.phase1/
  spec_builder                    → LLM駆動のspec生成 + ModelSpec/ResourceSpec/ExecutionSpec構築
  spec_freezer                    → SHA-256によるExecutionSpecロック + 検証 + agent_commandsバリデーション
sera.search/
  search_manager                  → メイン探索ループ (SearchManager.run())
  search_node                     → SearchNode データクラス
  tree_ops                        → Draft/Debug/Improve オペレータ (TreeOps)
  priority                        → compute_priority() + compute_exploration_bonus()
  validation                      → validate_experiment_config() (ホワイトリスト検証)
  failure_extractor               → FailureKnowledgeExtractor + FailureSummary (ECHO軽量版)
sera.execution/
  executor                        → Executor ABC + RunResult データクラス
  local_executor                  → LocalExecutor (subprocess ベース)
  slurm_executor                  → SlurmExecutor (submitit ベース)
  docker_executor                 → DockerExecutor (Docker SDK、GPU対応、OOM検出)
  experiment_generator            → ExperimentGenerator (LLMコード生成)
  ablation                        → AblationRunner / AblationResult (auto-ablation実験実行)
sera.evaluation/
  evaluator                       → Evaluator ABC (evaluate_initial / evaluate_full)
  statistical_evaluator           → StatisticalEvaluator (二段階逐次評価)
  bootstrap_evaluator             → BootstrapEvaluator (ブートストラップ信頼区間、evaluation.bootstrap=True時)
  feasibility                     → check_feasibility (epsilon制約チェック)
sera.learning/
  ppo_trainer                     → PPOTrainer (LoRAパラメータのみ更新、メソッド別Advantage計算)
  reward                          → compute_reward() (レジストリパターンで4手法ディスパッチ)
  rollout                         → PPORollout / PPORolloutV2 / PPORolloutV3 データクラス
  tool_usage_learning             → ToolCallRecord / ToolUsageStats / compute_reward_tool_aware
  turn_reward                     → TurnRewardEvaluator (Phase毎のターンレベル報酬評価)
  hierarchical_ppo                → HierarchicalAdvantageEstimator (HiPER 3層Advantage分解)
sera.lineage/
  lineage_manager                 → LineageManager (デルタ保存、マテリアライゼーション、スカッシュ)
  pruner                          → Pruner (LCB閾値 + パレート + 予算プルーニング)
  cache                           → LRUCache (OrderedDict ベース、最大10エントリ)
sera.paper/
  paper_composer                  → PaperComposer (6ステップの論文生成パイプライン、auto-ablation対応)
  paper_evaluator                 → PaperEvaluator (アンサンブルレビュー + メタレビュー)
  figure_generator                → FigureGenerator (matplotlib/graphviz)
  citation_searcher               → CitationSearcher (Semantic Scholar引用ループ)
  vlm_reviewer                    → VLMReviewer (OpenAI/Anthropic VLMによる図レビュー)
  evidence_store                  → EvidenceStore (インメモリのエビデンス集約器)
  latex_composer                  → LaTeXComposer (Markdown → LaTeX 変換)
sera.utils/
  seed                            → シード管理
  hashing                         → compute_spec_hash, compute_adapter_spec_hash
  logging                         → JsonlLogger (追記専用JSONL) + setup_structlog
  checkpoint                      → save_checkpoint / load_latest_checkpoint
```

### AllSpecs 構造

`AllSpecs` は plain データクラスで、10個のPydantic v2モデルを集約する:

```python
@dataclasses.dataclass
class AllSpecs:
    input1: Input1Model
    related_work: RelatedWorkSpecModel
    paper: PaperSpecModel
    paper_score: PaperScoreSpecModel
    teacher_paper_set: TeacherPaperSetModel
    problem: ProblemSpecModel
    model: ModelSpecModel
    resource: ResourceSpecModel
    plan: PlanSpecModel
    execution: ExecutionSpecModel
```

正規ファイルマッピング (`_SPEC_FILES`):

| フィールド名 | YAMLファイル名 |
|-------------|---------------|
| `input1` | `input1.yaml` |
| `related_work` | `related_work_spec.yaml` |
| `paper` | `paper_spec.yaml` |
| `paper_score` | `paper_score_spec.yaml` |
| `teacher_paper_set` | `teacher_paper_set.yaml` |
| `problem` | `problem_spec.yaml` |
| `model` | `model_spec.yaml` |
| `resource` | `resource_spec.yaml` |
| `plan` | `plan_spec.yaml` |
| `execution` | `execution_spec.yaml` |

I/Oヘルパー: `load_from_dir(specs_dir)` / `save_to_dir(specs_dir)`

### ExecutionSpecModel 構造

`ExecutionSpecModel` は以下の7つのサブ設定を集約する:

| サブ設定 | クラス | 主要パラメータ |
|---------|--------|--------------|
| `search` | `SearchConfig` | `max_nodes`, `max_depth`, `branch_factor`, `lambda_cost`, `beta_exploration`, `repeats`, `lcb_coef`, `squash_depth` |
| `evaluation` | `EvaluationConfig` | `timeout_per_run_sec`, `repeats`, `lcb_coef`, `sequential_eval_initial`, `sequential_eval_topk` |
| `learning` | `LearningConfig` | `enabled`, `lr`, `clip_range`, `batch_size`, `ppo_trigger_interval`, `gamma`, `gae_lambda`, `kl_coef`, `entropy_coef` |
| `lora_runtime` | `LoraRuntimeConfig` | `delta_inheritance`, `squash_depth`, `cache_max_entries` |
| `pruning` | `PruningConfig` | `reward_threshold`, `keep_topk`, `pareto`, `prune_interval`, `budget_limit` |
| `termination` | `TerminationConfig` | `max_wall_time_hours`, `max_steps`, `plateau_patience` |
| `paper_exec` | `PaperExecConfig` | `paper_revision_limit`, `n_writeup_reflections`, `citation_search_rounds`, `vlm_enabled` |

---

## 3. データフロー

```
Input-1 YAML (ユーザー入力)
  │
  ├─ Phase 0: RelatedWorkEngine.run()
  │    クエリ生成 (LLM or ヒューリスティック)
  │    → API検索 (フォールバック順) → 重複排除 → 引用グラフ展開
  │    → ランキング (citation_norm * weight + relevance * (1 - weight))
  │    → クラスタリング (LLMベース)
  │    → specs/ (related_work_spec.yaml, paper_spec.yaml, paper_score_spec.yaml, teacher_paper_set.yaml)
  │
  ├─ Phase 1: SpecBuilder + SpecFreezer
  │    LLM駆動のspec生成 → 10個のYAMLファイル保存
  │    → SHA-256ハッシュ計算 → execution_spec.yaml.lock 書き込み
  │    → ModelSpec: base_model.revisionの自動解決, adapter_spec_hashの計算
  │
  └─ research_cmd.py: Phase 2-6ループ
       │
       │  ExecutionSpec整合性検証 (verify → hash不一致でexit code 2)
       │  AllSpecs.load_from_dir()
       │  adapter_spec_hash整合性検証 (不一致でexit code 3)
       │  コンポーネント初期化 (AgentLLM, Executor, ExperimentGenerator,
       │                        StatisticalEvaluator, TreeOps, SearchManager,
       │                        PPOTrainer, LineageManager, Pruner)
       │
       ├─ SearchManager.run() ループ:
       │    │
       │    ├─ Phase 2: TreeOps.draft/debug/improve → 新しいSearchNode生成
       │    │    ルート初期化: draft(n) → baseline/open_problem/novel 各 n//3 個
       │    │    自動オペレータ選択: select_next_node()
       │    │
       │    ├─ Phase 3: ExperimentGenerator.generate() → runs/{node_id}/experiment.py (or .R, .jl)
       │    │    LocalExecutor/SlurmExecutor.run() → runs/{node_id}/metrics.json
       │    │    OOM検出: exit_code -7 (stderr パターンマッチ)
       │    │    タイムアウト: exit_code -9
       │    │
       │    ├─ Phase 4: StatisticalEvaluator
       │    │    evaluate_initial(): sequential_eval_initial シード数で迅速推定
       │    │    _is_topk() チェック → top-k ノードのみ evaluate_full() で残りシード実行
       │    │    update_stats(): mu, SE, LCB = mu - lcb_coef * SE
       │    │    check_feasibility(): epsilon制約チェック (bool / ge / le)
       │    │
       │    ├─ Phase 5: PPOTrainer.update()
       │    │    トリガー条件: ppo_trigger_interval ごと or プラトー検出
       │    │    報酬手法ディスパッチ: plan_spec.reward.method で選択
       │    │      outcome_rm → 従来のGAE計算
       │    │      mt_grpo   → ターン報酬反映済みのGAE計算
       │    │      hiper     → HierarchicalAdvantageEstimator で3層Advantage分解
       │    │    vLLMエンジンのsleep(level=2) → PPO学習 → wake()
       │    │    PPOクリップサロゲート損失 + 価値関数損失 + エントロピーボーナス
       │    │    LoRAパラメータのみ更新 (requires_grad かつ "lora" を含む名前)
       │    │    accelerate.Acceleratorによるデバイス管理 + 勾配クリッピング
       │    │    適応的KL係数制御
       │    │    → lineage/nodes/{id}/adapter_delta.safetensors
       │    │
       │    └─ Phase 6: LineageManager.maybe_squash() + Pruner.prune()
       │         squash_depth以上の深さのノードにスナップショット作成
       │         prune_interval ステップごとにプルーニング実行
       │
       │  → checkpoints/search_state_step_N.json (10ステップごと)
       │  → logs/*.jsonl (search, eval, ppo, agent_llm)
       │
       ├─ Phase 7: PaperComposer.compose() (6ステップパイプライン + オプションのauto-ablation)
       │    1. ログ要約 → experiment_summaries.json
       │    1b. Auto-ablation (ablation_runner提供時): AblationRunnerで最良ノードのアブレーション実験実行
       │    2. プロット集約 (FigureGenerator + LLMリフレクション) → figures/*.png
       │    3. 引用検索ループ (CitationSearcher, 最大20ラウンド)
       │    4. VLM図面記述 (VLMReviewer、有効時のみ)
       │    5. 論文本文生成 + ライティングリフレクションループ (n_writeup_reflections回)
       │    6. 最終統合 (図番号付け、引用キー整合性、参考文献セクション追加)
       │    → paper/paper.md + paper/paper.bib + paper/figure_descriptions.json + figures/*.png
       │
       ├─ Phase 8: PaperEvaluator.evaluate()
       │    アンサンブルLLMレビュー (num_reviews_ensemble人のレビュアー)
       │    各レビュアー: バイアスモード交互 (critical/generous)
       │    各レビュー: num_reviewer_reflections回のリフレクション
       │    → スコア集約 (平均) + メタレビュー生成 + 多数決による判定
       │    判定: accept / revise / reject (overall_score >= passing_score で passed)
       │    → 改善指示に基づく改訂ループ (Phase 7へ)
       │
       └─ export_cmd → outputs/best/
            best_node.json, adapter.safetensors, metrics_summary.json, report.json

  研究ループ終了コード:
    exit(2)  -- ExecutionSpec改竄検知
    exit(3)  -- adapter_spec_hashの不一致
    exit(11) -- 研究完了したが有効ノードなし
    exit(12) -- 予算超過
    exit(20) -- SIGINTによるグレースフル停止
```

---

## 4. 主要インターフェース

### 4.1 RunResult (sera.execution.executor)

```python
@dataclass
class RunResult:
    node_id: str
    success: bool
    exit_code: int        # 0=成功, -9=タイムアウト, -7=OOM, 137=SIGKILL, 127=FileNotFound
    stdout_path: Path
    stderr_path: Path
    metrics_path: Path | None
    artifacts_dir: Path
    wall_time_sec: float
    seed: int
```

### 4.2 metrics.json 契約 (言語非依存、実験スクリプトが出力)

```json
{
  "primary": {"name": "score", "value": 0.73, "higher_is_better": true},
  "constraints": [{"name": "...", "value": "...", "type": "...", "satisfied": true}],
  "secondary": [{"name": "cost", "value": 12.3, "lower_is_better": true}],
  "raw": {},
  "seed": 42,
  "wall_time_sec": 125.3,
  "score": 0.73
}
```

`StatisticalEvaluator` は `metrics_raw` 内の各dictから `metric_name` キー (例: `"score"`) の値を抽出して mu/SE/LCB を計算する。

### 4.3 PPORollout / PPORolloutV2 / PPORolloutV3 (sera.learning.rollout)

```python
@dataclass
class PPORollout:
    node_id: str
    prompt: str             # LLM入力 (仮説生成プロンプト)
    response: str           # LLM出力 (仮説 + 実験設計JSON)
    log_prob: float         # 出力トークン列のログ確率
    reward: float           # 計算された報酬
    value: float            # 価値関数推定値
    advantage: float = 0.0  # GAE計算された利得 (後で充填)
    returns: float = 0.0    # 割引リターン (後で充填)

@dataclass
class PPORolloutV2(PPORollout):
    turn_rewards: dict[str, float] = field(default_factory=dict)
    # MT-GRPO/HiPER用: Phase毎のターンレベル報酬
    # 例: {"phase0": 0.9, "phase3": 1.0, "phase4": 0.6}

@dataclass
class PPORolloutV3(PPORolloutV2):
    tool_trajectory: list = field(default_factory=list)
    # tool_aware報酬用: ToolCallRecordのリスト
    # ノードのライフタイム中のツール呼び出しを記録
```

`PPORolloutV2` は `turn_reward_evaluator` が有効な場合に `SearchManager` で使用される。`PPORolloutV3` は `tool_aware` 報酬手法と組み合わせて、ツール使用効率に基づく報酬調整に使用される。

### 4.4 報酬計算 (sera.learning.reward.compute_reward)

`compute_reward()` は **レジストリパターン** で `plan_spec.reward.method` に応じて 4 つの手法にディスパッチする:

| 手法 | 関数 | 説明 |
|------|------|------|
| `outcome_rm` | `compute_reward_outcome_rm` | 従来方式（デフォルト） |
| `mt_grpo` | `compute_reward_mt_grpo` | ターンレベル報酬の重み付き和 |
| `tool_aware` | `compute_reward_tool_aware_dispatch` | `mt_grpo` をベースに、ツール使用効率のボーナス・失敗ペナルティを加算 |
| `hiper` | `compute_reward_hiper` | HiPER（報酬値は `mt_grpo` に委譲、Advantage分解はPPOTrainer側） |

**Outcome RM（デフォルト）**:
```
R = primary_value
    - constraint_penalty * num_violated_constraints
    - lambda_cost * normalized_cost
    - kl_coef * kl_divergence
```

**MT-GRPO**:
```
R = Σ(weight_t * turn_reward_t)
    - constraint_penalty * num_violated_constraints
    - lambda_cost * normalized_cost
    - kl_coef * kl_divergence
```

共通:
```
  primary_value    = metrics_raw中のprimaryメトリクス値 (minimize方向なら符号反転)
  normalized_cost  = min(total_cost / budget_limit, 1.0)
  budget_limit     = max_wall_time_hours * 3600 (デフォルト: 14400秒)

失敗/タイムアウト/OOM → -100.0 (全手法共通)
```

### 4.5 Executor ABC (sera.execution.executor)

```python
class Executor(ABC):
    @abstractmethod
    def run(self, node_id: str, script_path: Path, seed: int,
            timeout_sec: int | None = None) -> RunResult: ...
```

実装状況:

| クラス | バックエンド | 状態 |
|-------|------------|------|
| `LocalExecutor` | `subprocess.Popen` | 実装済み |
| `SlurmExecutor` | `submitit.AutoExecutor` | 実装済み |
| `DockerExecutor` | `docker` Python SDK | 実装済み |

`LocalExecutor`、`SlurmExecutor`、`DockerExecutor` は多言語対応: `interpreter_command` と `seed_arg_format` が設定可能 (Python, R, Julia等)。

`DockerExecutor` は `docker.from_env()` でクライアントを生成し、コンテナ内で実験を実行する。GPU パススルー（nvidia runtime / `DeviceRequest`）、タイムアウト処理（`container.wait(timeout=...)`）、OOM 検出（Docker `OOMKilled` フラグ + exit code 137 + stderr パターン）に対応。`pip install docker` でインストールが必要（未インストール時はインポートエラー）。

`SlurmExecutor` は `compute_config: ComputeConfig | None` を受け取り、`_build_compute_params()` で submitit パラメータに自動マッピングする（`slurm_gpus_per_node`, `slurm_mem`, `slurm_cpus_per_task`, `constraint`）。`sbatch_extra` のユーザー指定値が常に優先される。`compute_config=None` の場合は従来動作と完全互換。

**非同期バッチ実行**: `SlurmExecutor` は非同期バッチパイプラインもサポートする:

- `submit_async(node_id, script_path, seed, timeout_sec)` -- ジョブを投入し、即座にハンドル辞書を返す（ブロックしない）
- `collect_result(handle)` -- ハンドルから `RunResult` を収集する
- `run_batch(tasks)` -- 3 フェーズパイプライン: Phase A（全ジョブ一括投入） → Phase B（全ジョブ非同期ポーリング） → Phase C（結果収集）
- `_async_poll_job()` / `_async_poll_job_squeue()` -- `asyncio.sleep` ベースの非同期ジョブポーリング

既存の同期 `run()` メソッドはそのまま維持されている。

### 4.6 Evaluator ABC (sera.evaluation.evaluator)

```python
class Evaluator(ABC):
    @abstractmethod
    async def evaluate_initial(self, node: Any) -> None: ...

    @abstractmethod
    async def evaluate_full(self, node: Any) -> None: ...
```

`StatisticalEvaluator` は二段階逐次評価を実装:
1. `evaluate_initial`: `sequential_eval_initial` シード数 (デフォルト: 1) で迅速推定
2. `evaluate_full`: 残りシード数 (`repeats - eval_runs`) で完全評価。top-kノードのみ実行

シード導出: `SHA256(base_seed:node_id:repeat_idx) % 2^31` (決定的)

### 4.7 AgentLLM (sera.agent.agent_llm)

3つのプロバイダに対応:

| プロバイダ | バックエンド | 特記事項 |
|-----------|------------|---------|
| `local` | transformers + peft (or vLLM) | LoRA動的切り替え、`trl.trainer.utils.selective_log_softmax` でログ確率計算、`AutoModelForCausalLMWithValueHead` で価値推定 |
| `openai` | `openai.AsyncOpenAI` | 環境変数からAPIキー取得、ネイティブtool calling対応 |
| `anthropic` | `anthropic.AsyncAnthropic` | 環境変数からAPIキー取得、ネイティブtool calling対応 |

推論エンジン選択 (`model_spec.inference.engine`):
- `"transformers"`: HuggingFace transformersによる直接推論
- `"vllm"`: `VLLMInferenceEngine` によるオフラインモード推論。LoRA hot-swap対応、`sleep(level=2)` / `wake_up()` でPPO学習時のGPUメモリ解放

**モデルファミリ別プロンプト整形** (§25.3.2):

`PROMPT_FORMATTERS` レジストリがモデルファミリに応じたプロンプト整形を提供する。`_format_prompt()` メソッドがローカルプロバイダで `tokenizer.apply_chat_template` が利用できない場合に使用される。

| フォーマッタ | 対応フォーマット | 対象モデル |
|-------------|----------------|-----------|
| `_ChatMLFormatter` | `chatml` | Qwen2系 |
| `_Llama3Formatter` | `llama3` | Llama 3系、CodeLlama |
| `_DeepSeekFormatter` | `deepseek` | DeepSeek系 |
| `_PromptFormatter` | `default` | パススルー（変換なし） |

**構造化出力型**:

- `ToolCall`: `tool_name`, `arguments`, `call_id`, `reasoning` を保持するデータクラス
- `GenerationOutput`: `text`, `tool_calls`, `purpose`, `text_log_prob`, `tool_call_log_probs` を保持するデータクラス

**ツール呼び出し** (`generate_with_tools`):

OpenAI/Anthropic プロバイダではネイティブ tool calling API を使用。ローカルプロバイダではプロンプトベースでツール定義を注入し、JSONレスポンスをパースする。

**Phase別ログ確率** (`get_turn_log_probs`):

MT-GRPO用にPhase別のレスポンスに対するログ確率を計算する。コンテキストを逐次拡張しながら各Phaseの `selective_log_softmax` を計算する。

全てのLLM呼び出しは `agent_llm_log.jsonl` にログ出力 (call_id, purpose, prompt_hash, latency_ms)。

---

## 5. ストレージ (フラットファイル)

現在の実装は全ての永続化にフラットファイルを使用する。SQLiteは使用していない。

```
sera_workspace/
  input1.yaml                     # ユーザー入力ファイル
  specs/                          # 10個のYAMLファイル + ロックファイル
    input1.yaml
    related_work_spec.yaml
    paper_spec.yaml
    paper_score_spec.yaml
    teacher_paper_set.yaml
    problem_spec.yaml
    model_spec.yaml
    resource_spec.yaml
    plan_spec.yaml
    execution_spec.yaml
    execution_spec.yaml.lock      # SHA-256ハッシュ (改ざん検出用)
  logs/
    agent_llm_log.jsonl           # 全LLM呼び出し (call_id, purpose, prompt_hash, latency_ms)
    search_log.jsonl              # ノード処理イベント (step, operator, status, mu, se, lcb, priority)
    ppo_log.jsonl                 # PPO更新イベント (mean_reward, kl_divergence, policy_loss, etc.)
    eval_log.jsonl                # 評価イベント (node_id, mu, se, lcb, feasible)
  checkpoints/
    search_state_step_N.json      # SearchManagerの完全な状態 (step, all_nodes, closed_set, best_node_id, open_list, ppo_buffer)
  lineage/
    nodes/<adapter_node_id>/
      meta.json                   # parent_id, depth, adapter_spec_hash, is_snapshot, tensor_names, tensor_shapes
      adapter_delta.safetensors   # 親からの重み差分
      adapter_snapshot.safetensors # 完全な重み (squashにより作成、任意)
  runs/<node_id>/
    experiment.py (or .R, .jl)    # 生成された実験スクリプト
    stdout.log
    stderr.log
    metrics.json                  # 実験結果メトリクス
    slurm_logs/                   # SLURM使用時のsubmititログ
  outputs/best/
    best_node.json                # 最良ノードの完全データ
    adapter.safetensors           # 最良アダプタの重み (存在する場合)
    metrics_summary.json          # メトリクスサマリー
    report.json                   # 研究レポート
  paper/
    paper.md                      # 生成された論文 (Markdown)
    paper.bib                     # 参考文献 (BibTeX形式)
    figure_descriptions.json      # VLM図面記述 (JSON)
    figures/                      # 生成された図 (*.png)
    experiment_summaries.json     # 実験サマリーデータ
```

---

## 6. 並行処理モデル

| フェーズ | 方式 | 詳細 |
|---------|------|------|
| Phase 0 | `asyncio` + 非同期HTTPクライアント | `httpx.AsyncClient` によるAPI呼び出し。プロバイダごとにフォールバック順序で逐次試行 |
| Phase 2 | 逐次 | ツリーノード展開は一度に1ノード (一貫したツリー状態を維持) |
| Phase 3 | 逐次 (ノード内) / 非同期バッチ | `LocalExecutor` / `SlurmExecutor` によるシード単位の実行。`SlurmExecutor` はジョブ投入後にポーリング (`sacct` / `squeue`)。非同期バッチモード (`run_batch`) では全ジョブを一括投入し、`asyncio` ベースで並列ポーリング |
| Phase 5 | 単一プロセスPPO | `accelerate.Accelerator()` による任意のマルチGPU対応 |
| Phase 7 | 逐次 (ステップ単位) | 6ステップの論文生成パイプライン |

**vLLM制約**: vLLMエンジンとPyTorch学習は同時にGPUメモリを共有できないため、PPO更新前に `sleep(level=2)` でGPUメモリを解放し、更新後に `wake_up()` で復帰する。この処理は `PPOTrainer._ppo_update_impl` の try/finally で保証されている。

---

## 7. エラー回復

### 7.1 SIGINT処理

`SearchManager._setup_signal_handler()` が `signal.SIGINT` をフックする:

1. 現在の `SearchManager` 状態をチェックポイントとして保存
2. `sys.exit(20)` で終了

### 7.2 チェックポイントからの再開

```bash
sera research --resume
```

`load_latest_checkpoint(checkpoint_dir)` が `search_state_step_*.json` をグロブしてソート順で最新を読み込み、`SearchManager.load_state()` で状態を復元する。

### 7.3 チェックポイント間隔

10ステップごと (`SearchManager.run` 内のハードコード値)。

### 7.4 ExecutionSpec改ざん検出

`research_cmd.py` の研究ループ開始前に `SpecFreezer.verify(specs_dir)` を呼び出す:

1. `execution_spec.yaml` のYAMLデータを読み込む
2. `compute_spec_hash()` で SHA-256 ハッシュを計算
3. `execution_spec.yaml.lock` に格納されたハッシュと比較
4. 不一致 → `sys.exit(2)` でアボート

### 7.5 adapter_spec_hash 整合性検証

`research_cmd.py` は ExecutionSpec 検証の後、`adapter_spec_hash` の整合性もチェックする。LoRA アダプタのスペック（rank, alpha, target_modules 等）のハッシュが lineage ツリー内のメタデータと不一致の場合は `sys.exit(3)` でアボートする。

### 7.6 学習の無効化

`ExecutionSpec.learning.enabled` が `False` の場合、PPOTrainer, LineageManager, Pruner は初期化されない。コンポーネント初期化中の例外もキャッチされ、警告を出力してPPO/Lineageを無効化する。

---

## 8. 変数の可変性 (三層分離)

コードベース全体で強制される設計制約:

| 層 | 所在 | Phase 1後に変更可能か | 例 |
|----|------|---------------------|-----|
| **凍結 (Frozen)** | `ExecutionSpecModel` | 不可 (SHA-256ロック) | `search.lr`, `search.clip_range`, `search.repeats`, `search.lcb_coef`, `search.max_nodes`, `learning.lr`, `learning.clip_range`, `adapter_spec.rank`, `adapter_spec.alpha` |
| **操作 (Manipulated)** | `ProblemSpec.manipulated_variables` | 可 (ノードごとに分岐) | 実験の `learning_rate`, `batch_size`, `method` (SearchNode.experiment_config内) |
| **導出 (Derived)** | 実行時計算 | 自動計算のみ | `priority`, `mu`, `se`, `lcb`, `feasible`, `reward` |

凍結層と操作層の変数は類似した名前を持つことがある (例: `ExecutionSpec.learning.lr` vs `ProblemSpec.manipulated_variables[].learning_rate`)。これらは異なる層に属する。

---

## 9. 拡張ポイント

| 拡張対象 | メカニズム | 現在の実装 |
|---------|----------|-----------|
| 実験実行バックエンド | `Executor` ABC を継承 | `LocalExecutor` (subprocess), `SlurmExecutor` (submitit), `DockerExecutor` (Docker SDK) が実装済み |
| 評価ロジック | `Evaluator` ABC を継承 | `StatisticalEvaluator`, `BootstrapEvaluator` が実装済み |
| 論文検索プロバイダ | `BaseScholarClient` ABC を継承 | `SemanticScholarClient`, `CrossRefClient`, `ArxivClient`, `WebSearchClient` が実装済み |
| 分岐オペレータ | `TreeOps` に新メソッド追加 | `draft`, `debug`, `improve` が実装済み |
| LLMプロバイダ | `AgentLLM` のプロバイダ分岐 | `local` (transformers/vLLM), `openai`, `anthropic` が対応済み |
| 実験言語 | `ProblemSpec.language` の設定 | `LanguageConfig` で `interpreter_command`, `file_extension`, `seed_arg_format`, `code_block_tag` を設定可能 |
| プロンプトテンプレート | `TEMPLATE_REGISTRY` に新テンプレート追加 | 21個のテンプレートが登録済み |
| 報酬手法 | `register_reward_method` デコレータで新手法を登録 | `outcome_rm`, `mt_grpo`, `tool_aware`, `hiper` が登録済み |
| Phase報酬評価器 | `TurnRewardEvaluator._PHASE_EVALUATORS` に新評価器を追加 | 5個の評価器が登録済み |
