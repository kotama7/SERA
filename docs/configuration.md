# SERA 設定リファレンス

## 概要

SERA の設定は以下の優先順位で適用されます（上が最優先）:

1. **CLI 引数** -- `sera freeze-specs` コマンドのオプションで指定
2. **Spec YAML ファイル** -- `specs/` ディレクトリ内の各 YAML ファイル
3. **Pydantic モデルのデフォルト値** -- 各 spec モデルに定義されたフォールバック値

## ExecutionSpecModel（実行仕様）

`sera.specs.execution_spec.ExecutionSpecModel` は SERA の探索・評価・学習・枝刈り・終了条件・論文生成の設定を一括管理するモデルです。**Phase 1 で凍結（freeze）された後は変更不可**です。

7 つのネストされたサブ設定で構成されます。

### 1. SearchConfig（探索設定）

木探索のハイパーパラメータを管理します。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `strategy` | `str` | `"best_first"` | 探索アルゴリズム名 |
| `priority_rule` | `str` | `"lcb_cost_explore"` | 優先度計算ルール |
| `max_nodes` | `int` | `100` | 探索木の最大ノード数 |
| `max_depth` | `int` | `10` | 木の最大深さ |
| `branch_factor` | `int` | `3` | ノードあたりの最大子ノード数 |
| `initial_root_children` | `int` | `5` | ルートノードの初期ドラフト数 |
| `max_debug_depth` | `int` | `3` | ノードあたりの最大デバッグ試行回数 |
| `min_diverse_methods` | `int` | `3` | ドラフト再トリガーの多様性閾値 |
| `draft_trigger_after` | `int` | `10` | 多様性チェック開始の最小評価済みノード数 |
| `lambda_cost` | `float` | `0.1` | 優先度におけるコストペナルティ係数 |
| `beta_exploration` | `float` | `0.05` | 探索ボーナス係数 |
| `repeats` | `int` | `3` | 完全評価の実験繰り返し回数 |
| `lcb_coef` | `float` | `1.96` | LCB 係数（1.96 は 95% 信頼区間に相当） |
| `sequential_eval` | `bool` | `True` | 逐次評価戦略を使用するか |
| `sequential_eval_initial` | `int` | `1` | 簡易推定用の初期シード数 |
| `sequential_eval_topk` | `int` | `5` | 完全評価を行う上位ノード数 |
| `sibling_context_k` | `int` | `5` | improve 操作時に参照する兄弟ノード数 |
| `squash_depth` | `int \| None` | `None` | lineage スナップショットのスカッシュ深さ（None の場合は `max_depth // 2`） |

### 2. EvaluationConfig（評価設定）

実験結果の統計評価に関する設定です。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `timeout_per_run_sec` | `int` | `600` | 1 回の実行のタイムアウト（秒） |
| `metric_aggregation` | `str` | `"mean"` | 繰り返しの集約方法 |
| `record_stderr` | `bool` | `True` | 実験の stderr を記録するか |
| `repeats` | `int` | `3` | 完全評価の実験繰り返し回数 |
| `lcb_coef` | `float` | `1.96` | LCB 係数 |
| `sequential_eval` | `bool` | `True` | 逐次評価戦略 |
| `sequential_eval_initial` | `int` | `1` | 初期シード数 |
| `sequential_eval_topk` | `int` | `5` | 完全評価の上位ノード数 |
| `bootstrap` | `bool` | `False` | ブートストラップ信頼区間を使用するか |
| `bootstrap_samples` | `int` | `1000` | ブートストラップサンプル数 |

### 3. LearningConfig（学習設定）

LoRA アダプタの PPO オンライン学習に関する設定です。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `enabled` | `bool` | `True` | オンライン学習を有効にするか |
| `algorithm` | `str` | `"ppo"` | 学習アルゴリズム名 |
| `update_target` | `str` | `"lora_only"` | 更新対象のパラメータ |
| `optimizer` | `str` | `"adamw"` | オプティマイザ名 |
| `lr` | `float` | `1e-4` | 学習率 |
| `lr_scheduler` | `str` | `"cosine"` | 学習率スケジューラ |
| `warmup_steps` | `int` | `50` | ウォームアップステップ数 |
| `clip_range` | `float` | `0.2` | PPO クリップ範囲 |
| `steps_per_update` | `int` | `128` | 更新あたりの PPO ステップ数 |
| `max_steps_per_node` | `int` | `200` | ノードあたりの最大学習ステップ数 |
| `batch_size` | `int` | `16` | 学習バッチサイズ |
| `mini_batch_size` | `int` | `4` | PPO 更新のミニバッチサイズ |
| `gradient_accumulation_steps` | `int` | `4` | 勾配累積ステップ数 |
| `max_grad_norm` | `float` | `0.5` | 勾配クリッピングの最大ノルム |
| `weight_decay` | `float` | `0.01` | 重み減衰 |
| `epochs_per_update` | `int` | `4` | PPO 更新あたりのエポック数 |
| `ppo_trigger_interval` | `int` | `5` | N 個の評価済みノードごとに PPO を実行 |
| `gamma` | `float` | `0.99` | GAE の割引率 |
| `gae_lambda` | `float` | `0.95` | 一般化アドバンテージ推定の lambda |
| `kl_control` | `bool` | `True` | 適応的 KL 係数制御を有効にするか |
| `kl_coef` | `float` | `0.01` | KL ペナルティ係数 |
| `kl_target` | `float` | `0.02` | 目標 KL ダイバージェンス |
| `entropy_coef` | `float` | `0.01` | エントロピーボーナス係数 |
| `value_loss_coef` | `float` | `0.5` | 価値損失係数 |

### 4. LoraRuntimeConfig（LoRA ランタイム設定）

アダプタの保存・継承・キャッシュに関する実行時設定です。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `merge_on_save` | `bool` | `False` | 保存時にアダプタをベースモデルにマージするか |
| `delta_inheritance` | `bool` | `True` | 子ノードが親のアダプタデルタを継承するか |
| `checkpoint_adapter_only` | `bool` | `True` | アダプタ重みのみをチェックポイントするか |
| `squash_depth` | `int` | `6` | デルタ lineage をスカッシュする深さ |
| `snapshot_on_topk` | `bool` | `True` | 上位ノードのスナップショットを作成するか |
| `cache_in_memory` | `bool` | `True` | アダプタ重みをメモリにキャッシュするか |
| `cache_max_entries` | `int` | `10` | LRU キャッシュの最大エントリ数 |

### 5. PruningConfig（枝刈り設定）

探索木の枝刈り戦略を管理します。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `strategy` | `str` | `"reward_threshold"` | 枝刈り戦略名 |
| `reward_threshold` | `float` | `0.0` | この報酬未満のノードを枝刈り |
| `max_consecutive_failures` | `int` | `3` | N 回連続失敗後に枝刈り |
| `keep_topk` | `int` | `5` | 親あたりの上位 K 子ノードのみ保持 |
| `pareto` | `bool` | `True` | パレート支配による枝刈りを有効にするか |
| `lcb_threshold` | `float \| None` | `None` | LCB 閾値枝刈りの割合（None の場合は自動: `best_lcb * 0.5`） |
| `budget_limit.unit` | `str` | `"gpu_minutes"` | 予算の単位 |
| `budget_limit.limit` | `float \| None` | `None` | 予算上限値（None は無制限） |
| `max_stale_nodes` | `int` | `20` | 強制枝刈り前の最大停滞ノード数 |
| `prune_interval` | `int` | `10` | N ステップごとに枝刈りを実行 |

`budget_limit` は `BudgetLimitConfig` というネストモデルで、旧形式のスカラー値もバリデータで自動変換されます。

### 6. TerminationConfig（終了条件）

探索ループの停止条件を定義します。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `max_wall_time_hours` | `float` | `4.0` | 最大実時間（時間） |
| `max_total_experiments` | `int` | `200` | 最大実験総数 |
| `target_score` | `float \| None` | `None` | このスコア達成で停止（None は無効） |
| `min_improvement` | `float` | `0.001` | 進捗とみなす最小改善量 |
| `max_steps` | `int \| None` | `None` | 最大探索ステップ数（None の場合は `max_nodes` を使用） |
| `stop_on_plateau` | `bool` | `False` | プラトーで停止するか |
| `plateau_patience` | `int` | `10` | プラトー停止前の改善なしステップ数 |
| `plateau_min_improvement` | `float` | `0.001` | プラトーカウンタをリセットする最小改善量 |
| `min_nodes_before_stop` | `int` | `10` | 終了を許可する前の最小ノード数 |

### 7. PaperExecConfig（論文生成設定）

Phase 7 の論文生成に関する設定です。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `paper_revision_limit` | `int` | `3` | 最大論文改訂ラウンド数 |
| `auto_compile_latex` | `bool` | `True` | LaTeX を自動コンパイルするか |
| `include_appendix` | `bool` | `True` | 付録セクションを含めるか |
| `auto_ablation` | `bool` | `True` | アブレーション研究を自動生成するか |
| `ablation_components` | `list[str]` | `[]` | アブレーション対象のコンポーネント |
| `n_writeup_reflections` | `int` | `3` | 執筆リフレクションラウンド数 |
| `citation_search_rounds` | `int` | `20` | 引用検索ラウンド数 |
| `plot_aggregation_reflections` | `int` | `5` | プロット集約リフレクション数 |
| `max_figures` | `int` | `12` | 最大図表数 |
| `figure_dpi` | `int` | `300` | 図の解像度（DPI） |
| `vlm_enabled` | `bool` | `True` | VLM ベースの図レビューを有効にするか |

## CLI 引数

`sera freeze-specs` コマンドは以下の引数を受け付けます。これらが `ExecutionSpecModel` の値をオーバーライドします。

### 探索関連

| CLI 引数 | 対応フィールド | デフォルト |
|---|---|---|
| `--max-nodes` | `SearchConfig.max_nodes` | `100` |
| `--max-depth` | `SearchConfig.max_depth` | `10` |
| `--branch-factor` | `SearchConfig.branch_factor` | `3` |
| `--lambda-cost` | `SearchConfig.lambda_cost` | `0.1` |
| `--beta` | `SearchConfig.beta_exploration` | `0.05` |
| `--repeats` | `SearchConfig.repeats` | `3` |
| `--lcb-coef` | `SearchConfig.lcb_coef` | `1.96` |
| `--no-sequential` | `SearchConfig.sequential_eval` を `False` に設定 | `False`（デフォルトは逐次評価有効） |
| `--seq-topk` | `SearchConfig.sequential_eval_topk` | `5` |

### 学習関連

| CLI 引数 | 対応フィールド | デフォルト |
|---|---|---|
| `--rank` | `AdapterSpec.rank` | `16` |
| `--alpha` | `AdapterSpec.alpha` | `32` |
| `--lr` | `LearningConfig.lr` | `1e-4` |
| `--clip` | `LearningConfig.clip_range` | `0.2` |
| `--ppo-steps` | `LearningConfig.steps_per_update` | `128` |
| `--no-kl` | KL 制御を無効化 | `False` |

### モデル・実行環境

| CLI 引数 | 対応フィールド | デフォルト |
|---|---|---|
| `--base-model` | `BaseModelConfig.id` | `"Qwen/Qwen2.5-Coder-7B-Instruct"` |
| `--dtype` | `BaseModelConfig.dtype` | `"bf16"` |

**モデルファミリ自動検出**: `--base-model` で指定されたモデル ID から `BaseModelConfig.family` が自動推定される（`infer_model_family()`）。ファミリに応じて LoRA の `target_modules` がデフォルト値から自動設定される。

| モデルファミリ | 推定パターン | デフォルト target_modules | プロンプト形式 |
|---|---|---|---|
| `qwen2` | ID に "qwen" を含む | `q_proj`, `k_proj`, `v_proj` | ChatML |
| `llama3` | ID に "llama-3" / "llama3" を含む | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Llama3 |
| `deepseek` | ID に "deepseek" を含む | `q_proj`, `v_proj` | DeepSeek |
| `codellama` | ID に "codellama" を含む | `q_proj`, `v_proj` | Llama2 |

カスタムモデルファミリは `model_spec.yaml` の `model_families` フィールドで定義可能:

```yaml
model_families:
  my_custom_model:
    chat_template: "chatml"
    prompt_format: "chatml"
    supports_system_prompt: true
    default_target_modules: ["q_proj", "v_proj", "gate_proj"]
```
| `--agent-llm` | `AgentLLMConfig`（`"provider:model_id"` 形式） | `"local:same_as_base"` |
| `--executor` | `ComputeConfig.executor_type`（`"local"`, `"slurm"`, `"docker"`） | `"local"` |
| `--gpu-count` | `ComputeConfig.gpu_count` | `1` |
| `--memory-gb` | `ComputeConfig.memory_gb` | `32` |
| `--cpu-cores` | `ComputeConfig.cpu_cores` | `8` |
| `--gpu-type` | `ComputeConfig.gpu_type`（`""`, `"A100"`, `"V100"` 等） | `""` |
| `--gpu-required` / `--no-gpu-required` | `ComputeConfig.gpu_required` | `True` |
| `--timeout` | `SandboxConfig.experiment_timeout_sec` | `3600` |
| `--auto` | Phase 1 で LLM による ProblemSpec 自動生成を有効化 | `False` |

**ComputeConfig → submitit 自動マッピング**: `--executor slurm` の場合、上記の `ComputeConfig` フィールドは submitit のネイティブパラメータ（`slurm_gpus_per_node`, `slurm_mem`, `slurm_cpus_per_task`, `constraint`）に自動変換される。`sbatch_extra` で同じパラメータが指定された場合は `sbatch_extra` が優先される。

## Phase 1 のロック機構

### ハッシュロック

Phase 1（`sera freeze-specs`）の実行時に、ExecutionSpec のみが SHA-256 ハッシュでロックされます。

```
specs/execution_spec.yaml  →  SHA-256 ハッシュ  →  specs/execution_spec.yaml.lock
```

ハッシュ計算は `sera.utils.hashing.compute_spec_hash()` で行われます:

```python
def compute_spec_hash(spec_dict: dict) -> str:
    canonical = json.dumps(spec_dict, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    h = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return f"sha256:{h}"
```

### 整合性検証

`sera research` の開始時に `SpecFreezer.verify()` が呼び出されます。`execution_spec.yaml` の現在のハッシュと `.lock` ファイル内のハッシュを比較し、不一致の場合はプロセスが **終了コード 2** で終了します。

```python
# sera/commands/research_cmd.py 内の該当コード
freezer = SpecFreezer()
if not freezer.verify(specs_dir):
    console.print("[red]ExecutionSpec tampered! Aborting.[/red]")
    sys.exit(2)
```

### 凍結される Spec ファイル一覧

Phase 1 で以下の 10 ファイルが `specs/` ディレクトリに YAML として保存されます:

| ファイル名 | 内容 |
|---|---|
| `input1.yaml` | 元の Input-1 仕様 |
| `related_work_spec.yaml` | Phase 0 の先行研究データ |
| `paper_spec.yaml` | 論文メタデータ |
| `paper_score_spec.yaml` | 論文スコアリング |
| `teacher_paper_set.yaml` | 教師論文セット |
| `problem_spec.yaml` | 問題定義・操作変数 |
| `model_spec.yaml` | モデル設定 |
| `resource_spec.yaml` | リソース設定 |
| `plan_spec.yaml` | 研究計画 |
| `execution_spec.yaml` | 実行仕様（ハッシュロック対象） |

ただし、ハッシュロックの対象は `execution_spec.yaml` のみです。

## 後方互換性バリデータ

旧フィールド名からの自動変換が `model_validator(mode="before")` で実装されています:

| 旧フィールド名 | 新フィールド名 | 対象クラス |
|---|---|---|
| `keep_top_k` | `keep_topk` | `PruningConfig` |
| `max_wallclock_hours` | `max_wall_time_hours` | `TerminationConfig` |
| `max_revisions` | `paper_revision_limit` | `PaperExecConfig` |

また、`PruningConfig.budget_limit` はスカラー値（`int` / `float`）が渡された場合、自動的に `BudgetLimitConfig(unit="gpu_minutes", limit=値)` に変換されます。

## 環境変数

SERA は API キーの値ではなく、**環境変数名**を `ResourceSpec.api_keys`（`ApiKeysConfig`）に格納します。実際のキー値は実行時に `os.environ` から取得されます。

| 環境変数名 | 用途 | デフォルトの変数名 |
|---|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API 認証 | `api_keys.semantic_scholar` |
| `CROSSREF_EMAIL` | CrossRef polite pool のメールアドレス | `api_keys.crossref_email` |
| `SERPAPI_API_KEY` | SerpAPI（Google Scholar 検索）認証 | `api_keys.serpapi` |
| `OPENAI_API_KEY` | OpenAI API 認証（agent_llm で使用時） | `api_keys.openai` |
| `ANTHROPIC_API_KEY` | Anthropic API 認証（agent_llm で使用時） | `api_keys.anthropic` |
| `HF_TOKEN` | HuggingFace モデルダウンロード認証 | （ResourceSpec 外で直接参照） |

## PlanSpecModel（研究計画仕様）

`sera.specs.plan_spec.PlanSpecModel` は研究計画・報酬設定・学習手法の選択を管理するモデルです。Phase 1 で凍結されますが、ハッシュロックの対象ではありません。

### RewardConfig（報酬設定）

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `method` | `Literal["outcome_rm", "mt_grpo", "tool_aware", "hiper"]` | `"outcome_rm"` | 報酬計算手法。`compute_reward()` がこの値でディスパッチする |
| `constraint_penalty` | `float` | `10.0` | 制約違反 1 件あたりのペナルティ |
| `kl_coef_in_reward` | `float` | `0.01` | KL ダイバージェンスの報酬ペナルティ係数 |

`method` の選択肢:

| 手法 | 説明 |
|------|------|
| `outcome_rm` | 従来の報酬計算（primary_value - penalties）。デフォルト |
| `mt_grpo` | Multi-Turn GRPO。ターンレベル報酬の重み付き和（`turn_rewards` が必要） |
| `tool_aware` | Tool-Aware 報酬。`mt_grpo` をベースに、ツール使用効率のボーナスと失敗ペナルティを加算（`R_adj = R_base + efficiency_bonus - failure_penalty`） |
| `hiper` | HiPER 階層的報酬。報酬値は `mt_grpo` と同一だが、Advantage 分解が 3 層階層的になる |

### TurnRewardSpec（ターンレベル報酬設定）

`method` が `mt_grpo` または `hiper` の場合に使用される Phase 毎の報酬評価器定義です。`PlanSpecModel.turn_rewards` に設定します。

```yaml
turn_rewards:
  phase_rewards:
    phase0: { evaluator: "citation_relevance", weight: 0.10 }
    phase2: { evaluator: "hypothesis_novelty", weight: 0.15 }
    phase3: { evaluator: "code_executability", weight: 0.25 }
    phase4: { evaluator: "metric_improvement", weight: 0.35 }
    phase7: { evaluator: "paper_score_delta",  weight: 0.15 }
```

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `phase_rewards` | `dict[str, PhaseRewardConfig]` | `{}` | Phase 名 → 報酬評価器の設定 |

**PhaseRewardConfig**:

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `evaluator` | `str` | （必須） | 評価器名（`citation_relevance`, `hypothesis_novelty`, `code_executability`, `metric_improvement`, `paper_score_delta`） |
| `weight` | `float` | `0.0` | MT-GRPO の重み付き和における重み |

### EchoConfig（ECHO 軽量版設定）

失敗ノードからの知識抽出・兄弟ノードへの注入（ECHO 軽量版）を制御する設定です。`PlanSpecModel.echo` に設定します。

```yaml
echo:
  enabled: false
  max_summaries_per_node: 3
  summary_max_tokens: 256
```

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `enabled` | `bool` | `False` | ECHO 機能の有効化。`True` の場合、失敗ノードの知識を兄弟に注入する |
| `max_summaries_per_node` | `int` | `3` | 1 ノードに注入する最大失敗サマリ数 |
| `summary_max_tokens` | `int` | `256` | 失敗サマリのエラーメッセージ・教訓の最大トークン長 |

### HiperConfig（HiPER 階層的 Advantage 設定）

`method` が `hiper` の場合に使用される 3 層階層的 Advantage 分解の設定です。`PlanSpecModel.hiper` に設定します。

```yaml
hiper:
  switch_level_weight: 0.3
  high_level_weight: 0.4
  low_level_weight: 0.3
  bootstrap_at_boundaries: true
```

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `switch_level_weight` | `float` | `0.3` | Switch レベル（Phase 間バランス）の Advantage 重み |
| `high_level_weight` | `float` | `0.4` | High レベル（全体報酬 - 価値）の Advantage 重み |
| `low_level_weight` | `float` | `0.3` | Low レベル（Phase 平均報酬 - 価値）の Advantage 重み |
| `bootstrap_at_boundaries` | `bool` | `True` | 境界でのブートストラップを有効にするか |

3 つの重みの合計は 1.0 になるべきです。

### plan_spec.yaml の設定例

```yaml
reward:
  method: "mt_grpo"
  constraint_penalty: 10.0
  kl_coef_in_reward: 0.01

turn_rewards:
  phase_rewards:
    phase0: { evaluator: "citation_relevance", weight: 0.10 }
    phase2: { evaluator: "hypothesis_novelty", weight: 0.15 }
    phase3: { evaluator: "code_executability", weight: 0.25 }
    phase4: { evaluator: "metric_improvement", weight: 0.35 }
    phase7: { evaluator: "paper_score_delta",  weight: 0.15 }

echo:
  enabled: true
  max_summaries_per_node: 3
  summary_max_tokens: 256

hiper:
  switch_level_weight: 0.3
  high_level_weight: 0.4
  low_level_weight: 0.3
  bootstrap_at_boundaries: true
```

**後方互換性**: 旧 YAML にこれらのフィールドがなくても Pydantic デフォルト値で動作します（`method="outcome_rm"`, `echo.enabled=false`, `turn_rewards=None`, `hiper=None`）。

### AgentCommandsConfig（エージェントコマンド設定）

`PlanSpecModel.agent_commands` はツール・関数・ループ設定を一括管理するモデルです（§5.8）。

#### ToolsConfig（ツール設定）

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `enabled` | `bool` | `False` | ツール実行を有効にするか |
| `api_rate_limit_per_minute` | `int` | `30` | 外部 API のレート制限 |
| `available_tools` | `dict[str, list[str]]` | 下記参照 | カテゴリ別の利用可能ツール（4 カテゴリ・18 ツール） |
| `phase_tool_map` | `dict[str, list[str]]` | 下記参照 | Phase ごとのツール制限 |

**available_tools のデフォルト（4 カテゴリ・18 ツール）:**

| カテゴリ | ツール |
|---------|--------|
| `search` | `semantic_scholar_search`, `semantic_scholar_references`, `semantic_scholar_citations`, `crossref_search`, `arxiv_search`, `web_search` |
| `execution` | `execute_experiment`, `execute_code_snippet`, `run_shell_command` |
| `file` | `read_file`, `write_file`, `read_metrics`, `read_experiment_log`, `list_directory` |
| `state` | `get_node_info`, `list_nodes`, `get_best_node`, `get_search_stats` |

**phase_tool_map のデフォルト:**

| Phase | 許可ツール |
|-------|----------|
| `phase0` | `semantic_scholar_search`, `semantic_scholar_references`, `semantic_scholar_citations`, `crossref_search`, `arxiv_search`, `web_search` |
| `phase2` | `get_node_info`, `list_nodes`, `get_best_node`, `get_search_stats`, `read_metrics`, `read_experiment_log`, `read_file` |
| `phase3` | `read_file`, `write_file`, `read_experiment_log`, `execute_code_snippet`, `read_metrics`, `list_directory` |
| `phase7` | `semantic_scholar_search`, `web_search`, `execute_code_snippet`, `read_file`, `read_metrics`, `list_directory` |

#### FunctionsConfig（関数設定）

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `available_functions` | `dict[str, list[str]]` | 下記参照 | カテゴリ別の利用可能関数（6 カテゴリ・19 関数） |
| `function_tool_bindings` | `dict[str, list[str]]` | 下記参照 | 関数→ツールバインディング（未登録の関数は SINGLE_SHOT） |

**available_functions のデフォルト（6 カテゴリ・19 関数）:**

| カテゴリ | 関数 |
|---------|------|
| `search` | `search_draft`, `search_debug`, `search_improve` |
| `execution` | `experiment_code_gen` |
| `spec` | `spec_generation_problem`, `spec_generation_plan` |
| `paper` | `paper_outline`, `paper_draft`, `paper_reflection`, `aggregate_plot_generation`, `aggregate_plot_fix`, `citation_identify`, `citation_select`, `citation_bibtex` |
| `evaluation` | `paper_review`, `paper_review_reflection`, `meta_review` |
| `phase0` | `query_generation`, `paper_clustering` |

**function_tool_bindings のデフォルト（10 バインディング）:**

| 関数名 | バインドされるツール |
|--------|-------------------|
| `search_draft` | `get_node_info`, `list_nodes`, `read_metrics` |
| `search_debug` | `read_experiment_log`, `read_file`, `execute_code_snippet` |
| `search_improve` | `get_best_node`, `read_metrics`, `get_search_stats` |
| `experiment_code_gen` | `read_file`, `execute_code_snippet` |
| `query_generation` | `semantic_scholar_search`, `arxiv_search` |
| `citation_identify` | `semantic_scholar_search`, `web_search` |
| `citation_select` | `semantic_scholar_search` |
| `aggregate_plot_generation` | `execute_code_snippet` |
| `aggregate_plot_fix` | `execute_code_snippet` |
| `paper_clustering` | `semantic_scholar_search` |

#### LoopDefaults（ループデフォルト設定）

`agent_commands.loop_defaults` で AgentLoop のデフォルトパラメータを設定します。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `max_steps` | `int` | `10` | ReAct ループの最大ステップ数 |
| `tool_call_budget` | `int` | `20` | ループあたりのツール呼び出し上限 |
| `observation_max_tokens` | `int` | `2000` | ツール観察の最大トークン数 |
| `timeout_sec` | `float` | `300.0` | ループのタイムアウト（秒） |

#### FunctionLoopOverride（関数別ループオーバーライド）

`agent_commands.function_loop_overrides` で関数ごとに `LoopDefaults` を上書きできます。各フィールドが `None` の場合は `loop_defaults` の値が使われます。

**デフォルトのオーバーライド:**

| 関数名 | max_steps | tool_call_budget | timeout_sec |
|--------|-----------|-----------------|-------------|
| `search_draft` | 5 | 10 | 120 |
| `search_debug` | 5 | 10 | 120 |
| `search_improve` | 5 | 10 | 120 |
| `experiment_code_gen` | 8 | 15 | 180 |
| `query_generation` | 5 | 10 | 120 |
| `citation_identify` | 8 | 15 | 180 |
| `citation_select` | 5 | 10 | 120 |
| `aggregate_plot_generation` | 5 | 10 | 120 |
| `aggregate_plot_fix` | 5 | 10 | 120 |
| `paper_clustering` | 3 | 5 | 60 |

#### ToolConfig（後方互換ラッパー）

`PlanSpecModel.tools` はフラットなフィールド構成で既存コードとの後方互換性を提供します。`agent_commands` のネスト形式が渡された場合、`_migrate_from_agent_commands` バリデータが自動的にフラットフィールドに変換します。

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `enabled` | `bool` | `False` | ツール実行を有効にするか |
| `max_steps_per_loop` | `int` | `10` | ReAct ループの最大ステップ数 |
| `tool_call_budget_per_loop` | `int` | `20` | ツール呼び出し上限 |
| `observation_max_tokens` | `int` | `2000` | ツール観察の最大トークン数 |
| `loop_timeout_sec` | `float` | `300.0` | ループタイムアウト（秒） |
| `api_rate_limit_per_minute` | `int` | `30` | 外部 API レート制限 |

#### agent_commands バリデーション（§5.8.4）

`SpecFreezer.freeze()` は ExecutionSpec ロック前に `_validate_agent_commands()` を呼び出し、以下の整合性チェックを実行します:

1. **ツール存在チェック**: `function_tool_bindings` で参照されるツールが `available_tools` に存在するか
2. **Phase ツール整合性チェック**: 関数にバインドされたツールが、その関数の Phase に対応する `phase_tool_map` のサブセットであるか

いずれも警告ログの出力のみで、プロセスは継続します。

### plan_spec.yaml の設定例（agent_commands 含む）

```yaml
agent_commands:
  tools:
    enabled: true
    api_rate_limit_per_minute: 30
    available_tools:
      search:
        - semantic_scholar_search
        - crossref_search
        - arxiv_search
      execution:
        - execute_experiment
        - execute_code_snippet
      file:
        - read_file
        - write_file
      state:
        - get_node_info
        - list_nodes
    phase_tool_map:
      phase0:
        - semantic_scholar_search
        - crossref_search
        - arxiv_search
      phase3:
        - read_file
        - write_file
        - execute_code_snippet
  functions:
    available_functions:
      search:
        - search_draft
        - search_debug
        - search_improve
      execution:
        - experiment_code_gen
    function_tool_bindings:
      search_draft:
        - get_node_info
        - list_nodes
      experiment_code_gen:
        - read_file
        - execute_code_snippet
  loop_defaults:
    max_steps: 10
    tool_call_budget: 20
    observation_max_tokens: 2000
    timeout_sec: 300.0
  function_loop_overrides:
    search_draft:
      max_steps: 5
      tool_call_budget: 10
      timeout_sec: 120
    experiment_code_gen:
      max_steps: 8
      tool_call_budget: 15
      timeout_sec: 180
```

## 三層変数可変性モデル

SERA は変数の可変性を三つの層に厳密に分離しています:

| 層 | 格納場所 | Phase 1 以降の変更 | 例 |
|---|---|---|---|
| **凍結層（Frozen）** | `ExecutionSpec` | 不可 | `lr`, `clip_range`, `repeats`, `lcb_coef`, `max_nodes` |
| **操作層（Manipulated）** | `ProblemSpec.manipulated_variables` | ノードごとに分岐可能 | 実験の learning_rate, batch_size, method |
| **導出層（Derived）** | ランタイム計算 | 自動計算のみ | `priority`, `mu`, `se`, `lcb`, `feasible`, `reward` |

凍結層と操作層の変数は類似した名前を持つ場合がありますが（例: `ExecutionSpec.learning.lr` と `ProblemSpec.manipulated_variables[].learning_rate`）、異なる層に属する別の概念です。凍結層は PPO 学習のハイパーパラメータを固定し、操作層は各探索ノードが分岐する実験パラメータを定義します。
