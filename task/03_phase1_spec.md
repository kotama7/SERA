# SERA 要件定義書 — Phase 1: Spec確定

> 本ファイルは TASK.md v13.0 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 5. Phase 1：Spec確定（最重要：ExecutionSpec固定）

### 5.1 目的
- Phase 0出力を取り込み、研究計画と実行パラメータを **完全に確定**
- **Phase 2–5の規定値（探索/評価/学習/剪定/停止/予算）をここで固定し、以後変更禁止**

### 5.2 Spec確定プロセス（具体）

```text
1. Phase 0 出力（RelatedWorkSpec, PaperSpec, PaperScoreSpec, TeacherPaperSet）をロード
2. LLM（AgentLLM、LoRA無し）に Input-1 + Phase 0 出力を与え、以下のSpec草案を生成：
   a. ProblemSpec（§5.5）
   b. PlanSpec（§5.7）— agent_commands（§5.8）を含む
3. ユーザが ModelSpec（§5.3）と ResourceSpec（§5.6）を確認・修正
   - CLIモード: 生成されたYAMLをエディタで開き、保存で確定
   - 自動モード（--auto）: LLM生成のまま確定
4. ExecutionSpec（§5.4）を CLI 引数 + 既定値から構築
5. agent_commands の整合性検証（§5.8.4）:
   - function_tool_bindings の各ツールが available_tools に含まれるか
   - function_tool_bindings のツールが phase_tool_map のサブセットか
   - available_functions の全関数が AgentFunctionRegistry に登録済みか
6. 全Spec を specs/ に保存
7. ExecutionSpec の SHA-256 ハッシュを specs/execution_spec.yaml.lock に記録
8. 以後、ExecutionSpec のロード時にハッシュを検証（不一致は致命的エラー）
```

### 5.3 ModelSpec（LoRA形状固定：必須）
```yaml
model_spec:
  base_model:
    id: "Qwen/Qwen2.5-Coder-7B-Instruct"  # 既定値。コード生成能力が必要
    revision: ""          # gitハッシュまたはtag（実行時に自動取得し固定）
    dtype: "bf16"
    load_method: "auto"   # "auto" | "4bit" | "8bit"（bitsandbytes量子化）
    max_seq_len: 8192
  agent_llm:
    # 仮説生成・論文執筆等に使うLLM（ベースモデルと同一でも別でもよい）
    provider: "local"     # "local" | "openai" | "anthropic"
    model_id: "same_as_base"  # "same_as_base" の場合はベースモデル+LoRAを使用
    # provider が "openai"/"anthropic" の場合は以下を使用：
    # model_id: "gpt-4o" / "claude-sonnet-4-20250514" 等
    # api_key_env: "OPENAI_API_KEY" / "ANTHROPIC_API_KEY"
    temperature: 0.7      # 仮説生成用
    max_tokens: 4096
  adapter_spec:
    type: "lora"
    target_modules: ["q_proj", "v_proj"]  # 既定値（最小構成）
    target_layers: "all"  # "all" | [0,1,2,...] | "0-15"
    rank: 16
    alpha: 32
    dropout: 0.05
    init: "zero"          # "zero" | "gaussian" | "kaiming"。zero推奨（初期状態でベースモデルと同一出力）
    delta_inheritance: true
  vlm:                          # VLM設定（Phase 7 のVLM統合に使用、オプション）
    provider: "openai"         # "openai" | "anthropic" | null（null=VLM無効）
    model_id: "gpt-4o"         # VLMモデル名
    # api_key_env: agent_llm と共有（同一providerの場合）または別途指定
    max_tokens: 4096
    temperature: 0.7
    max_images_per_call: 25    # 1回のAPI呼び出しで送信する最大画像数
  compatibility:
    adapter_spec_hash: "" # SHA-256(type+target_modules+target_layers+rank+alpha)（自動計算）
    tokenizer_revision: ""  # 自動取得
```

### 5.4 ExecutionSpec（Phase 2–5の規定値をここで固定：必須）
> **探索開始後に変更は禁止**。変更したい場合は新規runとして再初期化する。

```yaml
execution_spec:
  search:
    strategy: "best_first"
    priority_rule: "epsilon_constraint_lcb"
    lambda_cost: 0.1
    beta_exploration: 0.05
    max_nodes: 100
    max_depth: 10
    branch_factor: 3       # improve 時の子ノード生成数
    initial_root_children: 5  # ルートノードの初期 draft 数
    max_debug_depth: 3     # debug オペレータの最大連鎖深度（AIDE参考）
    min_diverse_methods: 3 # draft 再発動の多様性閾値（ユニーク手法数がこれ未満で draft）
    draft_trigger_after: 10 # draft 再発動の前提条件（評価済みノード数がこれ以上）

  evaluation:
    repeats: 3
    lcb_coef: 1.96          # 95%信頼区間
    sequential_eval: true
    sequential_eval_initial: 1   # 逐次評価の初回実行回数
    sequential_eval_topk: 5      # 追加評価を行う上位k候補
    bootstrap: false
    bootstrap_samples: 1000      # bootstrap=true 時のリサンプリング回数

  learning:
    algorithm: "ppo"
    update_target: "lora_only"
    clip_range: 0.2
    lr: 1e-4
    lr_scheduler: "cosine"        # "constant" | "cosine" | "linear_decay"
    steps_per_update: 128
    batch_size: 16
    mini_batch_size: 4
    epochs_per_update: 4
    gamma: 0.99                   # 割引率
    gae_lambda: 0.95              # GAE パラメータ
    kl_control: true
    kl_coef: 0.01                 # KLペナルティ係数
    kl_target: 0.02               # 目標KL divergence
    entropy_coef: 0.01            # エントロピーボーナス
    max_grad_norm: 0.5            # 勾配クリッピング
    value_loss_coef: 0.5
    ppo_trigger_interval: 5       # N個の評価済みノードごとにPPO更新

  lora_runtime:
    delta_inheritance: true
    squash_depth: 6
    snapshot_on_topk: true
    cache_in_memory: true
    cache_max_entries: 10         # メモリ上に保持する最大LoRA数

  pruning:
    pareto: true
    lcb_threshold: null           # null = 自動（best_lcb * 0.5）
    budget_limit:
      unit: "gpu_minutes"
      limit: null                 # null = 無制限
    max_stale_nodes: 20           # 評価後に改善されなかったノード数の上限
    keep_topk: 5                  # 剪定時に必ず保持する上位ノード数

  termination:
    stop_on_plateau: true
    plateau_patience: 10          # 最良LCBが10ステップ改善しなければ停止
    plateau_min_improvement: 0.001  # 改善とみなす最小値
    max_wall_time_hours: null     # null = 無制限
    min_nodes_before_stop: 10     # 最低限この数のノードを評価してから停止判定

  paper:
    paper_revision_limit: 3       # Phase 8 の最大改善ループ回数
    auto_ablation: true           # Phase 7 でアブレーション実験を自動実行するか
    ablation_components: []       # 空=自動検出。手動指定も可

    # --- AI-Scientist-v2 スタイル拡張（Phase 7） ---
    n_writeup_reflections: 3      # ライティング内反省ループ最大回数
    citation_search_rounds: 20    # 自動引用検索の最大ラウンド数
    plot_aggregation_reflections: 5  # 図集約スクリプトの反省ループ回数
    max_figures: 12               # 論文内の最大図数
    figure_dpi: 300               # 図の解像度
    vlm_enabled: true             # VLM統合を有効にするか（ModelSpec.vlm が必要）
```

### 5.5 ProblemSpec（必須：Phase 1で確定）
```yaml
problem_spec:
  title: ""                       # 研究タイトル（LLM生成→ユーザ承認）
  objective:
    description: ""               # 目的の自然言語記述
    metric_name: "score"          # primary metric の名前
    direction: "maximize"         # "maximize" | "minimize"
  constraints:
    - name: "format_valid"
      type: "bool"               # "bool" | "ge" | "le"
      threshold: true             # bool: true必須、ge/le: 数値
      epsilon: 0.0               # ε-constraint の許容幅
  secondary_metrics:
    - name: "cost"
      direction: "minimize"
      weight_in_tiebreak: 0.3    # タイブレーク時の重み
  manipulated_variables:          # 操作変数（実験で変化させるもの）
    - name: "learning_rate"
      type: "float"
      range: [1e-6, 1e-2]
      scale: "log"               # "linear" | "log"
    - name: "batch_size"
      type: "int"
      range: [8, 128]
      scale: "linear"
    - name: "method"
      type: "categorical"
      choices: ["baseline_A", "baseline_B", "proposed"]
  observed_variables:             # 観測変数（測定するもの）
    - name: "accuracy"
      type: "float"
    - name: "wall_time_sec"
      type: "float"
    - name: "gpu_memory_mb"
      type: "float"
  evaluation_design:
    type: "holdout"               # "holdout" | "cross_validation" | "bootstrap"
    test_split: 0.2               # holdout の場合
    cv_folds: null                # cross_validation の場合
  experiment_template: |
    # このテンプレートは Phase 3 で LLM がカスタマイズする
    # {variable_name} は操作変数で置換される
    python experiment.py \
      --lr {learning_rate} \
      --batch-size {batch_size} \
      --method {method} \
      --seed {seed} \
      --output-dir {output_dir}
```

### 5.6 ResourceSpec（必須：Phase 1で確定）
```yaml
resource_spec:
  compute:
    executor_type: "local"         # "local" | "slurm" | "docker"
    gpu_required: true
    gpu_type: ""                   # "A100" 等（空=任意）
    gpu_count: 1
    cpu_cores: 8
    memory_gb: 32
    slurm:                         # executor_type="slurm" 時のみ
      partition: "gpu"
      account: ""
      time_limit: "04:00:00"
      modules: ["cuda/12.1", "python/3.11"]
      sbatch_extra: []             # 追加の #SBATCH 行
    docker:                        # executor_type="docker" 時のみ
      image: "pytorch/pytorch:2.3.0-cuda12.1-cudnn9-devel"
      volumes: []
      env_vars: {}
      gpu_runtime: "nvidia"

  network:
    allow_internet: true           # Phase 3 の実験中にネットアクセスを許可するか
    allow_api_calls: true          # Phase 0/7/8 でAPI呼び出しを許可するか

  api_keys:                        # 環境変数名を指定（値そのものは書かない）
    semantic_scholar: "SEMANTIC_SCHOLAR_API_KEY"  # 任意
    crossref_email: "CROSSREF_EMAIL"              # polite pool 用
    serpapi: "SERPAPI_API_KEY"                     # 任意
    openai: "OPENAI_API_KEY"                      # agent_llm.provider="openai" 時
    anthropic: "ANTHROPIC_API_KEY"                # agent_llm.provider="anthropic" 時

  storage:
    work_dir: "./sera_workspace"   # 全出力のルートディレクトリ
    max_disk_gb: 50                # ディスク使用量上限

  sandbox:
    experiment_timeout_sec: 3600   # 1実験あたりのタイムアウト（秒）
    experiment_memory_limit_gb: 16 # 1実験あたりのメモリ上限
    isolate_experiments: true      # 実験をサブプロセスで隔離するか
```

### 5.7 PlanSpec（必須：Phase 1で確定）
```yaml
plan_spec:
  search_strategy:
    name: "best_first"            # ExecutionSpec.search.strategy と一致
    description: "LCBベースのBest-First探索"

  branching:
    generator: "llm"              # "llm" | "template" | "random"
    operators:                    # AIDE参考の3オペレータ（§6.5）
      - name: "draft"
        description: "新規アプローチの起草（ルート初期化 + 多様性不足時の再発動）"
        selection: "auto"         # §6.4 select_next_node() が状態に基づき自動選択
      - name: "debug"
        description: "失敗実験のエラー修復（深度制限付き、§6.5.2）"
        selection: "auto"
      - name: "improve"
        description: "最良ノードへの原子的改善（単一変数変更、§6.5.3）"
        selection: "auto"

  reward:
    formula: "primary - penalty(constraints) - lambda_cost * cost"
    primary_source: "metrics.primary.value"
    constraint_penalty: 10.0       # 制約違反1つあたりのペナルティ
    cost_source: "metrics.secondary[name='cost'].value"
    kl_penalty: true               # KLペナルティを報酬に含めるか
    kl_coef_in_reward: 0.01        # 報酬内のKL係数

  turn_rewards:                    # MT-GRPO ターンレベル報酬設定（§26.4.2）
    enabled: true                  # ターンレベル報酬を有効にするか（Phase A）
    phase_rewards:
      phase0:
        evaluator: "citation_relevance"    # 収集論文の関連性スコア
        weight: 0.1
      phase2:
        evaluator: "hypothesis_novelty"    # 仮説の新規性（既存ノードとの類似度逆数）
        weight: 0.15
      phase3:
        evaluator: "code_executability"    # コードが正常実行できたか（binary）
        weight: 0.25
      phase4:
        evaluator: "metric_improvement"    # 親ノード比のメトリクス改善率
        weight: 0.35
      phase7:
        evaluator: "paper_score_delta"     # PaperScore改善幅
        weight: 0.15

  agent_commands:                  # エージェントが使用可能なツール・関数の定義（§5.8）
    tools:
      enabled: true                # ツール実行エンジン（AgentLoop）を有効化するか
      api_rate_limit_per_minute: 30  # 外部API系ツールのレート制限

      # --- 利用可能ツール一覧（18ツール、カテゴリ別） ---
      available_tools:
        search:                    # Web/API 検索ツール
          - "semantic_scholar_search"
          - "semantic_scholar_references"
          - "semantic_scholar_citations"
          - "crossref_search"
          - "arxiv_search"
          - "web_search"
        execution:                 # コード実行ツール
          - "execute_experiment"
          - "execute_code_snippet"
          - "run_shell_command"
        file:                      # ファイル操作ツール
          - "read_file"
          - "write_file"
          - "read_metrics"
          - "read_experiment_log"
          - "list_directory"
        state:                     # 探索状態参照ツール
          - "get_node_info"
          - "list_nodes"
          - "get_best_node"
          - "get_search_stats"

      # --- Phase別ツール許可マップ ---
      # 各Phaseで使用可能なツールを制限（安全性・効率性のため）
      phase_tool_map:
        phase0:                    # 先行研究収集
          - "semantic_scholar_search"
          - "semantic_scholar_references"
          - "semantic_scholar_citations"
          - "crossref_search"
          - "arxiv_search"
          - "web_search"
        phase2:                    # 探索木生成（仮説立案）
          - "get_node_info"
          - "list_nodes"
          - "get_best_node"
          - "get_search_stats"
          - "read_metrics"
          - "read_experiment_log"
          - "read_file"
        phase3:                    # 実験実行（コード生成）
          - "read_file"
          - "write_file"
          - "read_experiment_log"
          - "execute_code_snippet"
          - "read_metrics"
          - "list_directory"
        phase7:                    # 論文生成
          - "semantic_scholar_search"
          - "web_search"
          - "execute_code_snippet"
          - "read_file"
          - "read_metrics"
          - "list_directory"

    functions:
      # --- 利用可能関数一覧（19関数、Phase別） ---
      available_functions:
        search:                    # 探索系（Phase 2）
          - "search_draft"         # 新規アプローチ起草
          - "search_debug"         # エラー修復
          - "search_improve"       # 原子的改善
        execution:                 # 実行系（Phase 3）
          - "experiment_code_gen"  # 実験コード生成
        spec:                      # Spec生成系（Phase 1）
          - "spec_generation_problem"  # ProblemSpec生成
          - "spec_generation_plan"     # PlanSpec生成
        paper:                     # 論文系（Phase 7）
          - "paper_outline"
          - "paper_draft"
          - "paper_reflection"
          - "aggregate_plot_generation"
          - "aggregate_plot_fix"
          - "citation_identify"
          - "citation_select"
          - "citation_bibtex"
        evaluation:                # 評価系（Phase 8）
          - "paper_review"
          - "paper_review_reflection"
          - "meta_review"
        phase0:                    # 先行研究系（Phase 0）
          - "query_generation"
          - "paper_clustering"

      # --- 関数→ツールバインディング ---
      # AgentLoop使用時に各関数が呼び出せるツール（allowed_tools = None の関数は単発生成）
      function_tool_bindings:
        search_draft: ["get_node_info", "list_nodes", "read_metrics"]
        search_debug: ["read_experiment_log", "read_file", "execute_code_snippet"]
        search_improve: ["get_best_node", "read_metrics", "get_search_stats"]
        experiment_code_gen: ["read_file", "execute_code_snippet"]
        query_generation: ["semantic_scholar_search", "arxiv_search"]
        citation_identify: ["semantic_scholar_search", "web_search"]
        citation_select: ["semantic_scholar_search"]
        aggregate_plot_generation: ["execute_code_snippet"]
        aggregate_plot_fix: ["execute_code_snippet"]
        paper_clustering: ["semantic_scholar_search"]
        # 以下の関数は allowed_tools = null（常に単発生成）
        # spec_generation_problem, spec_generation_plan,
        # paper_outline, paper_draft, paper_reflection, citation_bibtex,
        # paper_review, paper_review_reflection, meta_review

    loop_defaults:                 # AgentLoop 既定パラメータ
      max_steps: 10                # ReActループの最大ステップ数
      tool_call_budget: 20         # ループあたりのツール呼び出し上限
      observation_max_tokens: 2000 # ツール観測結果の最大トークン数
      timeout_sec: 300.0           # ループタイムアウト（秒）

    # --- 関数別 loop_config オーバーライド ---
    # loop_defaults と異なる値を持つ関数のみ記載
    function_loop_overrides:
      search_draft:       { max_steps: 5, tool_call_budget: 10, timeout_sec: 120 }
      search_debug:       { max_steps: 5, tool_call_budget: 10, timeout_sec: 120 }
      search_improve:     { max_steps: 5, tool_call_budget: 10, timeout_sec: 120 }
      experiment_code_gen: { max_steps: 8, tool_call_budget: 15, timeout_sec: 180 }
      query_generation:   { max_steps: 5, tool_call_budget: 10, timeout_sec: 120 }
      citation_identify:  { max_steps: 8, tool_call_budget: 15, timeout_sec: 180 }
      citation_select:    { max_steps: 5, tool_call_budget: 10, timeout_sec: 120 }
      aggregate_plot_generation: { max_steps: 5, tool_call_budget: 10, timeout_sec: 120 }
      aggregate_plot_fix: { max_steps: 5, tool_call_budget: 10, timeout_sec: 120 }
      paper_clustering:   { max_steps: 3, tool_call_budget: 5, timeout_sec: 60 }

  logging:
    log_every_node: true
    log_llm_prompts: true          # LLMへの入出力を全記録
    log_llm_responses: true
    checkpoint_interval: 10        # Nノードごとに探索状態をチェックポイント

  artifacts:
    save_all_experiments: true     # 全ノードの実験結果を保持
    save_pruned: false             # 剪定されたノードの結果を保持するか
    export_format: "json"          # "json" | "yaml"
```

### 5.8 agent_commands 設計説明

#### 5.8.1 概要
`agent_commands` は PlanSpec 内でエージェントが使用可能な**ツール**（外部操作）と**関数**（LLM呼び出しタスク）を Phase 1 で確定する。Phase 1 以降、この定義は ExecutionSpec と同様に凍結され変更禁止となる。

#### 5.8.2 ツール（18種）とカテゴリ

| カテゴリ | ツール数 | 用途 | 実装モジュール |
|---------|---------|------|--------------|
| search | 6 | Web/API経由の論文・情報検索 | `agent/tools/search_tools.py` |
| execution | 3 | コード実行・実験実行 | `agent/tools/execution_tools.py` |
| file | 5 | ファイル読み書き・メトリクス参照 | `agent/tools/file_tools.py` |
| state | 4 | 探索木の状態参照 | `agent/tools/state_tools.py` |

#### 5.8.3 関数（19種）と呼び出しパターン

| パターン | 関数数 | 条件 | 動作 |
|---------|--------|------|------|
| AGENT_LOOP | 10 | `tools.enabled=true` かつ `function_tool_bindings` にエントリあり | AgentLoop（ReActループ）経由でツールを使いながら生成 |
| SINGLE_SHOT | 9 | `function_tool_bindings` にエントリなし | 単発 `generate()` で処理 |

`tools.enabled=false` の場合、全19関数が SINGLE_SHOT にフォールバックする。

#### 5.8.4 phase_tool_map の意味

各 Phase で AgentLoop が使用できるツールを制限する。`function_tool_bindings` で定義された関数のツールは、その関数が属する Phase の `phase_tool_map` のサブセットでなければならない（検証は Phase 1 の freeze 時に実行）。

```text
検証ルール:
  ∀ func ∈ available_functions:
    func.function_tool_bindings ⊆ phase_tool_map[func.phase]
```

#### 5.8.5 loop_defaults と function_loop_overrides

`loop_defaults` は AgentLoop の全体既定値。`function_loop_overrides` で関数ごとに上書きできる。上書きされていないフィールドは `loop_defaults` の値を継承する。

#### 5.8.6 既存コードとの対応

| PlanSpec フィールド | 対応する実装 |
|-------------------|------------|
| `tools.enabled` | `ToolConfig.enabled` (`plan_spec.py`) |
| `tools.available_tools` | `ToolExecutor.ALL_TOOL_NAMES` (`tool_executor.py`) |
| `functions.available_functions` | `REGISTRY.list_all()` (`agent_functions.py`) |
| `functions.function_tool_bindings` | `AgentFunction.allowed_tools` （各関数定義） |
| `loop_defaults` | `ToolConfig.max_steps_per_loop` 等 (`plan_spec.py`) |
| `function_loop_overrides` | `AgentFunction.loop_config` （各関数定義） |

---
