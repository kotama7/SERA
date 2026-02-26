# SERA 要件定義書 — モジュール構成

> 本ファイルは TASK.md v13.1 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 21. モジュール構成（ソースコード構造）

```text
src/sera/
  __init__.py
  cli.py                          # Typer アプリケーション（§16）

  specs/                           # Pydantic モデル定義
    __init__.py
    input1.py                      # Input1Model
    problem_spec.py                # ProblemSpecModel
    model_spec.py                  # ModelSpecModel
    resource_spec.py               # ResourceSpecModel
    plan_spec.py                   # PlanSpecModel
    execution_spec.py              # ExecutionSpecModel（ハッシュ検証付き）
    paper_spec.py                  # PaperSpecModel
    paper_score_spec.py            # PaperScoreSpecModel
    related_work_spec.py           # RelatedWorkSpecModel
    teacher_paper_set.py           # TeacherPaperSetModel

  phase0/                          # 先行研究
    __init__.py
    related_work_engine.py         # API呼び出し + クエリ構築
    api_clients/
      __init__.py
      semantic_scholar.py          # SemanticScholarClient
      crossref.py                  # CrossRefClient
      arxiv.py                     # ArxivClient
      web_search.py                # WebSearchClient（SerpAPI/Tavily）
      base.py                      # BaseScholarClient（ABC）
    clustering.py                  # LLMベースの論文クラスタリング
    ranking.py                     # citation_norm + relevance_score

  phase1/                          # Spec確定
    __init__.py
    spec_builder.py                # LLM で Spec 草案生成
    spec_freezer.py                # ExecutionSpec ハッシュ固定

  search/                          # Phase 2: 探索（AIDE参考3オペレータ）
    __init__.py
    search_node.py                 # SearchNode データクラス（§6.2、failure_context フィールド含む）
    search_manager.py              # Best-First ループ本体 + select_next_node（§6.4, §6.11）
    tree_ops.py                    # draft / debug / improve 3オペレータ（§6.5）
    priority.py                    # compute_priority, compute_exploration_bonus（§6.3）
    validation.py                  # validate_experiment_config（§6.6.1）
    failure_extractor.py           # Phase B: ECHO軽量版 失敗知識抽出（§25.4.3）

  execution/                       # Phase 3: 実験実行
    __init__.py
    executor.py                    # Executor ABC
    local_executor.py              # LocalExecutor
    slurm_executor.py              # SlurmExecutor
    docker_executor.py             # DockerExecutor
    experiment_generator.py        # LLMによる実験コード生成

  evaluation/                      # Phase 4: 統計評価
    __init__.py
    evaluator.py                   # Evaluator ABC
    statistical_evaluator.py       # μ/SE/LCB 計算 + 逐次評価
    feasibility.py                 # ε-constraint 判定

  learning/                        # Phase 5: PPO
    __init__.py
    ppo_trainer.py                 # PPOTrainer（trl ラップ）
    reward.py                      # compute_reward, compute_reward_v2
    rollout.py                     # PPORollout, PPORolloutV2 データクラス
    turn_reward.py                 # Phase A: ターンレベル報酬評価器（§25.4.2 TurnRewardSpec）
    hierarchical_ppo.py            # Phase C: HiPER 3層階層PPO（§25.5）

  lineage/                         # Phase 6: 系譜管理
    __init__.py
    lineage_manager.py             # materialize, squash, delta保存
    pruner.py                      # Pareto剪定, LCB閾値剪定, 予算剪定
    cache.py                       # LRUCache for LoRA weights

  paper/                           # Phase 7-8: 論文
    __init__.py
    evidence_store.py              # EvidenceStore
    paper_composer.py              # PaperComposer（ライティング内反省ループ含む）
    paper_evaluator.py             # PaperEvaluator（アンサンブル+レビュアー反省ループ）
    figure_generator.py            # matplotlib/seaborn/graphviz 図生成
    vlm_reviewer.py                # VLMReviewer（図記述・キャプションレビュー・重複検出）
    citation_searcher.py           # CitationSearcher（Semantic Scholar自動引用検索ループ）

  agent/                           # エージェントLLM（tool-calling対応）
    __init__.py
    agent_llm.py                   # AgentLLM（ベースモデル+LoRA管理、推論、call_function統一エントリ）
    agent_functions.py             # AgentFunctionRegistry + パースユーティリティ（§27）
    agent_loop.py                  # AgentLoop — ReAct型反復ループ（§28）
    tool_executor.py               # ToolExecutor — 18ツールディスパッチ（§28）
    tool_policy.py                 # ToolPolicy — 安全制御・レート制限（§28）
    prompt_templates.py            # 各Phase用プロンプトテンプレート
    functions/                     # 19関数の定義・ハンドラ（§27）
      __init__.py
      search_functions.py          # search_draft, search_debug, search_improve
      execution_functions.py       # experiment_code_gen
      spec_functions.py            # spec_generation_problem, spec_generation_plan
      paper_functions.py           # paper系8関数
      evaluation_functions.py      # paper_review, paper_review_reflection, meta_review
      phase0_functions.py          # query_generation, paper_clustering
    tools/                         # 18ツールのハンドラ実装（§28）
      __init__.py
      search_tools.py              # Web/API検索ツール（6個）
      execution_tools.py           # コード実行ツール（3個）
      file_tools.py                # ファイルI/Oツール（5個）
      state_tools.py               # 内部状態参照ツール（4個）

  utils/
    __init__.py
    hashing.py                     # SHA-256 ハッシュ計算
    logging.py                     # structlog 設定 + JSONL出力
    checkpoint.py                  # 探索状態のチェックポイント/復帰
    seed.py                        # seed固定ユーティリティ

tests/
  test_specs/                      # Pydantic バリデーションテスト
  test_phase0/                     # API クライアントテスト（モック使用）
  test_search/                     # 探索・分岐生成テスト
  test_evaluation/                 # 統計評価テスト
  test_lineage/                    # materialize/squash テスト
  test_paper/                      # 論文生成テスト
  test_agent/                      # Agent系テスト（§27/§28）
    test_agent_functions.py        # レジストリ・パース・ハンドラ
    test_agent_loop.py             # AgentLoop制御・停止条件
    test_tool_executor.py          # ツールディスパッチ・ポリシー
    test_tool_policy.py            # 許可/拒否判定
  test_learning/                   # PPO・報酬・ターン報酬テスト
  test_cli/                        # CLI 統合テスト
  conftest.py                      # pytest フィクスチャ
```

---
