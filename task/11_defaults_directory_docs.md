# SERA 要件定義書 — 既定値・ディレクトリ構成・docs要件

> 本ファイルは TASK.md v13.1 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 13. 既定値（Defaults）一覧（Phase 1で固定、引数で変更可）

### 13.1 Phase 0（先行研究）
| パラメータ | 既定値 | CLI引数 | 説明 |
|-----------|--------|---------|------|
| top_k_papers | 10 | --topk | 収集する上位論文数 |
| recent_years_bias | 5 | --years-bias | 過去N年を優先 |
| citation_graph_depth | 1 | --citation-depth | 引用グラフ探索深度 |
| teacher_papers | 5 | --teacher-papers | 教師論文数 |
| API優先順位 | SemanticScholar→CrossRef→arXiv→Web | --api-priority | カンマ区切りで変更 |

### 13.2 Phase 2（探索）
| パラメータ | 既定値 | CLI引数 | 説明 |
|-----------|--------|---------|------|
| strategy | best_first | --strategy | 探索戦略 |
| lambda_cost | 0.1 | --lambda-cost | コストペナルティ係数 |
| beta_exploration | 0.05 | --beta | 探索ボーナス係数 |
| max_nodes | 100 | --max-nodes | 最大ノード数 |
| max_depth | 10 | --max-depth | 最大探索深度 |
| branch_factor | 3 | --branch-factor | 子ノード生成数 |

### 13.3 Phase 4（評価）
| パラメータ | 既定値 | CLI引数 | 説明 |
|-----------|--------|---------|------|
| repeats | 3 | --repeats | 反復実行回数 |
| lcb_coef | 1.96 | --lcb-coef | LCB係数（95%CI） |
| sequential_eval | true | --no-sequential | 逐次評価の無効化 |
| sequential_eval_topk | 5 | --seq-topk | 追加評価の上位k |

### 13.4 Phase 5（学習）
| パラメータ | 既定値 | CLI引数 | 説明 |
|-----------|--------|---------|------|
| algorithm | ppo | - | 固定（PPOのみ） |
| clip_range | 0.2 | --clip | PPOクリップ範囲 |
| lr | 1e-4 | --lr | 学習率 |
| steps_per_update | 128 | --ppo-steps | 更新あたりステップ数 |
| kl_control | true | --no-kl | KL制御の無効化 |
| rank | 16 | --rank | LoRAランク |
| alpha | 32 | --alpha | LoRA alpha |

### 13.5 Phase 5 ターン報酬（MT-GRPO）
| パラメータ | 既定値 | CLI引数 | 説明 |
|-----------|--------|---------|------|
| turn_rewards.enabled | true | --no-turn-rewards | ターンレベル報酬の有効/無効 |
| phase0.weight | 0.1 | --turn-w-phase0 | Phase 0 報酬の重み |
| phase2.weight | 0.15 | --turn-w-phase2 | Phase 2 報酬の重み |
| phase3.weight | 0.25 | --turn-w-phase3 | Phase 3 報酬の重み |
| phase4.weight | 0.35 | --turn-w-phase4 | Phase 4 報酬の重み |
| phase7.weight | 0.15 | --turn-w-phase7 | Phase 7 報酬の重み |

### 13.6 Phase 6（系譜）
| パラメータ | 既定値 | CLI引数 | 説明 |
|-----------|--------|---------|------|
| squash_depth | 6 | --squash-depth | スナップショット深度 |
| snapshot_on_topk | true | --no-snapshot-topk | Top-kスナップショット無効化 |

---

## 14. ディレクトリ構成（必須）

```text
sera_workspace/                    # ResourceSpec.storage.work_dir
  specs/
    input1.yaml
    related_work_spec.yaml
    problem_spec.yaml
    model_spec.yaml
    resource_spec.yaml
    plan_spec.yaml
    execution_spec.yaml
    execution_spec.yaml.lock       # SHA-256 ハッシュ（改竄検知）
    paper_spec.yaml
    paper_score_spec.yaml
    teacher_paper_set.yaml

  related_work/
    queries.jsonl                  # 全API呼び出しログ
    results/
      <query_id>.json              # 各クエリの生結果
    teacher_papers/
      <paper_id>.pdf               # ダウンロードした教師論文PDF
      <paper_id>.meta.json         # メタデータ

  lineage/
    nodes/<adapter_node_id>/
      meta.json                    # 系譜メタデータ（§9.4参照）
      adapter_delta.safetensors    # LoRA差分
      snapshot.safetensors         # 条件で生成（§10.3参照）

  runs/
    <node_id>/
      experiment.{ext}             # LLM生成の実験エントリポイント（§7.2）
      {supplementary_files}        # 補助ファイル（§7.2.1、multi_file=True 時）
      stdout.log
      stderr.log
      metrics.json                 # §7.4参照
      artifacts/                   # 図、チェックポイント等
      {build_file}                 # ビルド/依存ファイル（§7.3.3、dependency 設定時）
      build_stdout.log             # ビルドログ（§7.3.2、compiled=True 時）
      install_stdout.log           # 依存インストールログ（§7.3.3、dependency 設定時）

  logs/
    search_log.jsonl               # §17.1参照
    eval_log.jsonl                 # §17.2参照
    ppo_log.jsonl                  # §17.3参照（turn_rewards フィールド含む）
    paper_log.jsonl                # §17.4参照
    agent_llm_log.jsonl            # 全LLM呼び出しログ（プロンプト/レスポンス/tool_calls）
    turn_reward_log.jsonl          # ターンレベル報酬の計算ログ（Phase A）

  checkpoints/
    search_state_step_{N}.json     # 探索状態のチェックポイント
    open_list_step_{N}.json

  outputs/
    best/
      best_node.json               # best_node の全フィールド
      adapter.safetensors          # materialize 済みの全 LoRA ウェイト
      experiment.py                # best の実験スクリプト
      metrics_summary.json         # μ, SE, LCB, CI
      report.json                  # 全Spec + 結果サマリ

  paper/
    paper.md
    paper.bib
    paper.tex                      # optional
    figures/
      architecture.png
      ci_comparison.png
      convergence.png
      search_tree.png
      ablation_table.png

  docs/
    quickstart.md
    workflow.md
    architecture.md
    configuration.md
    reproducibility.md
    api_usage.md
    related_work.md
    modules/
      spec_builder.md
      search_manager.md
      evaluator.md
      adapter_manager.md
      ppo_trainer.md
      related_work_engine.md
      paper_composer.md
      evidence_store.md
```

---

## 15. docs/ 必須内容（詳細要件）

### 15.1 quickstart.md
- install（pip install -e . + 必要な環境変数設定）
- `sera init` → `sera phase0-related-work` → `sera freeze-specs` → `sera research` → `sera generate-paper` → `sera evaluate-paper` の最短手順
- 例（Input-1サンプル：HPC最適化テーマ）

### 15.2 workflow.md
- Phase 0〜8の流れ（Mermaid図）
- 入出力（Spec/ログ/成果物）の流れ

### 15.3 architecture.md
- 外部探索木と内部LoRA系譜木の対応図
- データフロー（spec→run→eval→ppo→paper）
- モジュール依存関係図

### 15.4 configuration.md
- 既定値一覧（§13の表を転記）
- CLI引数対応表
- Phase 1固定ルール（ExecutionSpec）
- 環境変数一覧

### 15.5 reproducibility.md
- seed固定の手順
- revision固定（model/tokenizer/adapter_spec_hash）
- APIログの再現手順
- 実験再実行手順（`sera replay --node-id <id> --seed <seed>`）

### 15.6 api_usage.md
- Scholar優先・フォールバック
- 各APIのエンドポイント/認証/レート制限
- リトライ戦略（exponential backoff、最大5回、初期1s〜最大60s）
- キャッシュの扱い（related_work/results/ に生結果保存、24時間キャッシュ有効）

### 15.7 related_work.md（必須：先行研究整理）
#### 必ず含める系統
- AI-Scientist / AI-Scientist-v2（論文生成・研究ループ・agentic tree search）
- CodeScientist（自律コード実験）
- LoRA/PEFT（LoRA一般）
- Delta-LoRA（delta概念の位置づけ）
- Adapter/Continual Learning（分岐専門化の背景）
- Scholar API関連（公式API非在・第三者API/代替API前提）

#### 先行研究→本システム対応表（必須）
| 先行研究 | 対応Phase | SERA での活用 |
|---------|-----------|-------------|
| AI-Scientist | Phase 7-8 | 論文生成・改善ループの全体設計 |
| AI-Scientist-v2 | Phase 2 | Agentic tree search のアーキテクチャ |
| CodeScientist | Phase 3-4 | 自律実験実行・評価パイプライン |
| LoRA/PEFT | Phase 5-6 | アダプタの基盤技術 |
| Delta-LoRA | Phase 6 | 差分継承の理論的背景 |
| Adapter CL | Phase 5-6 | 分岐専門化・系譜管理 |
| Semantic Scholar API | Phase 0 | 文献収集の第一候補API |

---
