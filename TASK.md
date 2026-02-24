# TASK.md
# Self-Evolving Research Agent（SERA）
# 完全最終要件定義書 v12（具体化+実装手順+分岐仕様明確化+AIDE統合+コンテキスト規則版）

---

## 0. ミッション

任意の研究テーマに対して、

1) Web/API による先行研究調査（Scholar優先）
2) 研究仕様（Spec群）を **動的に確定**
3) Best-First 木構造探索で仮説/実験案を探索
4) 実験を実行しログを収集
5) 統計評価（LCB等）で有望枝を選択
6) PPO（LoRAのみ更新）でエージェントを専門化
7) LoRA差分継承（delta inheritance）で多数分岐を保持
8) AI-Scientist型ワークフローを参考に論文生成
9) 論文評価基準（PaperScoreSpec）に基づき改善ループ

を自律実行する研究エージェントを実装せよ。

### 0.1 エージェントの定義

SERAの「エージェント」とは、**コード生成可能なLLM（ベースモデル）にLoRAアダプタを装着した推論エンジン**である。
このエージェントは以下の行為を遂行する：

| 行為 | Phase | 具体的出力 |
|------|-------|-----------|
| 仮説生成 | Phase 2 | 自然言語の仮説文 + 実験条件の差分JSON |
| 実験コード生成 | Phase 3 | `experiment.*`（ProblemSpec.languageに基づく多言語対応スクリプト。Python/R/Julia/Go/C++/bash等） |
| 結果分析 | Phase 4 | metrics.jsonの解釈、次ステップ提案 |
| 論文執筆 | Phase 7 | Markdown/LaTeX形式の論文草稿 |
| 自己改善 | Phase 5 | PPOによるLoRA更新を通じた方策改善 |

エージェントのLLM呼び出しは全て `AgentLLM` クラスを経由する（§21参照）。
ベースモデルは変更せず、**LoRAアダプタのみが探索木のノードごとに分岐・専門化**される。

---

## 1. 設計原則（Non-negotiable）

1. **テーマ非依存**：学習は研究対象そのものではなく、実行エージェント内部（方策/生成能力）に適用される。
2. **二重木構造**：外部探索木（仮説/実験案）と内部LoRA系譜木（専門化）を同期して管理する。
3. **LoRA差分継承が標準**：ノードは親との差分Δのみ保存し、必要時に累積復元する。
4. **統計的妥当性必須**：反復評価＋SE＋LCB（またはUCB/CI）を基本にする。
5. **再現性保証**：seed、モデルrevision、APIクエリと取得日時、実行コマンド、環境情報を必ず保存。
6. **実行前固定**：Phase 2/3/4/5 に関わる既定値（探索/評価/学習/剪定/停止/予算）は Phase 1 で **ExecutionSpecとして固定**し、以後変更不可。
7. **既定値＋引数上書き**：既定値を持つが、CLI引数またはSpecで上書き可能。上書きは Phase 1 までに確定し、以降は固定。
8. **docs/必須**：Quickstart/Workflow/Architecture/各モジュール詳細/先行研究整理を `docs/` に含める。
9. **変数可変性の三層分離**：システム内の変数は以下の3層に厳密に分離される。層の境界を越えた変更は禁止。

| 層 | 所属Spec | Phase 2以降の可変性 | 例 |
|---|---------|-------------------|---|
| **固定層（Frozen）** | ExecutionSpec | **完全不変** | lr, clip_range, repeats, lcb_coef, lambda_cost, beta, max_nodes, rank, alpha |
| **操作層（Manipulated）** | ProblemSpec.manipulated_variables | **分岐生成で変更可能**（ホワイトリスト） | 実験のlearning_rate, batch_size, method, data_augmentation等 |
| **導出層（Derived）** | 実行時自動計算 | **自動決定**（手動変更不可） | priority, mu, se, lcb, feasible, reward |

> **重要**: 固定層と操作層の区別を厳守すること。ExecutionSpec.learning.lr（PPOの学習率）は固定層であり、ProblemSpec.manipulated_variables に定義された learning_rate（実験対象モデルの学習率）は操作層である。名前が類似していても層が異なる。

---

## 2. 全体フェーズ

```text
Phase 0: 先行研究フェーズ（Web/API）
Phase 1: Spec確定（PaperSpec/PaperScoreSpec/TeacherPaperSet + ExecutionSpec固定）
Phase 2: 探索木生成（Best-First）
Phase 3: 実験実行（Executor）
Phase 4: 統計評価（Evaluator: repeats, SE, LCB, sequential eval）
Phase 5: 学習（PPO: LoRA-only）
Phase 6: 系譜管理・剪定（delta lineage + squash + pruning）
Phase 7: 論文生成（AI-Scientist参考）
Phase 8: 論文評価・改善（PaperScoreSpec, loop）
```

### 2.1 Phase間データフロー（具体）

```text
Input-1 (YAML)
  │
  ▼
Phase 0 ── LLM がクエリ生成 ──► Scholar API/Semantic Scholar/CrossRef/arXiv
  │                                     │
  │◄── papers[], clusters[], baselines ─┘
  │
  ▼
Phase 1 ── LLM が Spec群を草案 → ユーザ承認 or 自動確定
  │         出力: specs/*.yaml（9ファイル）
  │         ExecutionSpec は SHA-256 ハッシュを計算し specs/execution_spec.yaml.lock に保存
  │
  ▼
Phase 2 ── SearchManager が open_list から最高優先度ノードを pop
  │         LLM（AgentLLM）が子ノード候補を生成（仮説+実験条件 JSON）
  │         子ノードを open_list に追加
  │
  ▼
Phase 3 ── Executor がノードの実験コードを LLM 生成 → サンドボックス内実行
  │         出力: runs/<node_id>/metrics.json + stdout.log + stderr.log
  │
  ▼
Phase 4 ── Evaluator が metrics.json を集約、μ/SE/LCB を計算
  │         逐次評価: 初回1回 → Top-k なら repeats まで追加
  │
  ▼
Phase 5 ── PPOTrainer がロールアウト収集 → LoRA-only 更新 → Δ保存
  │         条件: 評価済みノードが ppo_trigger_interval 個溜まるごと
  │
  ▼
Phase 6 ── LineageManager が squash/prune を実行
  │         ループ: Phase 2 へ戻る（終了条件まで）
  │
  ▼
Phase 7 ── PaperComposer が EvidenceStore から証拠収集
  │         Step 1: ログ要約 → Step 2: 図生成・集約
  │         Step 3: 自動引用検索（Semantic Scholar、最大20ラウンド）
  │         Step 4: VLM 図記述生成（VLM有効時）
  │         Step 5: 論文1パス生成 + ライティング内反省ループ（最大3回、VLMレビュー含む）
  │         Step 6: 統合・最終出力
  │
  ▼
Phase 8 ── PaperEvaluator が PaperScoreSpec で採点
  │         アンサンブルレビュー（num_reviews_ensemble 件の独立レビュー）
  │         各レビュアーに反省ループ（num_reviewer_reflections 回）
  │         メタレビュー（Area Chair 集約）→ 改善指示 → Phase 7 再実行
           最大 paper_revision_limit 回ループ
```

---

## 3. 入力仕様（2種類）

### 3.1 Input-1（簡素版）
ユーザが与える最小入力。Input-1受領後、Phase 0 を経て Phase 1 で Input-2 を確定する。

```yaml
version: 1
data:
  description: ""     # データの性質（ログ、CSV、コード、PDF群…）
  location: ""        # パス/URI/リポジトリ
  format: ""          # "csv" | "json" | "parquet" | "code" | "pdf" | "mixed"
  size_hint: ""       # "small(<1GB)" | "medium(1-100GB)" | "large(>100GB)"
domain:
  field: ""           # "HPC", "materials", "software", "NLP", "CV", etc.
  subfield: ""        # より具体的な分野（"compiler optimization", "protein folding"等）
task:
  brief: ""           # 研究タスクの短い説明（1-3文）
  type: ""            # "optimization" | "prediction" | "generation" | "analysis" | "comparison"
goal:
  objective: ""       # 目標（"minimize runtime", "maximize accuracy", "minimize variance"...）
  direction: ""       # "minimize" | "maximize"
  baseline: ""        # 既知のベースライン値（あれば）
constraints:          # 制約条件（0個以上）
  - name: ""
    type: ""          # "ge" (>=) | "le" (<=) | "eq" (==) | "bool" (true/false)
    threshold: null
notes: ""             # 任意
```

### 3.2 Input-2（完全版）
Input-1 + Phase 0結果 + 実行前固定値を含む機械実行可能な仕様。以下を含む：

- `RelatedWorkSpec`（§4.4.1）
- `PaperSpec`（§4.4.2）
- `PaperScoreSpec`（§4.4.3）
- `TeacherPaperSet`（§4.4.4）
- `ProblemSpec`（§5.5）
- `ModelSpec`（§5.3）
- `ResourceSpec`（§5.6）
- `PlanSpec`（§5.7）
- `ExecutionSpec`（§5.4）

---

## 4. Phase 0：先行研究フェーズ（Web/API、Scholar優先）

### 4.1 目的
- 研究空間の把握・ベースライン候補抽出
- 論文フォーマット（PaperSpec）と評価基準（PaperScoreSpec）の **動的確定**
- 教師論文（TeacherPaperSet）確定（構造模倣/品質基準参照）

### 4.2 API仕様（具体）

#### 4.2.1 API優先順位
```text
1. Semantic Scholar API（無料、認証キーで上限緩和）
2. CrossRef API（DOIベース、メタデータ豊富）
3. arXiv API（プレプリント、全文取得可能）
4. Web検索（SerpAPI/Tavily経由、最終フォールバック）
```

> 注：Google Scholar は公式APIを提供していないため、Semantic Scholar を第一候補とする。
> SerpAPI の Google Scholar エンドポイント使用は ResourceSpec.api_keys.serpapi が設定されている場合のみ。

#### 4.2.2 各APIの具体エンドポイントと使用方法

**Semantic Scholar API**:
- エンドポイント: `https://api.semanticscholar.org/graph/v1/paper/search`
- 認証: `x-api-key` ヘッダ（任意、未設定時は 100 req/5min）
- パラメータ: `query`, `year`, `fieldsOfStudy`, `limit`, `offset`, `fields`
- 取得フィールド: `paperId,title,abstract,year,citationCount,authors,venue,externalIds,url`
- 引用グラフ: `/paper/{paper_id}/citations`, `/paper/{paper_id}/references`
- レート制限: キーあり 1 req/sec、キーなし 100 req/5min
- リトライ: HTTP 429 → exponential backoff（初期1s、最大60s、最大5回）

**CrossRef API**:
- エンドポイント: `https://api.crossref.org/works`
- 認証: `mailto` パラメータ（polite pool 用、ResourceSpec.contact_email）
- パラメータ: `query`, `filter=from-pub-date:{year}`, `rows`, `offset`, `sort=relevance`
- 取得フィールド: `DOI,title,abstract,author,published-print,is-referenced-by-count,container-title`

**arXiv API**:
- エンドポイント: `http://export.arxiv.org/api/query`
- 認証: 不要
- パラメータ: `search_query`, `start`, `max_results`, `sortBy=relevance`
- レート制限: 3秒間隔（公式ガイドライン）
- PDF取得: `https://arxiv.org/pdf/{arxiv_id}.pdf`（TeacherPaperSet用）

**Web検索フォールバック（SerpAPI）**:
- エンドポイント: `https://serpapi.com/search`（engine=google_scholar）
- 認証: `api_key` パラメータ
- 使用条件: 上記3つのAPIで `top_k_papers` を満たせなかった場合のみ

#### 4.2.3 クエリ構築アルゴリズム
```python
def build_queries(input1: Input1) -> list[str]:
    """Input-1からAPI検索クエリを生成する。LLMを使用。"""
    # 1. LLM に Input-1 の brief/field/objective を与え、3-5個の検索クエリを生成させる
    # 2. 各クエリは以下の形式：
    #    - メインクエリ: "{task.brief} {domain.field}"
    #    - 手法クエリ:   "{推定手法名} {domain.field}"
    #    - ベースラインクエリ: "state-of-the-art {goal.objective} {domain.field}"
    # 3. 年フィルタ: current_year - recent_years_bias 以降を優先
    # 4. 全クエリを queries.jsonl に保存
    pass
```

### 4.3 Phase 0 既定値（Phase 1で固定、引数で変更可能）
- `top_k_papers = 10`
- `recent_years_bias = 5`（過去5年優先）
- `citation_graph_depth = 1`（引用の引用までは追わない）
- `teacher_papers = 5`
- `ranking_weight = 0.6*citation_norm + 0.4*relevance_score`
  - `citation_norm = log(1 + citations) / max(log(1 + citations))` で [0,1] 正規化
  - `relevance_score` = Semantic Scholar API の score、またはLLMによる0-1スコアリング

### 4.4 Phase 0 出力（必須）

#### 4.4.1 RelatedWorkSpec
```yaml
related_work_spec:
  papers:
    - paper_id: "ss_abc123"           # Semantic Scholar ID（主キー）
      title: ""
      authors: ["Last, First", ...]
      year: 2024
      venue: "NeurIPS"
      abstract: ""
      citation_count: 150
      url: "https://..."
      doi: "10.1234/..."
      arxiv_id: "2401.12345"
      source_api: "semantic_scholar"   # 取得元API
      relevance_score: 0.85           # 0-1、LLM算出
      retrieval_query: "query string"  # 取得時のクエリ
      retrieved_at: "2026-02-21T10:30:00Z"

  clusters:                            # LLMによるクラスタリング
    - name: "tree_search_methods"
      description: "木構造探索による研究自動化手法"
      paper_ids: ["ss_abc123", ...]
      keywords: ["MCTS", "best-first", "beam search"]
    - name: "lora_adaptation"
      description: "LoRAベースのモデル適応手法"
      paper_ids: [...]
      keywords: ["LoRA", "PEFT", "adapter"]

  baseline_candidates:
    - name: "AI-Scientist-v2"
      paper_id: "ss_xyz789"
      reported_metric: {"name": "paper_score", "value": 7.2, "scale": "1-10"}
      method_summary: "Agentic tree search + LLM paper generation"
    # ...

  common_metrics:
    - name: "paper_score"
      description: "LLM-as-judge による論文品質スコア"
      scale: "1-10"
      higher_is_better: true
    # ...

  common_datasets:
    - name: "NeurIPS 2024 papers"
      description: "NeurIPS 2024 採択論文セット"
      url: ""
      size: ""
    # ...

  open_problems:
    - description: "探索木の深さとLoRA品質の関係が未解明"
      related_paper_ids: ["ss_abc123"]
      severity: "medium"  # "low" | "medium" | "high"
    # ...
```

#### 4.4.2 PaperSpec（動的確定）
```yaml
paper_spec:
  format: "arxiv"                     # "arxiv" | "conference_style" | "tech_report"
  max_pages: 12                       # 付録除く
  sections_required:
    - key: "abstract"
      max_words: 300
    - key: "introduction"
      must_contain: ["motivation", "contribution_list", "paper_organization"]
    - key: "related_work"
      must_contain: ["comparison_table"]
    - key: "method"
      must_contain: ["algorithm_pseudocode", "architecture_diagram"]
    - key: "experiments"
      must_contain: ["setup", "baselines", "main_results", "ablation"]
    - key: "results"
      must_contain: ["ci_table", "lcb_comparison"]
    - key: "discussion"
      must_contain: ["limitations", "future_work"]
    - key: "conclusion"
    - key: "references"

  figures_required:
    - type: "architecture_diagram"
      description: "システム全体のアーキテクチャ図"
    - type: "search_tree_visualization"
      description: "探索木の展開過程（上位5ノード表示）"
    - type: "ci_bar_chart"
      description: "主指標の信頼区間付き棒グラフ（全ベースライン比較）"
    - type: "convergence_curve"
      description: "探索ステップ vs 最良LCBの推移"
    - type: "ablation_table"
      description: "アブレーションスタディの結果表"

  stats_reporting:
    require_repeats: true              # 反復実験必須
    require_ci: true                   # 信頼区間必須
    ci_level: 0.95                     # 95% CI
    require_effect_size: false         # Cohen's d 等（任意）
    decimal_places: 3

  reproducibility_requirements:
    require_seed: true
    require_model_revision: true
    require_environment_info: true     # Python版、GPU型番、CUDAバージョン等
    require_command_log: true          # 実行コマンド全記録
    require_data_hash: true            # データセットのSHA-256
```

#### 4.4.3 PaperScoreSpec（動的確定）
```yaml
paper_score_spec:
  evaluator: "llm_as_judge"           # "llm_as_judge" | "rule_based" | "hybrid"
  evaluator_model: "same_as_base"     # 論文評価に使うモデル（LoRA無し）
  max_score: 10
  criteria:
    - name: "statistical_rigor"
      description: "反復実験、CI/LCB報告、適切な検定の有無"
      weight: 0.20
      rubric:
        10: "全指標に95%CI、効果量、適切な検定あり"
        7: "主指標にCI報告あり、検定なし"
        4: "反復実験あるが CI 未報告"
        1: "単一試行のみ"
    - name: "baseline_coverage"
      description: "適切なベースラインとの比較の網羅性"
      weight: 0.15
      rubric:
        10: "3つ以上の最新ベースラインと公正な比較"
        7: "2つのベースライン"
        4: "1つのベースライン"
        1: "ベースライン比較なし"
    - name: "ablation_quality"
      description: "アブレーションスタディの質と網羅性"
      weight: 0.15
      rubric:
        10: "主要コンポーネント全てを無効化した体系的アブレーション"
        7: "部分的アブレーション"
        4: "1つのみ"
        1: "アブレーションなし"
    - name: "reproducibility"
      description: "再現に必要な情報の完全性"
      weight: 0.15
      rubric:
        10: "seed/環境/コマンド/データ全て記載、コード公開"
        7: "主要情報は記載"
        4: "部分的"
        1: "再現不可能"
    - name: "contribution_clarity"
      description: "貢献の明確さと新規性の主張"
      weight: 0.15
      rubric:
        10: "貢献が明確にリスト化、先行研究との差異が具体的"
        7: "貢献は明確だが差異が曖昧"
        4: "貢献が不明確"
        1: "貢献なし/既存手法の再実装のみ"
    - name: "writing_quality"
      description: "論文の読みやすさ、構成、図表の質"
      weight: 0.10
      rubric:
        10: "明快な構成、適切な図表、読みやすい英語"
        7: "構成は良いが一部不明瞭"
        4: "構成に問題あり"
        1: "読解困難"
    - name: "limitations_honesty"
      description: "限界の誠実な記述"
      weight: 0.10
      rubric:
        10: "主要な限界を全て記述し、影響範囲を定量化"
        7: "主要な限界を記述"
        4: "形式的な限界記述のみ"
        1: "限界記述なし"

  passing_score: 6.0                   # この値以上で合格
  paper_revision_limit: 3              # 改善ループ最大回数

  # --- AI-Scientist-v2 スタイル拡張 ---
  ensemble:
    num_reviews_ensemble: 3            # 独立レビュー数（1=アンサンブルなし、3推奨）
    num_reviewer_reflections: 2        # 各レビュアーの反省ループ回数
    num_fs_examples: 2                 # Few-shot査読例の数（0=不使用）
    bias_mode: "critical"              # "critical"（既定）| "generous"
    meta_review: true                  # アンサンブル時にArea Chairメタレビューを生成するか
    temperature: 0.75                  # レビュー生成時のtemperature

  few_shot_reviews: []                 # Phase 0/1 で収集した査読例
    # 形式: [{paper_excerpt: str, review_json: str}]
    # TeacherPaperSet から査読済み論文がある場合、その査読をここに格納
    # 空の場合、汎用的な査読例テンプレートを使用
```

#### 4.4.4 TeacherPaperSet
```yaml
teacher_paper_set:
  selection_criteria: "relevance_score >= 0.8 AND citation_count >= 50"
  teacher_papers:
    - paper_id: "ss_abc123"
      title: ""
      role: "structure_reference"      # "structure_reference" | "method_reference" | "both"
      sections:                        # 実際の章構成を記録
        - "Abstract"
        - "1 Introduction"
        - "2 Related Work"
        # ...
      figure_count: 6
      table_count: 4
      experiment_style: "multi_baseline_with_ablation"
      stats_format: "mean_pm_std"      # "mean_pm_std" | "ci_bracket" | "lcb_table"
  structure_summary:
    avg_sections: 8
    avg_figures: 5
    avg_tables: 3
    common_experiment_pattern: "setup → baselines → main_results → ablation → analysis"
    common_stats_format: "mean_pm_std"
```

### 4.5 再現性（Phase 0必須）
全API呼び出しを `related_work/queries.jsonl` に以下の形式で保存：
```json
{
  "query_id": "q001",
  "api": "semantic_scholar",
  "endpoint": "https://api.semanticscholar.org/graph/v1/paper/search",
  "params": {"query": "...", "year": "2021-", "limit": 20},
  "timestamp_utc": "2026-02-21T10:30:00Z",
  "http_status": 200,
  "result_count": 20,
  "paper_ids_returned": ["ss_abc123", "ss_def456", ...],
  "retry_count": 0,
  "error": null
}
```

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
   b. PlanSpec（§5.7）
3. ユーザが ModelSpec（§5.3）と ResourceSpec（§5.6）を確認・修正
   - CLIモード: 生成されたYAMLをエディタで開き、保存で確定
   - 自動モード（--auto）: LLM生成のまま確定
4. ExecutionSpec（§5.4）を CLI 引数 + 既定値から構築
5. 全Spec を specs/ に保存
6. ExecutionSpec の SHA-256 ハッシュを specs/execution_spec.yaml.lock に記録
7. 以後、ExecutionSpec のロード時にハッシュを検証（不一致は致命的エラー）
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

---

## 6. Phase 2：探索木生成（AIDE参考 Best-First）

> **設計参考**: AIDE (AI-Driven Exploration in the Space of Code, Weco AI, 2025) の探索木アーキテクチャを参考とする。AIDEはML工学をコード空間の木探索として定式化し、Drafting / Debugging / Improving の3オペレータで解空間を探索する。SERAはこの3オペレータ設計を採用しつつ、統計評価（LCB）・LoRA差分継承・PPO学習を追加した拡張版として設計する。

### 6.1 AIDE との対応関係

| AIDE の概念 | SERA での対応 | 拡張点 |
|------------|-------------|--------|
| ノード = Pythonスクリプト + メトリクス | ノード = 仮説 + experiment_config + メトリクス + LoRA参照 | LoRA系譜の追加 |
| Drafting（新規解の起草） | `draft` オペレータ（ルート生成 + 新アプローチ導入） | 先行研究ベースの初期化 |
| Debugging（エラー修復） | `debug` オペレータ（実験失敗時の修復、深度制限付き） | AIDE同様の深度制限 |
| Improving（原子的改善） | `improve` オペレータ（単一変数の測定可能な変更） | LCBベースの統計的判断 |
| 最良の有効解を選択 | LCBベースの優先度関数 | ε-constraint + 多指標 |
| 実行して実メトリクスで評価 | 反復実行 + μ/SE/LCB | 逐次評価の追加 |

### 6.2 ノード定義（必須）
各探索ノード `node_id` は以下を持つ：

```python
@dataclass
class SearchNode:
    node_id: str                    # UUID v4
    parent_id: str | None           # ルートは None
    depth: int
    created_at: str                 # ISO 8601 UTC

    # 外部状態（AIDE: 各ノードが1つの完全な解を表す）
    hypothesis: str                 # 自然言語の仮説（例："learning rate を 1e-3 にすると精度向上"）
    experiment_config: dict         # 操作変数の具体値 {"lr": 1e-3, "batch_size": 32, ...}
    experiment_code: str | None     # LLM生成の実験コード（Phase 3 で設定）
    branching_op: str               # 適用されたオペレータ名: "draft" | "debug" | "improve"
    rationale: str                  # LLM が生成した分岐理由

    # 内部状態参照
    adapter_node_id: str | None     # LoRA系譜ノードID（Phase 5 後に設定）

    # 評価統計
    eval_runs: int                  # 実行済み反復数
    metrics_raw: list[dict]         # 各反復の metrics.json
    mu: float | None                # primary の平均
    se: float | None                # 標準誤差
    lcb: float | None               # LCB = mu - c*se

    # コスト
    total_cost: float               # 累計コスト（秒 or GPU分）
    wall_time_sec: float            # 実壁時間

    # 探索制御
    priority: float | None          # 計算済み優先度
    status: str                     # "pending" | "running" | "evaluated" | "failed" | "pruned" | "expanded"
    children_ids: list[str]         # 子ノードID群
    feasible: bool                  # ε-constraint を満たすか
    debug_depth: int                # debug 連鎖の深さ（AIDE参考: 深度制限用）
    error_message: str | None       # status="failed" 時のエラー内容
```

### 6.3 選択方針（AIDE参考）

AIDEは「最高性能の有効解をベースに次の改善を行う」方針を取る。SERAではこれをLCBベースの優先度関数として一般化する。

```python
def compute_priority(node: SearchNode, exec_spec: ExecutionSpec) -> float:
    """
    AIDE参考の選択方針:
    - AIDEは最高性能の有効スクリプトを選択する（シンプルな greedy）
    - SERAはこれを LCB + コスト + 探索ボーナスで拡張する
    """
    if not node.feasible:
        return -float('inf')  # 制約違反は最低優先度
    if node.lcb is None:
        return float('inf')   # 未評価ノードは最優先（探索促進）

    lcb_primary = node.lcb
    cost = node.total_cost
    bonus = compute_exploration_bonus(node)

    return lcb_primary - exec_spec.search.lambda_cost * cost + exec_spec.search.beta_exploration * bonus

def compute_exploration_bonus(node: SearchNode) -> float:
    """未探索領域の優遇。最小実装は 0。"""
    # 拡張実装: 1.0 / sqrt(node.eval_runs + 1)（UCB1風）
    # または: 同一 branching_op の使用回数の逆数
    return 0.0
```

### 6.4 ノード選択アルゴリズム（必須）

```python
def select_next_node(open_list, all_nodes, exec_spec) -> tuple[SearchNode, str]:
    """
    AIDE参考: 状態に応じて適用するオペレータを自動決定する。

    選択アルゴリズム:
    1. open_list が空 → 終了
    2. 未評価ノード（status="pending"）があれば最優先で実行
    3. 失敗ノード（status="failed", debug_depth < max_debug_depth）があれば debug 対象として選択
    4. それ以外 → 最高 priority の評価済みノードを improve 対象として選択
    5. 評価済みノードが一定数に達し、全体の多様性が不足 → draft で新アプローチ導入

    Returns:
        (selected_node, operator_name)  # operator_name: "draft" | "debug" | "improve"
    """
    # Step 1: 未評価ノードの処理
    pending = [n for n in open_list if n.status == "pending"]
    if pending:
        return (max(pending, key=lambda n: n.priority or float('inf')), "evaluate")

    # Step 2: 失敗ノードの修復（AIDE Debugging）
    failed = [n for n in all_nodes.values()
              if n.status == "failed"
              and n.debug_depth < exec_spec.search.max_debug_depth
              and n.node_id not in closed_set]
    if failed:
        # 最も浅い（修復しやすい）失敗ノードを選択
        target = min(failed, key=lambda n: n.debug_depth)
        return (target, "debug")

    # Step 3: 多様性不足時の新規起草（AIDE Drafting）
    evaluated = [n for n in all_nodes.values() if n.status == "evaluated"]
    unique_methods = len({n.experiment_config.get("method", "") for n in evaluated})
    if unique_methods < exec_spec.search.min_diverse_methods and len(evaluated) >= exec_spec.search.draft_trigger_after:
        return (None, "draft")  # 親なしの新規ノード

    # Step 4: 最良ノードの改善（AIDE Improving）
    best = max(evaluated, key=lambda n: n.lcb or float('-inf'), default=None)
    if best:
        return (best, "improve")

    return (None, "draft")  # フォールバック
```

### 6.5 3オペレータ設計（AIDE参考：必須）

AIDEの Drafting / Debugging / Improving を SERA に適応した3オペレータを定義する。
§6.4 の `select_next_node()` がオペレータを自動選択し、以下の各オペレータが子ノードを生成する。

#### 6.5.1 Draft オペレータ（新規解の起草）

```text
目的: 既存の探索木に存在しない、まったく新しいアプローチを導入する

発動条件（§6.4 で自動判定）:
  - ルート初期化時（tree_ops.draft() の初回呼び出し）
  - 探索中に多様性が不足した場合（unique_methods < min_diverse_methods）
  - 全ノードが制約違反の場合（feasible なノードがゼロ）

特徴:
  - 親ノードなし（または仮想ルートが親）
  - experiment_config は先行研究・LLMの自由提案から構築
  - AIDEの「完全に新しいスクリプトを起草する」に対応

出力: 1〜n 個の新規 SearchNode（parent_id=None, depth=0, branching_op="draft"）
```

```python
def draft(self, specs: AllSpecs, agent_llm: AgentLLM, rng, n: int) -> list[SearchNode]:
    """
    ルート生成時:
      1. baseline_candidates → ベースライン再現ノード
      2. open_problems → 課題解決ノード
      3. LLM自由提案 → 新規アプローチノード
      配分: 各 n//3（端数は自由提案に割当）

    探索中の再 draft 時:
      1. 既存の全ノードの hypothesis 一覧を LLM に提示
      2. 「これまでと異なるアプローチ」を明示的に要求
      3. 1〜2 個のみ生成（大量生成は探索の焦点を散らす）
    """
    pass
```

#### 6.5.2 Debug オペレータ（エラー修復：AIDE最重要の差別化要素）

```text
目的: 実験が失敗（exit_code != 0）したノードのコードを修復し、有効な結果を得る

発動条件:
  - node.status == "failed" かつ node.debug_depth < max_debug_depth

特徴（AIDE準拠）:
  - 親ノード = 失敗したノード自身（修復チェーン）
  - experiment_config は変更しない（コードのみ修正）
  - debug_depth をインクリメント（深度制限で無限修復ループを防止）
  - エラーメッセージ（stderr）をLLMに提示して修正案を生成
  - 原子的修正: 1つのエラーに対して1つの修正のみ

深度制限（AIDE参考）:
  - max_debug_depth（既定3）を超えたら修復を諦め、そのノードは "failed" のまま閉じる
  - 理由: 根本的にアプローチが間違っている場合、修復より draft（新規起草）が効率的
```

```python
def debug(self, failed_node: SearchNode, agent_llm: AgentLLM) -> SearchNode:
    """
    1. failed_node の experiment_code と error_message (stderr) を取得
    2. LLM に以下のプロンプトを与え、修正コードを生成:

    プロンプト:
    ---
    以下の実験コードがエラーで失敗しました。エラーを修正してください。
    修正は最小限にし、実験の意図（仮説）は変更しないでください。

    ## 実験コード
    {failed_node.experiment_code}

    ## エラーメッセージ
    {failed_node.error_message}

    ## 修正方針
    - import エラー → 正しいモジュール名に修正
    - 型エラー → 型変換を追加
    - 形状エラー → テンソル形状を修正
    - ファイル不在 → パスを修正、またはデータ生成コードを追加
    - それ以外 → 最小限の修正で動作するようにする

    ## 出力: 修正後の完全なコード（差分ではなく全体）
    ---

    3. 修正コードで新ノードを構築:
       - parent_id = failed_node.node_id
       - experiment_config = failed_node.experiment_config（変更なし）
       - experiment_code = LLM生成の修正コード
       - hypothesis = failed_node.hypothesis（変更なし）
       - branching_op = "debug"
       - debug_depth = failed_node.debug_depth + 1
       - status = "pending"
    """
    pass
```

#### 6.5.3 Improve オペレータ（原子的改善：AIDE核心）

```text
目的: 動作する最良のノードに対して、単一の測定可能な変更を加えて改善する

発動条件:
  - 最高LCBの有効ノードが選択された場合（通常の探索ステップ）

特徴（AIDE準拠）:
  - 親ノード = 最良（または選択された）の有効ノード
  - 変更は「原子的」= 1つの操作変数のみ変更、または意味的に1つの変更
  - 変更の影響が測定可能であること（CI で比較できる大きさ）
  - AIDEの「each change's impact is directly measurable」に対応

原子的変更の定義:
  - experiment_config の diff が1キーのみ（推奨）
  - 2キー以上変更する場合は、rationale で「なぜ同時変更が必要か」を明示
  - branch_factor 個の子を生成する場合、各子は異なる変数を変更する（多様性確保）
```

```python
def improve(self, parent: SearchNode, specs: AllSpecs, agent_llm: AgentLLM,
            all_nodes: dict, rng, n_children: int) -> list[SearchNode]:
    """
    1. parent の結果と全兄弟の結果を収集
    2. LLM に「原子的改善」を要求するプロンプトを構築
    3. n_children 個の改善案を生成（各案は1変数のみ変更を推奨）
    4. validate_experiment_config でバリデーション
    5. 原子性チェック: diff が2キー以上なら警告ログ（棄却はしない）

    プロンプト:
    ---
    あなたは研究アシスタントです。以下の実験結果を踏まえ、
    {n_children}個の **原子的な** 改善案を提案してください。

    「原子的」とは: 1つの変数のみを変更し、その変更の効果を測定可能にすること。

    ## 研究目的
    {problem_spec.objective.description}

    ## 操作可能な変数（これ以外の変数は変更禁止）
    {manipulated_variables}

    ## 現在の最良実験（改善ベース）
    仮説: {parent.hypothesis}
    設定: {parent.experiment_config}
    結果: {parent.mu} ± {parent.se} (LCB: {parent.lcb})
    制約: {constraint_status}

    ## これまでの兄弟・子ノードの結果（CI付き）
    {sibling_summaries}

    ## 最良ノードとの差
    最良LCB: {best_node.lcb}

    ## 統計的ガイダンス
    - 各提案は1つの変数のみ変更せよ（原子的変更）
    - CI が重複する変更は無意味。SE の 2倍以上の変化を狙え
    - {n_children}個の提案はそれぞれ異なる変数を変更せよ

    ## 出力形式（JSON配列）
    [
      {
        "hypothesis": "learning_rate を 5e-4 に下げると過学習が抑制される",
        "experiment_config": {"learning_rate": 5e-4},  ← 1変数のみ変更
        "rationale": "現在の lr=1e-3 で train_loss < val_loss の乖離が大きい",
        "changed_variable": "learning_rate"  ← 変更した変数名を明示
      },
      ...
    ]
    ---
    """
    pass
```

### 6.6 変更可能変数の境界

分岐生成（improve/draft）で変更できるのは **ProblemSpec.manipulated_variables に定義された変数のみ** である。

```text
■ 変更可能（操作層）: ProblemSpec.manipulated_variables に列挙された変数
  例: 実験対象の learning_rate, batch_size, method, optimizer, data_split

■ 変更禁止（固定層）: ExecutionSpec に属する全パラメータ
  例: PPOのlr, clip_range, repeats, lcb_coef, lambda_cost, beta, rank, alpha

■ 変更禁止（固定層）: ModelSpec, ResourceSpec に属する全パラメータ
  例: base_model.id, adapter_spec.rank, executor_type, timeout

■ LoRA形状パラメータの扱い:
  - rank, alpha, target_modules は ModelSpec.adapter_spec に属する（固定層）
  - Phase 1 で確定後、全ノードで同一のLoRA形状を使用する
  - 形状が変わると delta inheritance の加減算が不可能になるため、変更は原理的に禁止
  - adapter_spec_hash が異なる分岐は生成してはならない
```

> **原則**: experiment_config に含めてよいキーは `ProblemSpec.manipulated_variables[].name` と完全一致するもののみ。それ以外のキーはバリデーションエラーとして棄却する。

#### 6.6.1 experiment_config のバリデーション

```python
def validate_experiment_config(config: dict, problem_spec: ProblemSpecModel) -> tuple[bool, list[str]]:
    """
    分岐生成の出力をホワイトリスト検証する。

    Returns:
        (is_valid, error_messages)
    """
    allowed_keys = {v.name for v in problem_spec.manipulated_variables}
    errors = []

    # 1. ホワイトリスト検証: 未知のキーを拒否
    unknown_keys = set(config.keys()) - allowed_keys
    if unknown_keys:
        errors.append(f"Unknown keys (not in manipulated_variables): {unknown_keys}")

    # 2. 型・範囲検証: 各変数の制約を満たすか
    for var in problem_spec.manipulated_variables:
        if var.name not in config:
            continue  # 親からの暗黙継承（変更なし）
        value = config[var.name]
        if var.type == "float":
            if not (var.range[0] <= value <= var.range[1]):
                errors.append(f"{var.name}={value} out of range {var.range}")
        elif var.type == "int":
            if not (var.range[0] <= value <= var.range[1]):
                errors.append(f"{var.name}={value} out of range {var.range}")
        elif var.type == "categorical":
            if value not in var.choices:
                errors.append(f"{var.name}='{value}' not in choices {var.choices}")

    return (len(errors) == 0, errors)
```

### 6.7 LoRA系譜との接続

分岐生成は **外部探索木（仮説分岐）のみ** を扱う。LoRA系譜は以下のルールで **自動的に** 決定される：

```text
■ 分岐生成時（Phase 2）:
  - 子ノード.adapter_node_id = None（未割当）
  - 子ノードは親ノードの LoRA を暗黙的に継承する（参照のみ）

■ 実験実行時（Phase 3）:
  - 子ノードの実験は親の materialize 済み LoRA で AgentLLM を設定して実行

■ PPO更新時（Phase 5）:
  - PPO更新対象のノード群の「最良ノード」の親LoRAをベースに更新
  - 更新結果のΔを新しい adapter_node として lineage に登録
  - 当該ノード.adapter_node_id を新 adapter_node に設定

■ PPO未実行のノード:
  - adapter_node_id = 親の adapter_node_id のまま（差分Δなし = 同一アダプタ）

つまり:
  - 「LoRAを親から継承するか」 → 常に継承する（選択の余地なし）
  - 「新規分岐するか」 → PPO更新が走ったノードのみ自動的に新分岐
  - 「既存アダプタを再利用するか」 → PPO未実行ノードは親と同一アダプタ

分岐生成が「仮説 × 専門化方策」を決める必要はない。
仮説分岐は Phase 2、方策専門化は Phase 5 の責務であり、完全に分離される。
```

### 6.8 統計的コンテキストの提示

分岐生成（improve）自体は統計的判断を行わない（それは §6.3 の優先度関数と §10 の剪定の責務）。
ただし、LLMが統計的に有意義な提案を行えるよう、プロンプトに以下の統計的コンテキストを含める：

```text
■ 必須提示項目:
  - 親ノードの μ ± SE（95% CI）
  - 兄弟ノードの μ ± SE（95% CI）
  - 親と最良ノードの LCB 差（Δ_LCB）
  - 親ノードの制約充足状況

■ LLMへの指示（プロンプトに明記）:
  - 「CI が重複する兄弟との微小な変更は避け、統計的に区別可能な変更を提案せよ」
  - 「primary の改善幅が SE の 2 倍未満の変更は無意味である」

■ 統計的判断の責務分離:
  - LCB差の閾値判定 → 分岐生成の責務ではない。剪定（§10）と優先度関数（§6.3）が担う
  - 有意差検定 → 分岐生成は実施しない。プロンプトにCIを提示し、LLMの自律判断に委ねる
  - 逐次評価との関係 → 分岐生成は status="evaluated" のノードのみ参照する。
    eval_runs < repeats のノードも暫定 μ/SE を持つため参照対象に含む。
    ただし eval_runs=1 のノードは SE=inf であり、LLMへの提示時に
    「(暫定: n=1)」と注記する
```

#### 6.8.1 兄弟ノードコンテキストの構成規則

improve オペレータ（§6.5.3）がLLMに提示する「兄弟ノード情報」の範囲と粒度を定義する。

```text
■ 参照対象の範囲:
  1. 親ノードの直接の子（同一 parent_id）のうち status="evaluated" のもの
  2. 上記を LCB 降順でソートし、上位 sibling_context_k 件を選択
     - sibling_context_k = min(branch_factor * 2, 10)
     - 理由: LLMコンテキスト長の節約 + 情報過多による判断品質低下の防止
  3. 最良ノード（best_node）は parent_id に関わらず常に含める（参照点として）

■ 剪定済みノードの扱い:
  - status="pruned" のノードは参照対象に含めない
    理由: 剪定済みノードは探索的に不要と判断されたもの。
    コンテキスト枠の浪費が大きい
  - ただし、制約違反（feasible=false）で剪定されたノードは例外として
    最大2件まで含める（constraint_violation_examples）
    理由: 制約境界付近の情報は有用な負例として機能する

■ 失敗ノードの扱い:
  - status="failed" のノードは参照対象に含めない（debug オペレータが担当）
  - status="oom" / "timeout" のノードも含めない

■ 各ノードの提示粒度:
  提示項目（必須）:
    - hypothesis（1行要約）
    - experiment_config の親ノードとの diff（変更キーと値のみ）
    - μ ± SE（95% CI）
    - LCB
    - feasible フラグ
    - eval_runs 数（信頼度の指標として「n=3」等を付記）

  提示しない項目:
    - 実行ログ（stdout/stderr） → コンテキスト長の爆発を防止
    - experiment_code → improve は config レベルの変更であり、コード詳細は不要
    - metrics_raw → μ/SE に集約済み
    - rationale → 兄弟の分岐理由は改善提案に不要

■ draft オペレータの場合:
  兄弟ノードではなく「全評価済みノードの hypothesis 一覧」を提示する（§6.5.1）。
  提示粒度は hypothesis + method名 のみ（詳細不要、多様性確保が目的）。

■ debug オペレータの場合:
  兄弟ノードは参照しない。失敗ノード自身の experiment_code + error_message のみ（§6.5.2）。
```

### 6.9 失敗処理

```text
■ JSONパース失敗:
  - 最大3回リトライ（temperature += 0.1）
  - ```json ... ``` ブロック抽出の前処理
  - 3回失敗: 親ノードの config をそのままコピーした「同一実験」ノードを1つ生成
    （探索を完全停止させない安全策。ただし priority は低い）

■ バリデーション失敗（ホワイトリスト違反・範囲外）:
  - エラーメッセージをLLMにフィードバックして最大2回リトライ
  - リトライ後も全棄却: JSONパース失敗と同じフォールバック

■ 実験失敗（exit_code != 0）:
  - debug オペレータが自動発動（§6.5.2）
  - max_debug_depth（既定3）までリトライ
  - 超過したら諦めて "failed" のまま閉じる

■ 有効だが実験不能な設定（リソース超過）:
  - 事前検証: ResourceSpec.sandbox.experiment_memory_limit_gb を超えるメモリ要求は棄却
  - 実験実行時の判定: Phase 3 の Executor がOOMを検知（§7.5）→ status="oom"

■ OOM / timeout と debug オペレータの関係:
  - debug オペレータの対象は status="failed"（exit_code != 0 のロジックエラー）のみ
  - status="oom" / "timeout" は debug 対象外（コード修正では解決不能）
  - これらのノードは closed_set に入り、以後の探索・分岐生成から除外される
  - PPOバッファにも含めない（有意な metrics が存在しないため）

■ 制約違反ノードの学習対象としての扱い:
  - status="evaluated" かつ feasible=false のノードは PPO バッファに含める
  - 報酬計算（§9.2 compute_reward）で constraint_penalty が適用され、負の報酬信号となる
  - これにより PPO は制約違反領域を避けるように学習する
  - ただし、全ノードが制約違反の場合は PPO 更新をスキップする（§9.5）
```

### 6.10 探索と学習の境界

```text
■ 責務分離の原則:
  Phase 2（分岐生成）: 「何を探索するか」を決定する（3オペレータ: draft/debug/improve）
  Phase 5（PPO学習）: 「どれだけ上手に探索できるか」を改善する

■ PPO更新結果の反映タイミング:
  PPO更新によりLoRAが変わると、AgentLLMの生成品質が変化する。
  この変化は「次にAgentLLM.generate()が呼ばれたとき」に自然に反映される。
  つまり、PPO更新後に生成される子ノードは、更新前より質の高い仮説を生成する。

  具体的なタイミング:
  1. ステップN: ノードA を評価 → ppo_buffer に追加
  2. ステップN+k: ppo_buffer が ppo_trigger_interval に到達 → PPO更新実行
     → AgentLLM のLoRAが更新される
  3. ステップN+k+1: ノードB を展開 → AgentLLM.generate() で子ノード生成
     → この呼び出しは更新後のLoRAを使用する（自然に反映）

■ 分岐生成が学習に依存しない設計:
  分岐生成の入力は以下のみ:
  - parent の外部状態（hypothesis, experiment_config, metrics, error_message）
  - PlanSpec（固定）
  - ProblemSpec（固定）
  - RelatedWorkSpec（固定）

  学習結果（LoRA差分）は AgentLLM の内部状態として間接的にのみ影響する。
  分岐生成ロジック自体は学習の有無に関わらず同一のコードパスを通る。
```

### 6.11 探索メインループ（AIDE参考・疑似コード）
```python
def research_loop(specs, agent_llm):
    open_list = []                # SearchNode のリスト（priority でソート）
    closed_set = set()
    all_nodes = {}                # node_id → SearchNode
    best_node = None

    # Phase 2 初期化: ルートノード生成（draft オペレータ）
    root_children = tree_ops.draft(specs, agent_llm, rng, n=specs.exec.search.initial_root_children)
    for child in root_children:
        open_list.append(child)
        all_nodes[child.node_id] = child

    ppo_buffer = []
    step = 0

    while not should_terminate(step, best_node, specs):
        # Phase 2: select_next_node で次のノードとオペレータを自動決定（§6.4）
        selected, operator = select_next_node(open_list, all_nodes, specs.exec)

        if operator == "evaluate":
            # ── 未評価ノードの実行・評価 ──
            node = selected

            # Phase 3: 実験コード生成 + サンドボックス実行
            result = executor.run(node, specs, agent_llm)
            if result.exit_code != 0:
                node.status = "failed"
                node.error_message = result.stderr
                step += 1
                continue  # 次のループで debug が自動選択される

            # Phase 4: 統計評価（逐次）
            evaluate_node(node, specs)  # 初回は sequential_eval_initial 回
            if is_topk(node, all_nodes, k=specs.exec.evaluation.sequential_eval_topk):
                evaluate_node_full(node, specs)  # repeats まで追加
            node.status = "evaluated"
            node.priority = compute_priority(node, specs.exec)
            update_best(best_node, node)

        elif operator == "debug":
            # ── 失敗ノードの修復（§6.5.2） ──
            debug_child = tree_ops.debug(selected, agent_llm)
            open_list.append(debug_child)
            all_nodes[debug_child.node_id] = debug_child

        elif operator == "draft":
            # ── 新規アプローチの起草（§6.5.1） ──
            new_nodes = tree_ops.draft(specs, agent_llm, rng, n=2)
            for n in new_nodes:
                open_list.append(n)
                all_nodes[n.node_id] = n

        elif operator == "improve":
            # ── 最良ノードの原子的改善（§6.5.3） ──
            if selected.depth < specs.exec.search.max_depth:
                children = tree_ops.improve(
                    selected, specs, agent_llm, all_nodes, rng,
                    n_children=specs.exec.search.branch_factor
                )
                for child in children:
                    open_list.append(child)
                    all_nodes[child.node_id] = child

        # Phase 5: PPO 更新（評価済みバッファが溜まったら）
        evaluated_new = [n for n in all_nodes.values()
                         if n.status == "evaluated" and n.node_id not in closed_set]
        ppo_buffer.extend(evaluated_new)
        if len(ppo_buffer) >= specs.exec.learning.ppo_trigger_interval:
            ppo_trainer.update(ppo_buffer, agent_llm, specs)
            ppo_buffer.clear()

        # Phase 6: 系譜管理・剪定
        lineage_manager.maybe_squash(specs)
        pruned = pruner.prune(open_list, closed_set, specs)
        for p in pruned:
            closed_set.add(p.node_id)

        if selected and selected.node_id not in closed_set:
            if selected.status in ("evaluated", "failed"):
                closed_set.add(selected.node_id)
        step += 1

    return best_node
```

---

## 7. Phase 3：実験実行（Executor）

### 7.1 実行器の要件（必須）
- ローカル実行（MVP必須）
- SLURM実行（ResourceSpecで利用可能なら切替）
- コンテナ実行（ResourceSpecで必須なら、指定イメージで実行）

### 7.2 実験コード生成（具体）

```python
class ExperimentGenerator:
    """探索ノードの実験条件から実行可能な実験コードを生成する"""

    def generate(self, node: SearchNode, problem_spec: ProblemSpec, agent_llm: AgentLLM) -> str:
        """
        1. ProblemSpec.experiment_template をベースに、
           node.experiment_config の値で変数を埋める
        2. テンプレートが不十分な場合、LLM に以下を生成させる：
           - データ読み込み
           - 前処理
           - モデル構築/学習
           - 評価
           - metrics.json 出力
        3. 生成コードは必ず以下を満たす：
           - metrics.json を stdout ではなくファイルに出力
           - seed を受け取り np.random.seed / torch.manual_seed を設定
           - 例外時は stderr にトレースバックを出力し exit(1)
        """
        pass
```

### 7.3 実行インターフェース（プラグイン抽象クラス）

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunResult:
    node_id: str
    success: bool
    exit_code: int
    stdout_path: Path        # runs/<node_id>/stdout.log
    stderr_path: Path        # runs/<node_id>/stderr.log
    metrics_path: Path | None  # runs/<node_id>/metrics.json（成功時のみ）
    artifacts_dir: Path      # runs/<node_id>/artifacts/
    wall_time_sec: float
    seed: int

class Executor(ABC):
    @abstractmethod
    def run(self, node_id: str, script_path: Path, seed: int, timeout_sec: int) -> RunResult:
        """
        実験スクリプトを実行し、結果を返す。

        - 実行前: runs/<node_id>/ ディレクトリを作成
        - stdout/stderr をファイルにリダイレクト
        - タイムアウト超過は RunResult(success=False, exit_code=-9) を返す
        - OOM は RunResult(success=False, exit_code=-7) を返す
        """
        pass

class LocalExecutor(Executor):
    """subprocess.Popen でローカル実行"""
    pass

class SlurmExecutor(Executor):
    """sbatch でジョブ投入、sacct で完了待ち"""
    pass

class DockerExecutor(Executor):
    """docker run で隔離実行"""
    pass
```

### 7.4 metrics.json の規約（必須）
Executorは `runs/<node_id>/metrics.json` を出力する。

```json
{
  "primary": {"name": "score", "value": 0.73, "higher_is_better": true},
  "constraints": [
    {"name": "format_valid", "value": 1, "type": "bool", "satisfied": true},
    {"name": "latency_ms", "value": 45.2, "type": "le", "threshold": 100, "satisfied": true}
  ],
  "secondary": [
    {"name": "cost_gpu_sec", "value": 12.3, "lower_is_better": true},
    {"name": "memory_mb", "value": 2048.0, "lower_is_better": true}
  ],
  "raw": {"epoch": 10, "train_loss": 0.32, "val_loss": 0.41},
  "environment": {
    "python": "3.11.5",
    "torch": "2.3.0",
    "cuda": "12.1",
    "gpu": "NVIDIA A100 80GB"
  },
  "seed": 42,
  "wall_time_sec": 125.3
}
```

### 7.5 実験失敗時のハンドリング
```text
1. exit_code != 0 の場合：
   - metrics.json が存在しない → node.status = "failed"、primary = -inf、feasible = false
   - metrics.json が部分的 → 読めるフィールドのみ使用、不足は NaN
2. タイムアウトの場合：
   - node.status = "timeout"、cost = timeout_sec（全消費扱い）
3. OOM の場合：
   - node.status = "oom"、cost = 最大予算（ペナルティ扱い）
4. リトライ：
   - 失敗ノードのリトライは行わない（別の seed で新ノードとして扱う）
```

---

## 8. Phase 4：統計評価（Evaluator）

### 8.1 統計値（必須）
- `repeats` 回実行（または測定）して：
  - 平均 `μ = mean(values)`
  - 標準誤差 `SE = std(values) / sqrt(n)`
  - `LCB = μ - c * SE`（c = ExecutionSpec.evaluation.lcb_coef）
  - `UCB = μ + c * SE`（参考記録用）

### 8.2 逐次評価（sequential eval：具体アルゴリズム）
```python
def evaluate_node_sequential(node: SearchNode, executor: Executor, exec_spec: ExecutionSpec):
    """
    逐次評価アルゴリズム:
    1. 初回: sequential_eval_initial 回（既定1回）だけ実行
    2. μ, SE, LCB を暫定計算（n=1の場合 SE=inf, LCB=-inf）
    3. 全open_listノードのLCBでソートし、Top-k（sequential_eval_topk=5）に入るか判定
    4. Top-kに入るなら、repeats（既定3回）まで追加実行
    5. 追加実行ごとに μ, SE, LCB を再計算
    """
    # Step 1: 初回実行
    for i in range(exec_spec.evaluation.sequential_eval_initial):
        result = executor.run(node.node_id, seed=base_seed + i)
        node.metrics_raw.append(load_metrics(result.metrics_path))
        node.eval_runs += 1

    update_stats(node, exec_spec)  # μ, SE, LCB 計算

def evaluate_node_full(node: SearchNode, executor: Executor, exec_spec: ExecutionSpec):
    """Top-kノードに対して残りの反復を実行"""
    remaining = exec_spec.evaluation.repeats - node.eval_runs
    for i in range(remaining):
        result = executor.run(node.node_id, seed=base_seed + node.eval_runs + i)
        node.metrics_raw.append(load_metrics(result.metrics_path))
        node.eval_runs += 1

    update_stats(node, exec_spec)

def update_stats(node: SearchNode, exec_spec: ExecutionSpec):
    values = [m["primary"]["value"] for m in node.metrics_raw if m is not None]
    n = len(values)
    if n == 0:
        node.mu, node.se, node.lcb = None, None, None
        return
    node.mu = sum(values) / n
    if n >= 2:
        variance = sum((v - node.mu) ** 2 for v in values) / (n - 1)
        node.se = (variance ** 0.5) / (n ** 0.5)
    else:
        node.se = float('inf')
    node.lcb = node.mu - exec_spec.evaluation.lcb_coef * node.se
```

### 8.3 多指標（必須）
- primary / constraints / secondary を区別
- 制約は ε-constraint として feasibility 判定に使用：
  ```python
  def check_feasibility(node: SearchNode, problem_spec: ProblemSpec) -> bool:
      for constraint in problem_spec.constraints:
          metric = find_constraint_metric(node.metrics_raw, constraint.name)
          if constraint.type == "bool":
              if not metric["satisfied"]:
                  return False
          elif constraint.type == "ge":
              if metric["value"] < constraint.threshold - constraint.epsilon:
                  return False
          elif constraint.type == "le":
              if metric["value"] > constraint.threshold + constraint.epsilon:
                  return False
      return True
  ```
- secondary はタイブレーク時に使用：
  - LCBが同値（差が `plateau_min_improvement` 以下）の場合、secondary の加重和で順位決定

---

## 9. Phase 5：学習（PPOのみ、LoRA-only、差分継承）

### 9.1 学習の位置づけ（必須）
- 学習はテーマではなく **実行エージェント**の専門化
- 学習対象は LoRA パラメータのみ（ベースモデル凍結）
- 具体的には、エージェントLLMの「仮説生成」「実験設計」「コード生成」能力を研究ドメインに特化させる

### 9.1.1 外部探索木とLoRA系譜木の同期ルール（必須）

二重木構造（設計原則2）の同期は以下のルールに従う：

```text
外部探索木（SearchNode）          LoRA系譜木（adapter_node）
━━━━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━━━━━
root_child_1 (depth=0)  ───────► adapter_root (zero-init LoRA)
  ├─ child_A (depth=1)  ───────► adapter_root（同一。PPO未実行）
  │    └─ child_A1      ───────► adapter_A1（PPO更新でΔ生成）
  ├─ child_B (depth=1)  ───────► adapter_root（同一。PPO未実行）
  └─ child_C (depth=1)  ───────► adapter_C（PPO更新でΔ生成）
       └─ child_C1      ───────► adapter_C（同一。PPO未実行）
            └─ child_C1a ───────► adapter_C1a（PPO更新でΔ生成）
```

```python
def sync_adapter_assignment(search_node: SearchNode, ppo_updated: bool,
                            new_adapter_node_id: str | None):
    """
    PPO更新の有無に応じて adapter_node_id を設定する。

    ルール:
    1. PPO更新なし → 親の adapter_node_id を継承
    2. PPO更新あり → 新しい adapter_node_id を設定
    3. ルートノード（parent=None） → adapter_root（zero-init）を設定
    """
    if search_node.parent_id is None:
        search_node.adapter_node_id = "adapter_root"
    elif ppo_updated:
        search_node.adapter_node_id = new_adapter_node_id
    else:
        parent = get_node(search_node.parent_id)
        search_node.adapter_node_id = parent.adapter_node_id
```

> **重要**: 外部探索木のノード数 ≥ LoRA系譜木のノード数。PPO更新が走らなければ、多数の探索ノードが同一のアダプタを共有する。これは正常な動作であり、LoRAノードの「1対多」関係を許容する。

### 9.2 PPOのロールアウト収集（具体）
```python
class PPORollout:
    """1つの探索ノードの実験サイクルを1エピソードとするロールアウト"""
    node_id: str
    prompt: str              # LLMへの入力（仮説生成プロンプト）
    response: str            # LLMの出力（仮説+実験設計JSON）
    log_prob: float          # 出力トークン列の対数確率
    reward: float            # 計算された報酬
    value: float             # 価値関数の推定値
    advantage: float         # GAE で計算されたアドバンテージ

def compute_reward(node: SearchNode, plan_spec: PlanSpec) -> float:
    """
    報酬計算:
      R = primary_value
          - constraint_penalty * num_violated_constraints
          - lambda_cost * normalized_cost
          - kl_coef * kl_divergence

    各項の詳細:
      - primary_value: metrics.primary.value（direction=minimize の場合は -value）
      - constraint_penalty: PlanSpec.reward.constraint_penalty（既定10.0）
      - num_violated_constraints: 満たさなかった制約の数
      - normalized_cost: cost / budget_limit（budget_limit=null の場合は cost/max_observed_cost）
      - kl_divergence: 現LoRA と親LoRA 間のKL推定値（TRL の approx_kl）
    """
    pass
```

### 9.3 PPO更新ループ（具体）
```python
class PPOTrainer:
    """
    trl.PPOTrainer をラップし、LoRA-only 更新を実装する。

    依存: transformers, peft, trl
    """

    def update(self, rollouts: list[PPORollout], agent_llm: AgentLLM, specs: AllSpecs):
        """
        1. 現在のノードの LoRA アダプタをロード（materialize済み）
        2. rollouts から batch を構成（batch_size=16, mini_batch_size=4）
        3. epochs_per_update（既定4）回のミニバッチ更新：
           a. old_log_prob と new_log_prob の比率 r(θ) を計算
           b. クリッピング: min(r*A, clip(r, 1-ε, 1+ε)*A)
           c. 価値関数損失: MSE(V_pred, returns)
           d. エントロピーボーナス: entropy_coef * H(π)
           e. 合計損失 = -policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
           f. 勾配クリッピング: max_grad_norm=0.5
           g. optimizer.step()（LoRA パラメータのみ更新される）
        4. KL divergence を計算：
           - kl > kl_target * 1.5 → kl_coef *= 2（自動増加）
           - kl < kl_target / 1.5 → kl_coef /= 2（自動減少）
        5. 更新後のLoRAアダプタと親アダプタの差分Δを計算：
           Δ = updated_lora_weights - parent_lora_weights
        6. Δを adapter_delta.safetensors として保存
        7. ppo_log.jsonl にログ出力
        """
        pass
```

### 9.4 差分継承（delta inheritance：必須）
- 子のLoRA = 親のLoRA + Δ
- 保存するのは Δのみ（A/B差分）

保存形式：
```text
lineage/nodes/<adapter_node_id>/
  meta.json:
    {
      "adapter_node_id": "...",
      "parent_adapter_node_id": "...",  # null for root
      "search_node_id": "...",
      "adapter_spec_hash": "sha256:...",
      "depth": 3,
      "created_at": "2026-02-21T12:00:00Z",
      "delta_norm_l2": 0.0023,          # Δの L2 ノルム（品質監視用）
      "is_snapshot": false
    }
  adapter_delta.safetensors:
    - model.layers.0.self_attn.q_proj.lora_A.delta  (shape: [rank, in_features])
    - model.layers.0.self_attn.q_proj.lora_B.delta  (shape: [out_features, rank])
    - model.layers.0.self_attn.v_proj.lora_A.delta
    - model.layers.0.self_attn.v_proj.lora_B.delta
    - ...（全 target_layers × target_modules）
```

### 9.5 PPO更新トリガー条件
```text
PPO更新は以下のいずれかで発火：
1. 評価済みノードが ppo_trigger_interval（既定5）個溜まった
2. 現在の最良LCBが plateau_patience ステップ改善していない（学習促進のため）

更新しない条件：
1. 評価済みノードが2個未満（統計的に意味がない）
2. 全ノードが制約違反（有効な報酬信号がない）
```

---

## 10. Phase 6：系譜管理・剪定

### 10.1 LoRA系譜（必須）
- 各探索ノードは LoRA系譜ノードを参照する
- `adapter_spec_hash` が一致しない差分の合成は禁止（互換性検証）
  ```python
  def validate_compatibility(child_meta: dict, parent_meta: dict) -> bool:
      return child_meta["adapter_spec_hash"] == parent_meta["adapter_spec_hash"]
  ```

### 10.2 materialize（復元：必須）
```python
def materialize(adapter_node_id: str, lineage_dir: Path, cache: LRUCache) -> dict[str, Tensor]:
    """
    adapter_node_id のフルLoRAウェイトを復元する。

    アルゴリズム:
    1. キャッシュにあればそのまま返す
    2. meta.json を読み、is_snapshot=true ならスナップショットをロードして返す
    3. 親→ルートまでの系譜パスを構築: [root, ..., parent, self]
    4. 系譜パス上で最も近いスナップショットまたはキャッシュを探す
    5. そこからΔを順次累積: Θ = Θ_snapshot + Σ Δ_i
    6. 結果をキャッシュに保存
    7. キャッシュが cache_max_entries を超えたらLRU追い出し
    """
    # 1. キャッシュ確認
    if adapter_node_id in cache:
        return cache[adapter_node_id]

    # 2. スナップショット確認
    meta = load_meta(lineage_dir / adapter_node_id / "meta.json")
    if meta["is_snapshot"]:
        weights = load_safetensors(lineage_dir / adapter_node_id / "snapshot.safetensors")
        cache[adapter_node_id] = weights
        return weights

    # 3-5. 系譜を辿って累積
    path = build_lineage_path(adapter_node_id, lineage_dir)  # [root, ..., self]
    base_weights = None
    start_idx = 0
    for i, nid in enumerate(reversed(path)):
        if nid in cache:
            base_weights = cache[nid].copy()
            start_idx = len(path) - i
            break
        m = load_meta(lineage_dir / nid / "meta.json")
        if m["is_snapshot"]:
            base_weights = load_safetensors(lineage_dir / nid / "snapshot.safetensors")
            start_idx = len(path) - i
            break

    if base_weights is None:
        base_weights = {k: torch.zeros_like(v) for k, v in get_lora_template(meta).items()}
        start_idx = 0

    for nid in path[start_idx:]:
        delta = load_safetensors(lineage_dir / nid / "adapter_delta.safetensors")
        for key in base_weights:
            delta_key = key.replace(".weight", ".delta")
            if delta_key in delta:
                base_weights[key] = base_weights[key] + delta[delta_key]

    # 6-7. キャッシュ
    cache[adapter_node_id] = base_weights
    return base_weights
```

### 10.3 squash（スナップショット化）
`ExecutionSpec.lora_runtime` に従い：

```python
def maybe_squash(lineage_manager, exec_spec):
    """
    スナップショット生成条件:
    1. depth >= squash_depth（既定6）のノード
    2. snapshot_on_topk=true かつ Top-k（pruning.keep_topk）に含まれるノード
    3. 既にスナップショット済みのノードはスキップ

    処理:
    1. materialize で全ウェイトを復元
    2. snapshot.safetensors として保存
    3. meta.json の is_snapshot = true に更新
    4. このノードより上の祖先のΔは不要にはしない（他の枝が参照する可能性）
    """
    pass
```

### 10.4 剪定（必須：具体アルゴリズム）
```python
def prune(open_list, closed_set, all_nodes, exec_spec) -> list[SearchNode]:
    """
    剪定アルゴリズム:

    1. 保護リスト構築:
       - best_node とその祖先（root まで）
       - LCB上位 keep_topk（既定5）ノードとその祖先
       - status="running" のノード

    2. Pareto剪定（pareto=true の場合）:
       - primary(LCB) と cost の2軸でPareto支配判定
       - 支配されるノード = 他のノードに primary も cost も負けるノード
       - 支配されかつ保護リストにないノードを候補に

    3. LCB閾値剪定:
       - lcb_threshold が null なら: threshold = best_lcb * 0.5
       - LCB < threshold のノード（保護リスト除外）を候補に

    4. 予算剪定:
       - 全ノードの累計コスト > budget_limit なら、LCBワースト順に削除

    5. 候補に対して:
       - node.status = "pruned"
       - open_list から除去
       - save_pruned=false なら runs/<node_id>/ を削除
       - LoRA delta は保持（他ノードの祖先の可能性）
    """
    pass
```

---

## 11. Phase 7：論文生成（AI-Scientist-v2 参考）

### 11.1 目的
Evidence（実験ログ・統計・図）に基づき、PaperSpecの形式で論文草稿を生成。
AI-Scientist-v2 のワークフロー（VLM統合・ライティング内反省ループ・自動引用検索）を参考に、
高品質な論文を自律的に生成する。

### 11.2 EvidenceStore（論文生成の入力）
```python
@dataclass
class EvidenceStore:
    """Phase 2-6 の全結果を論文生成に必要な形式で保持"""
    best_node: SearchNode
    top_nodes: list[SearchNode]       # LCB上位ノード群
    all_evaluated_nodes: list[SearchNode]
    search_log: list[dict]            # search_log.jsonl の全エントリ
    eval_log: list[dict]              # eval_log.jsonl の全エントリ
    ppo_log: list[dict]               # ppo_log.jsonl の全エントリ
    problem_spec: ProblemSpec
    related_work: RelatedWorkSpec
    execution_spec: ExecutionSpec

    def get_main_results_table(self) -> str:
        """主要結果の表（ベースライン比較、CI付き）をMarkdown形式で返す"""
        pass

    def get_ablation_data(self) -> dict:
        """アブレーション実験データを返す（auto_ablation で実行されたもの）"""
        pass

    def get_convergence_data(self) -> list[tuple[int, float]]:
        """(step, best_lcb) の時系列データを返す"""
        pass

    def get_experiment_summaries(self) -> dict[str, list[dict]]:
        """stage別（baseline/research/ablation）の実験結果をJSON要約形式で返す"""
        pass
```

### 11.3 ワークフロー（具体：AI-Scientist-v2 スタイル）

Phase 7 は以下の **6ステップ** を順次実行する。ステップ5（ライティング内反省ループ）により、
Phase 8 に行く前に論文品質を内部で高める。

```python
class PaperComposer:
    """AI-Scientist-v2 スタイルの論文生成器"""

    def compose(self, evidence: EvidenceStore, paper_spec: PaperSpec,
                teacher_papers: TeacherPaperSet, agent_llm: AgentLLM,
                vlm: VLMReviewer | None = None) -> Paper:
        """
        ========================================
        ステップ 1: 実験ログ要約（Log Summarization）
        ========================================
        - evidence.get_experiment_summaries() から stage 別の構造化要約を生成
        - 各ノードの実験記述、手法、有意性、数値結果をJSON形式で抽出
        - 出力: baseline_summary.json, research_summary.json, ablation_summary.json

        ========================================
        ステップ 2: 図生成・集約（Plot Aggregation）
        ========================================
        - FigureGenerator で基本図を生成:
          a. CI棒グラフ: 各手法の primary ± CI
          b. 収束曲線: evidence.get_convergence_data()
          c. 探索木可視化: graphviz で上位ノードのツリー
          d. アブレーション表: LaTeX/Markdown テーブル
        - LLM が実験結果を統合した追加集約図スクリプトを生成（反省ループ最大5回）
        - 最大12枚、300 DPI
        - 出力: paper/figures/*.png

        ========================================
        ステップ 3: 自動引用検索（Citation Search Loop）
        ========================================
        - 最大 citation_search_rounds（既定20）ラウンド:
          a. LLM が現在の論文コンテキストから「最も重要な不足引用」を特定
          b. LLM が Semantic Scholar 検索クエリを生成
          c. Semantic Scholar API で論文検索
          d. LLM が検索結果から関連論文を選択し、説明を付与
          e. BibTeX エントリを生成し paper.bib に追加
          f. LLM が "No more citations needed" と判断したら早期終了
        - related_work_spec.papers の既存引用はベースとして保持
        - 出力: paper/paper.bib（既存+新規引用）

        ========================================
        ステップ 4: VLM 図記述生成（VLM Figure Description）
        ========================================
        ※ vlm が None（VLM未設定）の場合はスキップ
        - VLM が paper/figures/ 内の各 PNG を視覚的に分析
        - 各図に対して科学的記述（軸、トレンド、パターン、統計的特徴）を生成
        - 出力: figure_descriptions: dict[str, str]（ファイル名→記述のマッピング）
        - この記述はステップ5のLLMコンテキストに注入される

        ========================================
        ステップ 5: 論文本体生成 + ライティング内反省ループ
        ========================================
        5a. アウトライン生成
            - LLM に evidence.problem_spec + teacher_papers.structure_summary を与え、
              paper_spec.sections_required に沿ったアウトライン（各章の箇条書き）を生成

        5b. 初版LaTeX/Markdown生成（1パス）
            - LLM（big model）に以下を統合入力:
              * 研究概要（problem_spec）
              * 実験要約（ステップ1の出力）
              * 図ファイル一覧 + VLM記述（ステップ4の出力）
              * 引用一覧（ステップ3の出力）
              * teacher_papers の構造サマリー
              * LaTeX/Markdownテンプレート（paper_spec.format に応じて選択）
            - 全セクションを1パスで生成:
              a. Abstract: 目的→手法→結果→結論を300語以内で
              b. Introduction: motivation → 課題 → 提案 → 貢献リスト → 構成
              c. Related Work: related_work_spec.clusters ごとに段落、比較表
              d. Method: ProblemSpec → 探索アルゴリズム → 評価方法 → PPO+LoRA → 擬似コード
              e. Experiments:
                 - Setup（データ/モデル/ハイパーパラメータ）
                 - evidence.get_main_results_table() を挿入
                 - ベースライン比較の解釈
              f. Ablation: evidence.get_ablation_data() に基づく表と考察
              g. Discussion: limitations（open_problems から自動抽出）+ future_work
              h. Conclusion: 貢献の再確認、主要結果の要約

        5c. ライティング内反省ループ（最大 n_writeup_reflections 回、既定3）
            各ラウンドで以下を実行:
            i.   LaTeX の場合: pdflatex 4-pass コンパイル + chktex 構文チェック
                 Markdown の場合: pandoc で整合性検証
            ii.  未使用図の検出（figures/ にあるが本文で参照されていない図）
            iii. 無効な図参照の検出（本文で参照されているが figures/ にない図）
            iv.  VLM 図・キャプション・本文参照レビュー（vlm が有効な場合）:
                 - VLM が各図について以下を評価:
                   * 図の内容とキャプションの整合性
                   * 本文中の参照箇所での説明の適切性
                   * 図の情報量と有用性（ページ制約下で残す価値があるか）
                 - 出力: 図ごとの {img_review, caption_review, figrefs_review}
            v.   VLM 重複図検出（vlm が有効な場合）:
                 - 全図を VLM に送り、内容が類似する図のペアを特定
                 - 本文とAppendixの間の重複も検出
            vi.  反省プロンプトを LLM に送信（上記i-vの結果を含む）:
                 - 構文エラーの修正
                 - 科学的正確性・明快さの改善
                 - 実験要約との整合性（データ捏造の防止）
                 - 未使用図の組み込みまたは無効参照の削除
                 - VLMレビュー結果に基づく図・キャプション修正
                 - 重複図の統合・削除
            vii. LLM が修正版を返す（または "I am done" で早期終了）

        ========================================
        ステップ 6: 統合・最終出力
        ========================================
        - 図表番号の最終整合
        - 引用キーの整合（paper.bib との一致確認）
        - 出力: paper/paper.md（or paper.tex）, paper/figures/*, paper/paper.bib

        """
        pass
```

### 11.4 VLMReviewer（図の視覚レビュー）

```python
class VLMReviewer:
    """VLM（Vision Language Model）による図の視覚的レビュー"""

    def __init__(self, model: str, provider: str):
        """
        model: VLMモデル名（例: "gpt-4o", "claude-sonnet-4-20250514"）
        provider: "openai" | "anthropic"
        ※ ModelSpec.vlm で設定。未設定の場合、PaperComposer は VLM ステップをスキップ
        """

    def describe_figures(self, figure_paths: list[Path]) -> dict[str, str]:
        """
        各図を視覚的に分析し、科学的記述を生成。
        戻り値: {filename: description} のマッピング
        """
        pass

    def review_figure_caption_refs(self, figure_path: Path, caption: str,
                                    text_refs: list[str]) -> dict:
        """
        図・キャプション・本文参照の整合性をレビュー。

        戻り値:
        {
            "img_description": "...",    # VLMによる図の記述
            "img_review": "...",         # 図自体の品質評価
            "caption_review": "...",     # キャプションとの整合性
            "figrefs_review": "...",     # 本文参照の適切性
            "informative": true/false,   # ページ制約下で残す価値があるか
            "suggestion": "keep" | "move_to_appendix" | "remove" | "merge"
        }
        """
        pass

    def detect_duplicate_figures(self, figure_paths: list[Path]) -> list[dict]:
        """
        内容が類似する図のペアを検出。
        戻り値: [{"fig_a": str, "fig_b": str, "similarity": str, "recommendation": str}]
        """
        pass
```

### 11.5 CitationSearcher（自動引用検索）

```python
class CitationSearcher:
    """Semantic Scholar API を使った自動引用検索（AI-Scientist-v2 参考）"""

    def __init__(self, semantic_scholar_client: SemanticScholarClient, agent_llm: AgentLLM):
        pass

    def search_loop(self, context: str, existing_bibtex: str,
                    max_rounds: int = 20) -> list[dict]:
        """
        自動引用検索ループ:
        1. LLM が context（研究概要+実験要約）から最も重要な不足引用を特定
        2. LLM が Semantic Scholar 用の検索クエリを生成
        3. Semantic Scholar API で検索
        4. LLM が検索結果から関連論文を選択、説明を付与
        5. BibTeX エントリを生成
        6. LLM が "No more citations needed" と判断するか、max_rounds に達したら終了

        戻り値: [{paper_id, title, authors, year, bibtex_key, bibtex_entry, description}]
        """
        pass
```

### 11.6 自動アブレーション（auto_ablation=true の場合）
```text
アブレーション実験の自動設計:
1. best_node の experiment_config から各操作変数を1つずつ「ベースライン値」に戻す
2. 例: best_config = {lr: 1e-3, batch: 64, method: "proposed"}
   - ablation_1: {lr: default, batch: 64, method: "proposed"}  # lr の効果
   - ablation_2: {lr: 1e-3, batch: default, method: "proposed"}  # batch の効果
   - ablation_3: {lr: 1e-3, batch: 64, method: "baseline_A"}  # method の効果
3. 各アブレーション条件を repeats 回実行し、CI を計算
4. best と比較して有意差があるかを判定
```

### 11.7 出力（必須）
- `paper/paper.md`（既定）
- `paper/figures/*.png`（matplotlib で生成、最大12枚、300 DPI）
- `paper/paper.bib`（BibTeX：既存引用 + 自動検索引用）
- `paper/paper.tex`（オプション：pandoc で md → tex 変換）
- `paper/figure_descriptions.json`（VLM による図記述、VLM有効時のみ）
- `paper/citation_search_log.jsonl`（引用検索の各ラウンドの記録）

---

## 12. Phase 8：論文評価・改善ループ（AI-Scientist-v2 参考）

### 12.1 PaperScore 評価方法（具体）

AI-Scientist-v2 の査読方式を参考に、**Few-shot査読例**、**アンサンブルレビュー**、**レビュアー反省ループ**を導入。

```python
class PaperEvaluator:
    """PaperScoreSpec に基づく LLM-as-Judge 論文評価（AI-Scientist-v2 スタイル）"""

    def evaluate(self, paper_md: str, paper_score_spec: PaperScoreSpec,
                 agent_llm: AgentLLM) -> PaperScoreResult:
        """
        ============================================================
        ステップ 1: 単体レビュー生成（num_reviews_ensemble 回の独立レビュー）
        ============================================================
        各レビューは以下の手順で生成:

        a. システムプロンプト:
           ---
           あなたは一流学術会議の査読者です。提出された論文を批判的かつ公正に評価してください。
           不確実な場合は低めのスコアをつけ、リジェクトを推奨してください。
           ---
           ※ bias_mode 設定に応じてシステムプロンプトを切替:
           - "critical"（既定）: 不確実なら低スコア
           - "generous": 不確実なら高スコア

        b. Few-shot 査読例の注入:
           - paper_score_spec.few_shot_reviews から最大 num_fs_examples 件を選択
           - 各例は {paper_excerpt, review_json} のペア
           - 査読品質のアンカーとして機能（スコアの校正効果）

        c. 評価プロンプト:
           ---
           ## 評価基準（PaperScoreSpec）
           {paper_score_spec の各 criteria を rubric 付きで列挙}

           ## 参考: 査読例
           {few_shot_reviews（num_fs_examples 件）}

           ## 論文
           {paper_md 全文}

           ## 指示
           まず <THOUGHT> タグ内で論文の長所・短所を詳細に分析してください。
           次に以下の JSON 形式で評価結果を出力してください。

           ## 出力形式（JSON）
           {
             "summary": "論文の要約（2-3文）",
             "strengths": ["強み1", "強み2", ...],
             "weaknesses": ["弱み1", "弱み2", ...],
             "questions": ["質問1", "質問2", ...],
             "limitations": ["限界1", "限界2", ...],
             "scores": {
               "statistical_rigor": {"score": 7, "justification": "..."},
               "baseline_coverage": {"score": 8, "justification": "..."},
               "originality": {"score": 6, "justification": "..."},
               "clarity": {"score": 7, "justification": "..."},
               ...
             },
             "overall_score": 7,
             "confidence": 4,
             "decision": "accept" | "weak_accept" | "borderline" | "weak_reject" | "reject",
             "missing_items": [
               {"category": "ablation_quality", "description": "...", "severity": "high|medium|low"},
               ...
             ],
             "improvement_instructions": [
               {"priority": 1, "action": "...", "requires_experiment": true},
               {"priority": 2, "action": "...", "requires_experiment": false},
               ...
             ]
           }
           ---

        d. レビュアー反省ループ（最大 num_reviewer_reflections 回、既定2）:
           各ラウンドで:
           - レビュアーが自身の評価の正確性・公正性を再検討
           - スコアと根拠の整合性を確認
           - 見落とした観点がないか検討
           - 修正版レビューを出力（または "I am done" で早期終了）

           反省プロンプト:
           ---
           ラウンド {current}/{max} です。
           上記であなたが作成したレビューの正確性と公正性を慎重に再検討してください。
           評価基準のルーブリックとスコアの整合性、見落とした長所・短所がないか確認してください。
           元のレビューの方向性は維持しつつ、明らかな問題があれば修正してください。
           ---

        ============================================================
        ステップ 2: アンサンブル集約（num_reviews_ensemble > 1 の場合）
        ============================================================
        - 各スコア次元について: 全レビューのスコアを収集 → 平均 → 最近整数に丸め
        - overall_score: 重み付き平均を再計算（LLM の自己評価値とは独立に算出）
        - strengths/weaknesses/questions/limitations: 全レビューからユニークな項目を統合
        - improvement_instructions: 全レビューから統合し、出現頻度でソート

        アンサンブル集約が有効な場合、メタレビュー（Area Chair モード）を追加生成:
        ---
        あなたはエリアチェアです。以下の {n} 件の独立した査読結果を統合し、
        最終的な判定と改善指示を作成してください。
        各レビュアーの意見の相違点に特に注意し、根拠に基づいて判断してください。

        {individual_reviews}

        出力: 統合 PaperScoreResult（上記と同じJSON形式）
        ---

        ============================================================
        ステップ 3: 結果構築
        ============================================================
        1. overall_score = Σ(score_i * weight_i)（重み付き平均を計算）
        2. passed = overall_score >= passing_score
        3. PaperScoreResult を返す
        """
        pass

@dataclass
class PaperScoreResult:
    scores: dict[str, dict]         # {criteria_name: {"score": int, "justification": str}}
    overall_score: float
    confidence: float               # レビュアーの確信度（1-5）
    summary: str                    # 論文の要約
    strengths: list[str]            # 強みリスト
    weaknesses: list[str]           # 弱みリスト
    questions: list[str]            # 質問リスト
    limitations: list[str]          # 限界リスト
    missing_items: list[dict]
    improvement_instructions: list[dict]
    decision: str                   # "accept" | "weak_accept" | "borderline" | "weak_reject" | "reject"
    passed: bool                    # overall_score >= passing_score
    individual_reviews: list[dict] | None  # アンサンブル時の個別レビュー（記録用）
    meta_review: str | None         # アンサンブル時のメタレビュー
```

### 12.2 改善ループ（具体）
```python
def paper_improvement_loop(paper_composer, paper_evaluator, evidence, specs, agent_llm):
    """
    1. paper_composer.compose() で初版生成（Phase 7：ライティング内反省ループ含む）
    2. paper_evaluator.evaluate() で採点（アンサンブル+レビュアー反省含む）
    3. passed=true なら終了
    4. passed=false なら:
       a. improvement_instructions をソート（priority 順）
       b. requires_experiment=true の指示があれば:
          - 追加実験を Phase 3-4 で実行（ExecutionSpec の範囲内）
          - evidence を更新
       c. requires_experiment=false の指示は LLM で論文テキストを修正
       d. 修正版に対してライティング内反省ループを再実行（ステップ5c）
       e. 修正版を paper_evaluator.evaluate() で再採点
    5. paper_revision_limit（既定3）回まで繰り返し
    6. 最終版の paper_score を paper_log.jsonl に記録
    7. 全ラウンドの個別レビュー・メタレビューも paper_log.jsonl に記録

    注意:
    - 追加実験は max_nodes の残り枠内でのみ実行可能
    - 残り枠がなければ requires_experiment=true の改善はスキップ
    - Phase 7 のライティング内反省ループと Phase 8 の外部評価ループは二重構造:
      Phase 7（内部）: 構文・図・整合性の自己改善（コンパイルエラー、VLMレビュー等）
      Phase 8（外部）: 科学的品質の第三者的評価（査読シミュレーション）
    """
    for iteration in range(specs.exec.paper.paper_revision_limit):
        paper = paper_composer.compose(evidence, specs.paper_spec, specs.teacher_papers, agent_llm)
        result = paper_evaluator.evaluate(paper.to_markdown(), specs.paper_score_spec, agent_llm)

        log_paper_iteration(iteration, result)

        if result.passed:
            return paper, result

        # 改善
        for instruction in sorted(result.improvement_instructions, key=lambda x: x["priority"]):
            if instruction["requires_experiment"]:
                if can_run_more_experiments(specs):
                    run_additional_experiment(instruction, evidence, specs)
            else:
                paper = revise_paper_section(paper, instruction, agent_llm)

    return paper, result  # 最終版（未合格でも返す）
```

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

### 13.5 Phase 6（系譜）
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
      experiment.py                # LLM生成の実験スクリプト
      stdout.log
      stderr.log
      metrics.json                 # §7.4参照
      artifacts/                   # 図、チェックポイント等

  logs/
    search_log.jsonl               # §17.1参照
    eval_log.jsonl                 # §17.2参照
    ppo_log.jsonl                  # §17.3参照
    paper_log.jsonl                # §17.4参照
    agent_llm_log.jsonl            # 全LLM呼び出しログ（プロンプト/レスポンス）

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

## 16. CLI仕様（必須）

### 16.1 実装
- フレームワーク: **Typer**（型ヒント駆動、自動ヘルプ生成）
- エントリポイント: `sera` コマンド（`pyproject.toml` の `[project.scripts]` で定義）

### 16.2 コマンド
```bash
# Phase 0-1: 初期化
sera init input1.yaml                # Input-1 読み込み、workspace作成
sera phase0-related-work             # Phase 0: 先行研究収集
sera freeze-specs                    # Phase 1: 全Spec確定、ExecutionSpec固定

# Phase 2-6: 研究ループ
sera research                        # Phase 2-6 統合ループ実行
sera research --resume               # チェックポイントから再開

# 出力
sera export-best                     # best成果物を outputs/best/ に集約

# Phase 7-8: 論文
sera generate-paper                  # Phase 7: 論文生成
sera evaluate-paper                  # Phase 8: 論文評価・改善ループ

# ユーティリティ
sera status                          # 現在の探索状態サマリ表示
sera show-node <node_id>             # ノード詳細表示
sera replay --node-id <id> --seed <s>  # 特定ノードの実験再実行
sera validate-specs                  # Spec整合性チェック
```

### 16.3 引数（全コマンド共通）
```bash
# Phase 0 関連
--topk 20                  # top_k_papers
--teacher-papers 8         # teacher_papers
--citation-depth 2         # citation_graph_depth
--years-bias 3             # recent_years_bias
--api-priority "semantic_scholar,crossref,arxiv"

# Phase 2 関連
--max-nodes 200
--max-depth 12
--branch-factor 5
--lambda-cost 0.2
--beta 0.1

# Phase 4 関連
--repeats 5
--lcb-coef 2.576           # 99%CI
--no-sequential
--seq-topk 10

# Phase 5 関連
--rank 32
--alpha 64
--lr 5e-5
--clip 0.1
--ppo-steps 256
--no-kl

# モデル関連
--base-model "meta-llama/Llama-3.1-8B-Instruct"
--dtype "4bit"
--agent-llm "openai:gpt-4o"    # provider:model_id 形式

# リソース関連
--executor "slurm"
--work-dir "./my_workspace"
--timeout 7200
--no-web
--auto                      # Phase 1 のユーザ承認をスキップ
```

**重要**：引数反映は `freeze-specs` 実行まで。以後はSpec固定で動く。

### 16.4 終了コード
| コード | 意味 |
|--------|------|
| 0 | 正常終了 |
| 1 | 一般エラー（設定ミス、ファイル不在等） |
| 2 | ExecutionSpec 改竄検知 |
| 3 | adapter_spec_hash 不整合 |
| 10 | Phase 0 全API失敗 |
| 11 | Phase 3 全実験失敗（有効ノードゼロ） |
| 12 | 予算超過による強制終了 |
| 20 | ユーザによる中断（Ctrl+C → チェックポイント保存後終了） |

---

## 17. ログ仕様（必須）

### 17.1 search_log.jsonl
```json
{
  "event": "node_selected",
  "node_id": "uuid-...",
  "parent_id": "uuid-...",
  "depth": 3,
  "priority": 0.72,
  "lcb": 0.68,
  "chosen_reason": "highest_priority",
  "open_list_size": 15,
  "total_nodes": 42,
  "budget_consumed": {"gpu_minutes": 23.5},
  "timestamp": "2026-02-21T12:00:00Z"
}
```

### 17.2 eval_log.jsonl
```json
{
  "event": "evaluation_complete",
  "node_id": "uuid-...",
  "repeat_idx": 2,
  "metrics": {"primary": {"name": "score", "value": 0.73}, "constraints": [...]},
  "mu": 0.71,
  "se": 0.015,
  "lcb": 0.681,
  "ucb": 0.739,
  "n_repeats_done": 3,
  "cost_sec": 125.3,
  "timestamp": "2026-02-21T12:05:00Z"
}
```

### 17.3 ppo_log.jsonl
```json
{
  "event": "ppo_update",
  "adapter_node_id": "adapter-uuid-...",
  "parent_adapter_node_id": "adapter-uuid-...",
  "search_node_ids": ["uuid-1", "uuid-2", "uuid-3"],
  "mean_reward": 0.65,
  "kl_divergence": 0.018,
  "policy_loss": -0.023,
  "value_loss": 0.15,
  "entropy": 2.3,
  "kl_coef_current": 0.01,
  "total_steps": 128,
  "delta_norm_l2": 0.0023,
  "timestamp": "2026-02-21T12:10:00Z"
}
```

### 17.4 paper_log.jsonl
```json
{
  "event": "paper_evaluation",
  "iteration": 1,
  "overall_score": 6.8,
  "scores": {"statistical_rigor": 7, "baseline_coverage": 8, "ablation_quality": 5, ...},
  "passed": false,
  "missing_items": [{"category": "ablation_quality", "description": "..."}],
  "actions_taken": ["added_ablation_experiment", "revised_introduction"],
  "additional_experiments_run": 2,
  "timestamp": "2026-02-21T13:00:00Z"
}
```

### 17.5 agent_llm_log.jsonl（新規）
```json
{
  "event": "llm_call",
  "call_id": "call-uuid-...",
  "phase": "phase2",
  "purpose": "branch_generation",
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "adapter_node_id": "adapter-uuid-...",
  "prompt_tokens": 1523,
  "completion_tokens": 412,
  "temperature": 0.7,
  "prompt_hash": "sha256:...",
  "response_hash": "sha256:...",
  "latency_ms": 2340,
  "timestamp": "2026-02-21T12:01:00Z"
}
```

---

## 18. 非機能要件（必須）

### 18.1 言語・ランタイム
- Python >= 3.11
- パッケージマネージャ: pip（pyproject.toml で依存管理）

### 18.2 主要依存ライブラリ
```toml
[project]
dependencies = [
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "trl>=0.8.0",
    "safetensors>=0.4.0",
    "typer>=0.12.0",
    "pyyaml>=6.0",
    "httpx>=0.27.0",           # API呼び出し（async対応）
    "tenacity>=8.2.0",         # リトライ制御
    "matplotlib>=3.8.0",       # 図生成
    "seaborn>=0.13.0",         # 統計可視化
    "graphviz>=0.20.0",        # 探索木可視化
    "numpy>=1.26.0",
    "pydantic>=2.6.0",         # Spec のバリデーション
    "rich>=13.7.0",            # CLIの出力整形
    "structlog>=24.1.0",       # 構造化ログ
]

[project.optional-dependencies]
slurm = ["submitit>=1.5.0"]    # SLURM実行
docker = ["docker>=7.0.0"]     # Docker実行
latex = ["pandoc"]             # LaTeX変換（システムパッケージ）

[project.scripts]
sera = "sera.cli:app"
```

### 18.3 Spec バリデーション
- 全Spec は **Pydantic v2 モデル** で定義し、ロード時に自動バリデーション
- ExecutionSpec は `model_validator` でハッシュ検証を実施

### 18.4 プラグイン設計
- `Executor`（LocalExecutor / SlurmExecutor / DockerExecutor）
- `Evaluator`（StatisticalEvaluator / BootstrapEvaluator）
- `TreeOps`（LLMTreeOps / TemplateTreeOps — draft/debug/improve 3オペレータ）
- `PaperComposer`（MarkdownComposer / LaTeXComposer）
- 各プラグインは ABC（抽象基底クラス）を継承し、`ResourceSpec` / `PlanSpec` で切り替え

### 18.5 エラーハンドリング・リトライ
```text
API呼び出し（Phase 0）:
  - tenacity: retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=60))
  - HTTP 429 → backoff、HTTP 5xx → リトライ、HTTP 4xx → 次のAPI

実験実行（Phase 3）:
  - タイムアウト → status="timeout"、リトライなし（新ノードとして扱う）
  - OOM → status="oom"、リトライなし
  - その他エラー → status="failed"、リトライなし

PPO更新（Phase 5）:
  - NaN/Inf 検出 → 更新スキップ、警告ログ
  - GPU OOM → batch_size を半減してリトライ（最大2回）

中断復帰:
  - Ctrl+C → SIGINT ハンドラで checkpoints/ にスナップショット保存
  - sera research --resume でチェックポイントからロード
  - チェックポイント: open_list, closed_set, best_node, ppo_buffer, step
```

### 18.6 ネットワーク制御
- `ResourceSpec.network.allow_internet` に従う
- Phase 3 の実験中に allow_internet=false の場合、サブプロセスの環境変数 `http_proxy`/`https_proxy` を無効化

### 18.7 決定論的再現
- seed固定: `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`
- LLM temperature=0 を設定しても完全決定論にはならないため、全LLM入出力をログ保存（agent_llm_log.jsonl）

---

## 19. 受け入れ基準（Acceptance Criteria）

最低限、以下を満たすこと：

1. Input-1 から Input-2（全9 Spec）を生成し、Phase 1で ExecutionSpec を SHA-256 ハッシュ付きで固定できる
2. Semantic Scholar API 優先＋フォールバック（CrossRef→arXiv→Web）が動作する（Phase 0）
3. Best-First探索が動作し、max_nodes 内で複数ノードを生成・評価・選択できる（Phase 2–4）
4. LCB（μ, SE, LCB）が repeats 回の実行から算出され、優先度に反映される（Phase 4）
5. 逐次評価により Top-k ノードのみが全 repeats を実行し、それ以外は early stop する（Phase 4）
6. PPOで LoRA-only 更新が動作し、差分 Δ が adapter_delta.safetensors に保存される（Phase 5）
7. 差分継承の materialize が整合し、depth >= squash_depth で自動 squash される（Phase 6）
8. Pareto 剪定が動作し、支配されるノードが open_list から除去される（Phase 6）
9. best成果物（best_node.json + adapter.safetensors + metrics_summary.json + report.json）が `outputs/best/` に揃う
10. PaperSpec に沿った論文草稿が CI 付き図表を含んで生成される（Phase 7）
11. 自動引用検索ループが Semantic Scholar API で動作し、citation_search_log.jsonl に記録される（Phase 7）
12. ライティング内反省ループが最大 n_writeup_reflections 回動作し、構文エラー・未使用図等を自己修正する（Phase 7）
13. VLM が有効な場合、図記述生成・キャプションレビュー・重複図検出が動作する（Phase 7）
14. PaperScoreSpec に基づき LLM-as-judge 評価が動作し、アンサンブルレビュー（num_reviews_ensemble 件）+レビュアー反省ループが機能する（Phase 8）
15. 改善ループが最大 paper_revision_limit 回回り、Phase 7（内部反省）と Phase 8（外部評価）の二重ループ構造が機能する（Phase 7-8）
16. `docs/` が仕様通り揃い、quickstart 通りに第三者が `pip install` → `sera init` → `sera research` → `sera generate-paper` を再現できる
17. 全 LLM 呼び出しが agent_llm_log.jsonl に記録され、API 呼び出しが queries.jsonl に記録される（再現性）
18. ExecutionSpec の改竄（ハッシュ不一致）を検知し、exit code 2 で停止する

---

## 20. MVP優先順位（実装順）

1) **Phase 0**（先行研究：Semantic Scholar API + フォールバック + queries.jsonl 記録）
2) **Phase 1**（Spec 確定 + ExecutionSpec SHA-256 固定 + Pydantic バリデーション）
3) **Phase 2–4**（Best-First 探索 + LLM分岐生成 + LocalExecutor + LCB 逐次評価）
4) **Phase 5–6**（PPO LoRA-only + delta safetensors + materialize + squash + Pareto 剪定）
5) **Phase 7**（PaperComposer + 図生成 + VLMレビュー + 自動引用検索 + ライティング内反省ループ + EvidenceStore）
6) **Phase 8**（LLM-as-judge アンサンブル評価 + レビュアー反省ループ + 改善ループ）
7) **拡張**（SlurmExecutor / DockerExecutor / LaTeX出力 / Bootstrap評価）

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
    search_node.py                 # SearchNode データクラス（§6.2）
    search_manager.py              # Best-First ループ本体 + select_next_node（§6.4, §6.11）
    tree_ops.py                    # draft / debug / improve 3オペレータ（§6.5）
    priority.py                    # compute_priority, compute_exploration_bonus（§6.3）
    validation.py                  # validate_experiment_config（§6.6.1）

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
    reward.py                      # compute_reward
    rollout.py                     # PPORollout データクラス

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

  agent/                           # エージェントLLM
    __init__.py
    agent_llm.py                   # AgentLLM（ベースモデル+LoRA管理、推論）
    prompt_templates.py            # 各Phase用プロンプトテンプレート

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
  test_cli/                        # CLI 統合テスト
  conftest.py                      # pytest フィクスチャ
```

---

## 付録A：先行研究（docs/related_work.md の最低要件）

> **注意**：実装時は Phase 0 で収集した文献を基礎に、ここで規定した枠組みに従って整理文書を生成すること（テンプレ＋自動挿入で可）。

### A-1 自律研究・論文生成（AI-Scientist系）
- AI-Scientist：アイデア→実験→論文→査読風評価の一気通貫
- AI-Scientist-v2：agentic tree search / 管理エージェントの強化
- SERA対応：Phase 7–8、及び Phase 2（探索木）

### A-2 自律コード実験（CodeScientist等）
- コード実験中心の研究支援
- SERA対応：Phase 3–4（実験と評価）

### A-3 探索（tree search / agentic search）
- SERA対応：Phase 2（Best-First + 統計LCB + ε-constraint）

### A-4 LoRA / PEFT / 継続学習・アダプタ管理
- LoRA（PEFTの基盤）
- Delta-LoRA（delta概念の参照）
- アダプタ継続学習（複数タスク/分岐の背景）
- SERA対応：Phase 5–6（PPO+LoRA差分継承+系譜）

### A-5 Scholar API / 文献収集基盤
- Scholar公式APIがない可能性→第三者API/代替API前提
- SERA対応：Phase 0（優先順位＋再現ログ）

---

## 付録B：実装上の注意（禁止事項）

### B-1 ExecutionSpec固定
- Phase 2以降に探索/評価/学習の規定値を暗黙変更してはならない（ExecutionSpec固定違反）
- PlanSpec.branching.ops の重みを探索中に動的変更してはならない（固定原則違反）

### B-2 変数可変性の境界
- 分岐生成の experiment_config に ProblemSpec.manipulated_variables に存在しないキーを含めてはならない（ホワイトリスト違反）
- ExecutionSpec に属するパラメータ（PPOのlr, repeats, lcb_coef 等）を experiment_config 経由で変更してはならない（層境界違反）
- 分岐生成時にバリデーション（validate_experiment_config）をスキップしてはならない

### B-3 LoRA系譜の整合性
- adapter_spec_hash が異なる delta を合成してはならない（互換性違反）
- PPO更新なしのノードに新規 adapter_node_id を割り当ててはならない（系譜整合性違反）
- 分岐生成ロジック内で adapter_node_id を直接操作してはならない（責務分離違反。adapter_node_id は Phase 5 でのみ設定される）

### B-4 再現性
- API検索の結果を保存せずに進めてはならない（再現性違反）
- LLM呼び出しをログに記録せずに進めてはならない（再現性違反）
- 実験スクリプト内で seed を設定せずに実行してはならない（再現性違反）
- metrics.json を標準出力のみに頼ってはならない（ファイル出力必須）

### B-5 評価一貫性
- PaperSpec/PaperScoreSpec を Phase 7以降に恣意的に差し替えてはならない（評価一貫性違反）

---

## 付録C：AgentLLM インターフェース

```python
class AgentLLM:
    """
    SERAの全LLM呼び出しを統一管理するクラス。

    責務:
    1. ベースモデルのロード（HuggingFace transformers）
    2. LoRAアダプタの動的切り替え（peft）
    3. 推論（generate）
    4. 外部APIプロバイダへの転送（OpenAI/Anthropic）
    5. 全呼び出しのログ記録（agent_llm_log.jsonl）
    """

    def __init__(self, model_spec: ModelSpec, resource_spec: ResourceSpec):
        if model_spec.agent_llm.provider == "local":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_spec.base_model.id,
                revision=model_spec.base_model.revision,
                torch_dtype=getattr(torch, model_spec.base_model.dtype),
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_spec.base_model.id,
                revision=model_spec.compatibility.tokenizer_revision,
            )
            # 初期LoRA（zero init → ベースモデルと同一出力）
            self.model = get_peft_model(self.model, LoraConfig(
                r=model_spec.adapter_spec.rank,
                lora_alpha=model_spec.adapter_spec.alpha,
                target_modules=model_spec.adapter_spec.target_modules,
                lora_dropout=model_spec.adapter_spec.dropout,
                init_lora_weights=model_spec.adapter_spec.init == "zero",
            ))
        else:
            # OpenAI/Anthropic API クライアント
            self.client = create_api_client(model_spec.agent_llm)

    def generate(self, prompt: str, purpose: str, adapter_node_id: str | None = None,
                 temperature: float | None = None, max_tokens: int | None = None) -> str:
        """
        LLM推論を実行し、結果をログに記録する。

        Args:
            prompt: 入力テキスト
            purpose: 呼び出し目的（"branch_generation", "experiment_code", "paper_section", etc.）
            adapter_node_id: 使用するLoRAアダプタのID（local時のみ）
            temperature: 温度パラメータ（None=ModelSpec既定値）
            max_tokens: 最大生成トークン数（None=ModelSpec既定値）
        """
        pass

    def load_adapter(self, adapter_node_id: str):
        """指定のLoRAアダプタをロード（materialize してから set_adapter）"""
        pass

    def get_log_probs(self, prompt: str, response: str) -> float:
        """PPO用：response の対数確率を計算（local時のみ）"""
        pass
```

---

## 22. 実装手順書（Implementation Guide）

> **このセクションは実装者（Claude等のAIエージェント）向けの具体的な作業指示書である。**
> §20のMVP優先順位に従い、各ステップで「何を作り」「何をテストし」「何が完了条件か」を明示する。

### 22.1 実装の大原則

```text
1. ボトムアップ: ユーティリティ → Specモデル → 各Phase モジュール → CLI → 統合テスト
2. 各ステップ完了後に必ずテストを書いて通すこと（Red-Green サイクル）
3. モックファースト: 外部API・LLM・GPU を必要とする部分はモックで先にテストを通す
4. 1ファイル1責務: 1つのモジュールが複数の責務を持たないこと
5. 型ヒント必須: 全関数に引数・戻り値の型ヒントをつけること
6. passは禁止: 本書の擬似コードにある pass は実装時に必ず実コードに置き換えること
```

### 22.2 Step 0: プロジェクトブートストラップ

**作業内容**: プロジェクトの骨格を作成する。

```bash
# 実行するコマンド
mkdir -p src/sera/{specs,phase0/api_clients,phase1,search,execution,evaluation,learning,lineage,paper,agent,utils}
mkdir -p tests/{test_specs,test_phase0,test_search,test_evaluation,test_lineage,test_paper,test_cli}
touch src/sera/__init__.py
touch src/sera/{cli,specs/__init__,phase0/__init__,phase0/api_clients/__init__,phase1/__init__,search/__init__,execution/__init__,evaluation/__init__,learning/__init__,lineage/__init__,paper/__init__,agent/__init__,utils/__init__}.py
touch tests/__init__.py tests/conftest.py
```

**pyproject.toml を作成**（§18.2 の内容をそのまま使用）:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sera"
version = "0.1.0"
description = "Self-Evolving Research Agent"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "trl>=0.8.0",
    "safetensors>=0.4.0",
    "typer>=0.12.0",
    "pyyaml>=6.0",
    "httpx>=0.27.0",
    "tenacity>=8.2.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "graphviz>=0.20.0",
    "numpy>=1.26.0",
    "pydantic>=2.6.0",
    "rich>=13.7.0",
    "structlog>=24.1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "pytest-mock>=3.12", "respx>=0.21", "ruff>=0.3"]
slurm = ["submitit>=1.5.0"]
docker = ["docker>=7.0.0"]

[project.scripts]
sera = "sera.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/sera"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
target-version = "py311"
line-length = 120
```

**tests/conftest.py の雛形**:
```python
import pytest
from pathlib import Path
import tempfile
import json
import yaml

@pytest.fixture
def tmp_workspace(tmp_path):
    """一時的な sera_workspace ディレクトリを作成"""
    dirs = ["specs", "related_work/results", "related_work/teacher_papers",
            "lineage/nodes", "runs", "logs", "checkpoints", "outputs/best",
            "paper/figures", "docs/modules"]
    for d in dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path

@pytest.fixture
def sample_input1():
    """テスト用 Input-1"""
    return {
        "version": 1,
        "data": {"description": "UCI Iris dataset", "location": "./data/iris.csv", "format": "csv", "size_hint": "small(<1GB)"},
        "domain": {"field": "ML", "subfield": "classification"},
        "task": {"brief": "Classify iris species", "type": "prediction"},
        "goal": {"objective": "maximize accuracy", "direction": "maximize", "baseline": "0.95"},
        "constraints": [{"name": "inference_time_ms", "type": "le", "threshold": 100}],
        "notes": "",
    }

@pytest.fixture
def mock_llm_response():
    """LLM応答のモック生成器"""
    def _mock(content: str):
        return content
    return _mock
```

**完了条件**:
- `pip install -e ".[dev]"` が成功する
- `python -c "import sera"` がエラーなく通る
- `pytest` が 0 テスト 0 エラーで通る

#### 22.2.1 GPU ノードでの環境セットアップ（SLURM クラスタ）

> **重要**: ログインノード（GPU なし）で `pip install` を実行すると、PyTorch が CUDA ランタイムを検出できず `torch.cuda.is_available() = False` になる。local LLM を GPU で実行するには、**必ず GPU のある計算ノード上**でインストールを行うこと。

**セットアップスクリプト**: `scripts/setup_env.sh`

このスクリプトは以下を自動で行う:
1. GPU の存在と CUDA バージョンを検出
2. `.venv` を作成
3. 検出された CUDA バージョンに対応する PyTorch ホイールを `--index-url` 指定でインストール
4. `sera[dev,slurm]` をインストール
5. `torch.cuda.is_available() = True` であることを検証

```bash
# GPU ノード上でセットアップ実行
srun --partition=<gpu-partition> --time=01:00:00 bash scripts/setup_env.sh

# 以後は .venv を activate して使用
source .venv/bin/activate
```

**研究実行ジョブスクリプト**: `scripts/run_research.sh`

```bash
# SLURM ジョブとして研究を実行
sbatch scripts/run_research.sh

# 中断した研究の再開
sbatch scripts/run_research.sh --resume
```

**GPU が不要な操作**（ログインノードで実行可能）:
- `sera init` / `sera phase0-related-work` / `sera freeze-specs` / `sera status` / `sera validate-specs`
- `pytest -m "not gpu" tests/`

**GPU が必要な操作**（計算ノードで実行すること）:
- `sera research`（local LLM 使用時）
- `sera generate-paper` / `sera evaluate-paper`（local LLM 使用時）

> **注意**: `ModelSpec.agent_llm.provider` が `"openai"` や `"anthropic"` の場合は API 経由のため GPU 不要。

---

### 22.3 Step 1: ユーティリティモジュール（`src/sera/utils/`）

**作業順序と各ファイルの実装内容**:

#### 22.3.1 `utils/seed.py`
```python
"""実装すべき関数"""
def set_global_seed(seed: int) -> None:
    """np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all, random.seed を一括設定"""

def get_seed_for_node(base_seed: int, node_id: str, repeat_idx: int) -> int:
    """ノードIDと反復インデックスから決定論的にseedを導出。hash(node_id + repeat_idx) % 2**31"""
```

#### 22.3.2 `utils/hashing.py`
```python
"""実装すべき関数"""
def compute_spec_hash(spec_dict: dict) -> str:
    """dictをcanonical JSON化しSHA-256を計算。'sha256:xxxx' 形式で返す"""

def compute_adapter_spec_hash(adapter_spec: dict) -> str:
    """type+target_modules+target_layers+rank+alpha からハッシュ。形状契約の同一性判定に使用"""

def verify_spec_hash(spec_path: Path, lock_path: Path) -> bool:
    """spec_path のハッシュと lock_path の記録を比較"""
```

#### 22.3.3 `utils/logging.py`
```python
"""実装すべき関数・クラス"""
class JsonlLogger:
    """指定パスに JSONL 形式でログを追記するロガー"""
    def __init__(self, path: Path): ...
    def log(self, event: dict) -> None: ...  # timestamp 自動付与、json.dumps + '\n' で追記

def setup_structlog() -> None:
    """structlog の設定。Rich + JSONL 出力"""
```

#### 22.3.4 `utils/checkpoint.py`
```python
"""実装すべき関数"""
def save_checkpoint(state: dict, checkpoint_dir: Path, step: int) -> Path: ...
def load_latest_checkpoint(checkpoint_dir: Path) -> dict | None: ...
```

**テスト**: `tests/test_utils/` に各関数の単体テスト。hashing はラウンドトリップ、seed は決定論性を検証。

---

### 22.4 Step 2: Pydantic Spec モデル（`src/sera/specs/`）

**全 Spec を Pydantic v2 BaseModel として定義する。§3〜§5 の YAML スキーマをそのまま Python クラスに変換する。**

**実装順序**（依存関係順）:

```text
1. input1.py          — Input1Model（§3.1 のフィールド）
2. related_work_spec.py — Paper, Cluster, BaselineCandidate, RelatedWorkSpecModel（§4.4.1）
3. paper_spec.py       — SectionRequirement, FigureRequirement, PaperSpecModel（§4.4.2）
4. paper_score_spec.py — Criterion, PaperScoreSpecModel（§4.4.3）
5. teacher_paper_set.py — TeacherPaper, TeacherPaperSetModel（§4.4.4）
6. problem_spec.py     — Constraint, Variable, ProblemSpecModel（§5.5）
7. model_spec.py       — BaseModelConfig, AdapterSpec, ModelSpecModel（§5.3）
8. resource_spec.py    — ComputeConfig, NetworkConfig, ResourceSpecModel（§5.6）
9. plan_spec.py        — BranchingOp, RewardConfig, PlanSpecModel（§5.7）
10. execution_spec.py  — SearchConfig, EvaluationConfig, LearningConfig, ... ExecutionSpecModel（§5.4）
```

**実装パターン（全Specで共通）**:

```python
# 例: execution_spec.py
from pydantic import BaseModel, model_validator
from sera.utils.hashing import compute_spec_hash

class SearchConfig(BaseModel):
    strategy: str = "best_first"
    priority_rule: str = "epsilon_constraint_lcb"
    lambda_cost: float = 0.1
    beta_exploration: float = 0.05
    max_nodes: int = 100
    max_depth: int = 10
    branch_factor: int = 3
    initial_root_children: int = 5

class EvaluationConfig(BaseModel):
    repeats: int = 3
    lcb_coef: float = 1.96
    sequential_eval: bool = True
    sequential_eval_initial: int = 1
    sequential_eval_topk: int = 5
    bootstrap: bool = False
    bootstrap_samples: int = 1000

# ... 他の Config も同様 ...

class ExecutionSpecModel(BaseModel):
    search: SearchConfig = SearchConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    learning: LearningConfig = LearningConfig()
    lora_runtime: LoraRuntimeConfig = LoraRuntimeConfig()
    pruning: PruningConfig = PruningConfig()
    termination: TerminationConfig = TerminationConfig()
    paper: PaperConfig = PaperConfig()

    def compute_hash(self) -> str:
        return compute_spec_hash(self.model_dump())

    # YAML <-> Model 変換
    @classmethod
    def from_yaml(cls, path: Path) -> "ExecutionSpecModel":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("execution_spec", data))

    def to_yaml(self, path: Path) -> None:
        with open(path, "w") as f:
            yaml.dump({"execution_spec": self.model_dump()}, f, default_flow_style=False)
```

**共通ユーティリティ** — `specs/__init__.py` に以下を実装:
```python
class AllSpecs:
    """全Specを束ねるコンテナ。Phase 1 完了後に構築される"""
    input1: Input1Model
    related_work: RelatedWorkSpecModel
    paper_spec: PaperSpecModel
    paper_score_spec: PaperScoreSpecModel
    teacher_papers: TeacherPaperSetModel
    problem_spec: ProblemSpecModel
    model_spec: ModelSpecModel
    resource_spec: ResourceSpecModel
    plan_spec: PlanSpecModel
    execution_spec: ExecutionSpecModel

    @classmethod
    def load_from_dir(cls, specs_dir: Path) -> "AllSpecs": ...
    def save_to_dir(cls, specs_dir: Path) -> None: ...
```

**テスト**: `tests/test_specs/` — 各 Spec について:
- 既定値でインスタンス化できること
- YAML ラウンドトリップ（to_yaml → from_yaml で同一）
- 不正値で ValidationError が発生すること
- ExecutionSpec のハッシュが決定論的であること

**完了条件**: 全10 Spec モデルが定義され、テストが通る。

---

### 22.5 Step 3: Phase 0 — 先行研究エンジン（`src/sera/phase0/`）

**実装順序**:

```text
1. api_clients/base.py         — BaseScholarClient（ABC）
2. api_clients/semantic_scholar.py — SemanticScholarClient
3. api_clients/crossref.py     — CrossRefClient
4. api_clients/arxiv.py        — ArxivClient
5. api_clients/web_search.py   — WebSearchClient（SerpAPI）
6. ranking.py                  — citation_norm, relevance_score, rank_papers
7. clustering.py               — cluster_papers（LLMベース）
8. related_work_engine.py      — RelatedWorkEngine（統合エントリポイント）
```

**APIクライアント共通パターン**:
```python
# api_clients/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@dataclass
class PaperResult:
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str
    abstract: str
    citation_count: int
    url: str
    doi: str
    arxiv_id: str
    source_api: str

class BaseScholarClient(ABC):
    @abstractmethod
    async def search(self, query: str, limit: int, year_from: int | None) -> list[PaperResult]: ...

    @abstractmethod
    async def get_references(self, paper_id: str, limit: int) -> list[PaperResult]: ...

    @abstractmethod
    async def get_citations(self, paper_id: str, limit: int) -> list[PaperResult]: ...
```

```python
# api_clients/semantic_scholar.py — 具体実装の構造
class SemanticScholarClient(BaseScholarClient):
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "paperId,title,abstract,year,citationCount,authors,venue,externalIds,url"

    def __init__(self, api_key: str | None = None):
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(base_url=self.BASE_URL, headers=headers, timeout=30.0)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=60),
           retry=retry_if_exception_type(httpx.HTTPStatusError))
    async def search(self, query: str, limit: int = 20, year_from: int | None = None) -> list[PaperResult]:
        params = {"query": query, "limit": limit, "fields": self.FIELDS}
        if year_from:
            params["year"] = f"{year_from}-"
        resp = await self._client.get("/paper/search", params=params)
        resp.raise_for_status()
        # パース処理...
```

**related_work_engine.py**（統合）:
```python
class RelatedWorkEngine:
    """
    Phase 0 の統合エントリポイント。
    1. Input-1 → LLM でクエリ生成
    2. API優先順位に従い論文取得
    3. ランキング・クラスタリング
    4. RelatedWorkSpec, PaperSpec, PaperScoreSpec, TeacherPaperSet を生成
    5. 全クエリを queries.jsonl に記録
    """
    def __init__(self, clients: list[BaseScholarClient], agent_llm: AgentLLM, logger: JsonlLogger): ...

    async def run(self, input1: Input1Model, config: Phase0Config) -> Phase0Output:
        # 1. クエリ生成
        queries = await self._build_queries(input1)
        # 2. 各クエリでAPI検索（フォールバック付き）
        papers = await self._search_with_fallback(queries, config)
        # 3. ランキング
        ranked = rank_papers(papers, config.ranking_weight)
        # 4. クラスタリング
        clusters = await cluster_papers(ranked[:config.top_k_papers], self.agent_llm)
        # 5. Spec 生成
        return Phase0Output(
            related_work_spec=self._build_related_work_spec(ranked, clusters),
            paper_spec=await self._build_paper_spec(ranked, input1),
            paper_score_spec=await self._build_paper_score_spec(ranked, input1),
            teacher_paper_set=self._build_teacher_set(ranked, config),
        )

    async def _search_with_fallback(self, queries, config):
        """§4.2.1 の優先順位に従い、APIフォールバック"""
        all_papers = []
        for query in queries:
            for client in self.clients:  # 優先順位順
                try:
                    results = await client.search(query.text, limit=config.top_k_papers)
                    # queries.jsonl にログ
                    self.logger.log({...})
                    all_papers.extend(results)
                    if len(all_papers) >= config.top_k_papers:
                        break
                except Exception as e:
                    self.logger.log({"error": str(e), ...})
                    continue  # 次のAPIへフォールバック
        return all_papers
```

**テスト方法**:
- `respx` で HTTP レスポンスをモック（各APIクライアント）
- `mock_llm_response` でクエリ生成・クラスタリングの LLM 応答をモック
- `related_work_engine` は統合テスト（全モック）

**完了条件**:
- 各 API クライアントがモックで正常応答を返す
- フォールバック（API1失敗→API2成功）が動作する
- queries.jsonl にクエリログが記録される
- Phase0Output から全4つの Spec が生成される

---

### 22.6 Step 4: Phase 1 — Spec確定＋凍結（`src/sera/phase1/`）

**実装ファイル**:

```text
1. spec_builder.py  — LLM で ProblemSpec, PlanSpec を草案生成
2. spec_freezer.py  — ExecutionSpec ハッシュ計算・lock ファイル書き出し・検証
```

```python
# spec_builder.py
class SpecBuilder:
    def __init__(self, agent_llm: AgentLLM): ...

    async def build_problem_spec(self, input1: Input1Model, related_work: RelatedWorkSpecModel) -> ProblemSpecModel:
        """
        LLM に input1 + related_work を与え、ProblemSpec の JSON を生成させる。
        生成結果を ProblemSpecModel(**json_output) でバリデーション。
        バリデーション失敗時は最大3回リトライ（エラーメッセージをLLMに返して修正させる）。
        """

    async def build_plan_spec(self, input1: Input1Model, problem_spec: ProblemSpecModel) -> PlanSpecModel:
        """同様にLLMでPlanSpecを生成"""

# spec_freezer.py
class SpecFreezer:
    def freeze(self, specs: AllSpecs, specs_dir: Path) -> None:
        """
        1. 全 Spec を specs_dir に YAML で保存
        2. ExecutionSpec のハッシュを計算
        3. execution_spec.yaml.lock に書き出し
        """

    def verify(self, specs_dir: Path) -> bool:
        """execution_spec.yaml と .lock のハッシュを比較。不一致は False"""
```

**テスト**: LLMモックでSpec生成→バリデーション通過、ハッシュのラウンドトリップ検証

---

### 22.7 Step 5: AgentLLM（`src/sera/agent/`）

**注意: これは全Phase で横断的に使われるため、早めに実装する。ただし完全実装はPhase 5（PPO）時点。Step 5 では推論機能のみ。**

```text
1. agent_llm.py         — §付録C のインターフェース実装
2. prompt_templates.py  — 各Phase用プロンプトテンプレート（文字列テンプレート集）
```

**agent_llm.py の実装戦略**:

```python
class AgentLLM:
    def __init__(self, model_spec: ModelSpecModel, resource_spec: ResourceSpecModel, log_path: Path):
        self.model_spec = model_spec
        self.logger = JsonlLogger(log_path)
        self._provider = model_spec.agent_llm.provider

        if self._provider == "local":
            # transformers + peft でモデルロード
            # ※ GPU がない環境ではスキップ可能に（テスト用）
            self._init_local_model()
        elif self._provider == "openai":
            import openai
            self._client = openai.AsyncOpenAI(api_key=os.environ[resource_spec.api_keys.openai])
        elif self._provider == "anthropic":
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=os.environ[resource_spec.api_keys.anthropic])

    async def generate(self, prompt: str, purpose: str, ...) -> str:
        # 1. provider に応じて推論
        # 2. agent_llm_log.jsonl にログ
        # 3. 結果を返す

    # load_adapter, get_log_probs は Step 9（PPO実装時）に追加
```

**prompt_templates.py**:
```python
"""
各Phaseで使うプロンプトテンプレートを定数として定義。
str.format() または jinja2 で変数埋め込み。

定義すべきテンプレート:
- QUERY_GENERATION_PROMPT      — Phase 0: Input-1 → 検索クエリ生成
- PAPER_CLUSTERING_PROMPT      — Phase 0: 論文クラスタリング
- RELEVANCE_SCORING_PROMPT     — Phase 0: 関連度スコアリング
- SPEC_GENERATION_PROMPT       — Phase 1: ProblemSpec/PlanSpec 生成
- DRAFT_PROMPT                 — Phase 2: draft オペレータ（§6.5.1 のプロンプト）
- DEBUG_PROMPT                 — Phase 2: debug オペレータ（§6.5.2 のプロンプト）
- IMPROVE_PROMPT               — Phase 2: improve オペレータ（§6.5.3 のプロンプト）
- EXPERIMENT_CODE_PROMPT       — Phase 3: 実験コード生成
- PAPER_OUTLINE_PROMPT         — Phase 7: アウトライン生成
- PAPER_FULL_GENERATION_PROMPT — Phase 7: 論文全体1パス生成（ステップ5b）
- PAPER_WRITEUP_REFLECTION_PROMPT — Phase 7: ライティング内反省（ステップ5c）
- CITATION_SEARCH_PROMPT       — Phase 7: 不足引用特定+検索クエリ生成（ステップ3）
- CITATION_SELECT_PROMPT       — Phase 7: 検索結果から関連論文選択（ステップ3）
- PLOT_AGGREGATION_PROMPT      — Phase 7: 図集約スクリプト生成（ステップ2）
- VLM_FIGURE_DESCRIPTION_PROMPT — Phase 7: VLM図記述生成（ステップ4）
- VLM_FIGURE_CAPTION_REVIEW_PROMPT — Phase 7: VLM図・キャプション整合性レビュー（ステップ5c-iv）
- VLM_DUPLICATE_DETECTION_PROMPT — Phase 7: VLM重複図検出（ステップ5c-v）
- PAPER_EVALUATION_PROMPT      — Phase 8: 論文評価（§12.1 のプロンプト）
- REVIEWER_REFLECTION_PROMPT   — Phase 8: レビュアー反省ループ（§12.1 ステップ1d）
- META_REVIEW_PROMPT           — Phase 8: Area Chair メタレビュー（§12.1 ステップ2）
- PAPER_REVISION_PROMPT        — Phase 8: 論文修正指示

各テンプレートは {変数名} プレースホルダーを含み、呼び出し側で .format(**kwargs) する。
"""
```

**テスト**: OpenAI/Anthropic クライアントはモック。local は conftest で `skip_if_no_gpu` マーカー。

---

### 22.8 Step 6: Phase 2 — 探索木（`src/sera/search/`）

```text
1. search_node.py       — SearchNode データクラス（§6.2）
2. priority.py          — compute_priority, compute_exploration_bonus（§6.3）
3. validation.py        — validate_experiment_config（§6.6.1）
4. tree_ops.py          — TreeOps クラス: draft / debug / improve 3オペレータ（§6.5）
5. search_manager.py    — SearchManager: select_next_node + メインループ（§6.4, §6.11）
```

**tree_ops.py の重要ポイント（AIDE参考3オペレータ）**:

```python
class TreeOps:
    """AIDE参考の3オペレータを提供する（§6.5）"""

    def __init__(self, specs: AllSpecs, agent_llm: AgentLLM, rng):
        self.specs = specs
        self.agent_llm = agent_llm
        self.rng = rng

    def draft(self, n: int) -> list[SearchNode]:
        """§6.5.1: 新規アプローチの起草。親なし。
        ルート時: baseline/open_problem/novel を n//3 ずつ配分。
        再draft時: 既存ノード一覧を提示し異なるアプローチを要求。"""

    def debug(self, failed_node: SearchNode) -> SearchNode:
        """§6.5.2: 失敗ノードの修復。experiment_config は変更せずコードのみ修正。
        debug_depth をインクリメント。max_debug_depth 超過なら呼ばれない（§6.4で制御）。"""

    def improve(self, parent: SearchNode, all_nodes: dict,
                n_children: int) -> list[SearchNode]:
        """§6.5.3: 原子的改善。1子=1変数変更。
        validate_experiment_config でバリデーション。"""
```

**search_manager.py の重要ポイント**:

```python
import heapq

class SearchManager:
    def __init__(self, specs: AllSpecs, agent_llm: AgentLLM, executor: Executor,
                 evaluator: Evaluator, ppo_trainer: PPOTrainer | None,
                 lineage_manager: LineageManager, tree_ops: TreeOps,
                 logger: JsonlLogger):
        self.open_list: list[SearchNode] = []
        self.closed_set: set[str] = set()
        self.all_nodes: dict[str, SearchNode] = {}
        self.best_node: SearchNode | None = None
        self.tree_ops = tree_ops
        self.step = 0
        ...

    def run(self) -> SearchNode:
        """§6.11 の research_loop を実装。チェックポイント保存・復帰対応。
        1. tree_ops.draft() でルートノード生成
        2. select_next_node() でオペレータ自動選択（§6.4）
        3. evaluate / debug / draft / improve を分岐実行
        4. PPO更新・剪定を適宜実行"""

    def select_next_node(self) -> tuple[SearchNode | None, str]:
        """§6.4: 状態に基づくオペレータ自動選択。
        pending → 'evaluate', failed → 'debug', 多様性不足 → 'draft', else → 'improve'"""

    def _should_terminate(self) -> bool:
        """§5.4 termination 条件を全てチェック"""
```

**テスト**:
- `search_node.py`: データクラスの生成・シリアライズ
- `priority.py`: 既知の入力に対する期待値の一致
- `validation.py`: ホワイトリスト検証（許可キー/禁止キー/範囲外の各パターン）
- `tree_ops.py`: モックLLMで draft/debug/improve 各オペレータの単体テスト
- `search_manager.py`: max_nodes=5 の小規模探索が完走すること（全モック、3オペレータ遷移確認）

---

### 22.9 Step 7: Phase 3 — 実験実行（`src/sera/execution/`）

```text
1. executor.py              — Executor ABC + RunResult（§7.3）
2. local_executor.py        — LocalExecutor（subprocess.Popen）
3. experiment_generator.py  — ExperimentGenerator（LLMによるコード生成）
4. slurm_executor.py        — SlurmExecutor（MVP後、スタブのみ先に作成）
5. docker_executor.py       — DockerExecutor（MVP後、スタブのみ先に作成）
```

**local_executor.py の実装ポイント**:

```python
import subprocess
import time
import signal

class LocalExecutor(Executor):
    def __init__(self, work_dir: Path, resource_spec: ResourceSpecModel):
        self.work_dir = work_dir
        self.timeout = resource_spec.sandbox.experiment_timeout_sec
        self.memory_limit = resource_spec.sandbox.experiment_memory_limit_gb

    def run(self, node_id: str, script_path: Path, seed: int, timeout_sec: int | None = None) -> RunResult:
        run_dir = self.work_dir / "runs" / node_id
        run_dir.mkdir(parents=True, exist_ok=True)
        timeout = timeout_sec or self.timeout

        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"

        start = time.monotonic()
        with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
            try:
                proc = subprocess.Popen(
                    ["python", str(script_path), "--seed", str(seed), "--output-dir", str(run_dir)],
                    stdout=out, stderr=err,
                    env={**os.environ, "SERA_NODE_ID": node_id, "SERA_SEED": str(seed)},
                )
                exit_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return RunResult(node_id=node_id, success=False, exit_code=-9, ...)
            except MemoryError:
                return RunResult(node_id=node_id, success=False, exit_code=-7, ...)

        wall_time = time.monotonic() - start
        metrics_path = run_dir / "metrics.json" if (run_dir / "metrics.json").exists() else None

        return RunResult(
            node_id=node_id, success=(exit_code == 0), exit_code=exit_code,
            stdout_path=stdout_path, stderr_path=stderr_path,
            metrics_path=metrics_path, artifacts_dir=run_dir / "artifacts",
            wall_time_sec=wall_time, seed=seed,
        )
```

**テスト**: 簡単な Python スクリプト（`echo '{"primary":{"name":"acc","value":0.9}}' > metrics.json`）を実行し、RunResult を検証。タイムアウトテストも。

**slurm_executor.py の実装ポイント**:

SlurmExecutor は `submitit` ライブラリを使用して SLURM クラスタにジョブを投入する。
ResourceSpecModel の SlurmConfig（partition, account, time_limit, modules, sbatch_extra）を読み込む。

```python
import time
from pathlib import Path
from sera.execution.executor import Executor, RunResult
from sera.specs.resource_spec import SlurmConfig

class SlurmExecutor(Executor):
    """submitit 経由で SLURM ジョブを投入・完了待ち・結果収集"""

    def __init__(self, work_dir: Path, slurm_config: SlurmConfig, python_executable: str = "python"):
        self.work_dir = Path(work_dir)
        self.slurm_config = slurm_config
        self.python_executable = python_executable

    def run(self, node_id: str, script_path: Path, seed: int, timeout_sec: int | None = None) -> RunResult:
        """
        1. runs/<node_id>/ ディレクトリを作成
        2. submitit.AutoExecutor を設定（partition, account, time_limit, modules, sbatch_extra）
        3. ラッパー関数（subprocess で実験スクリプトを実行）をジョブとして投入
        4. job.results() でブロッキング完了待ち（timeout_sec 超過時はジョブキャンセル）
        5. stdout/stderr をジョブログから run_dir にコピー
        6. metrics.json の有無をチェック
        7. OOM判定: sacct の MaxRSS / State=OUT_OF_MEMORY、または stderr 内のパターン検出
        8. RunResult を返却
        """
        pass
```

実装要件:
- `submitit` が未インストールの場合は `ImportError` を明確なメッセージ付きで送出
- `slurm_config.modules` の各モジュールは `module load` コマンドでロード（sbatch スクリプトの前処理として）
- `slurm_config.sbatch_extra` は submitit の `additional_parameters` として渡す
- タイムアウト: `timeout_sec` を SLURM の wall-time limit と独立に管理。Python 側のポーリングループで `timeout_sec` 超過時に `scancel` でジョブをキャンセルし `exit_code=-9` を返す
- OOM検知: SLURM ジョブステート `OUT_OF_MEMORY` またはexit code 137、stderr 内の OOM パターンで判定し `exit_code=-7` を返す
- ポーリング間隔: 10秒（デフォルト）
- ジョブ名: `sera-{node_id[:8]}` で識別可能にする

**テスト**: submitit をモックし、正常完了・タイムアウト・OOM・submitit未インストールのケースを検証。

---

### 22.10 Step 8: Phase 4 — 統計評価（`src/sera/evaluation/`）

```text
1. evaluator.py              — Evaluator ABC
2. statistical_evaluator.py  — StatisticalEvaluator（§8.1〜8.2 の実装）
3. feasibility.py            — check_feasibility（§8.3）
```

**statistical_evaluator.py**:
```python
class StatisticalEvaluator:
    """§8.1〜8.2 の実装。update_stats, evaluate_node_sequential, evaluate_node_full"""

    def __init__(self, executor: Executor, exec_spec: ExecutionSpecModel, problem_spec: ProblemSpecModel): ...

    def evaluate_initial(self, node: SearchNode) -> None:
        """sequential_eval_initial 回実行して暫定統計を計算"""

    def evaluate_full(self, node: SearchNode) -> None:
        """repeats まで追加実行して最終統計を計算"""

    def is_topk(self, node: SearchNode, all_nodes: list[SearchNode]) -> bool:
        """LCBでソートしてTop-kに入るか判定"""

    @staticmethod
    def update_stats(node: SearchNode, lcb_coef: float) -> None:
        """§8.2 の update_stats をそのまま実装"""
```

**テスト**: 既知の値リスト（[0.7, 0.8, 0.9]）に対する μ, SE, LCB の計算結果を検証。

---

### 22.11 Step 9: Phase 5-6 — PPO + LoRA系譜（`src/sera/learning/` + `src/sera/lineage/`）

**これが最も複雑なステップ。GPU が必要な部分はモックテスト優先。**

```text
learning/
  1. rollout.py       — PPORollout データクラス
  2. reward.py        — compute_reward（§9.2）
  3. ppo_trainer.py   — PPOTrainer（§9.3）

lineage/
  4. cache.py         — LRUCache（collections.OrderedDict ベース）
  5. lineage_manager.py — materialize, save_delta, squash（§10.2〜10.3）
  6. pruner.py        — Pareto剪定、LCB閾値剪定、予算剪定（§10.4）
```

**ppo_trainer.py の実装戦略**:

```python
class PPOTrainer:
    """
    trl ライブラリの PPOTrainer をラップ。
    ただし trl の API は頻繁に変わるため、以下の自前実装も許容:

    自前実装の場合の手順:
    1. rollouts から (prompt, response, reward) のバッチを構成
    2. model.generate() で response を再生成し、new_log_prob を取得
    3. old_log_prob との比率 r = exp(new_log_prob - old_log_prob) を計算
    4. GAE で advantage を計算
    5. PPO クリッピング損失を計算
    6. optimizer.step()（LoRA パラメータのみ）
    7. KL 制御

    trl 使用の場合:
    - trl.PPOConfig + trl.PPOTrainer を使い、peft モデルをそのまま渡す
    - trl が LoRA パラメータのみを更新することを確認
    """
```

**lineage_manager.py — materialize のテスト戦略**:
```python
# テスト: GPU不要（CPU上で小さなテンソルで検証）
def test_materialize_chain():
    """
    root(zeros) → child1(+0.1) → child2(+0.2) の3段系譜で、
    materialize(child2) == zeros + 0.1 + 0.2 を検証
    """

def test_materialize_with_snapshot():
    """
    root → child1 → child2(snapshot) → child3(+delta) で、
    materialize(child3) == snapshot + delta を検証（root/child1 は参照しない）
    """

def test_squash_creates_snapshot():
    """squash_depth=2 のとき、depth=2 のノードにスナップショットが生成されることを検証"""
```

---

### 22.12 Step 10: Phase 7-8 — 論文生成・評価（`src/sera/paper/`）

```text
1. evidence_store.py      — EvidenceStore（§11.2）
2. figure_generator.py    — FigureGenerator（matplotlib/seaborn/graphviz）
3. vlm_reviewer.py        — VLMReviewer（§11.4：図記述・キャプションレビュー・重複検出）
4. citation_searcher.py   — CitationSearcher（§11.5：Semantic Scholar自動引用検索ループ）
5. paper_composer.py      — PaperComposer（§11.3：6ステップ + ライティング内反省ループ）
6. paper_evaluator.py     — PaperEvaluator + PaperScoreResult（§12.1：アンサンブル+レビュアー反省）
```

**figure_generator.py の実装**:
```python
class FigureGenerator:
    def __init__(self, output_dir: Path, max_figures: int = 12, dpi: int = 300): ...

    def ci_bar_chart(self, nodes: list[SearchNode], output_name: str) -> Path:
        """各ノードの μ ± CI を棒グラフで描画。matplotlib.errorbar 使用"""

    def convergence_curve(self, data: list[tuple[int, float]], output_name: str) -> Path:
        """step vs best_lcb の折れ線グラフ"""

    def search_tree(self, nodes: dict[str, SearchNode], top_n: int, output_name: str) -> Path:
        """graphviz でツリー描画。LCB値をノードラベルに表示"""

    def ablation_table(self, data: dict, output_name: str) -> Path:
        """アブレーション結果をヒートマップまたはテーブル画像で出力"""

    def aggregate_plots(self, evidence: EvidenceStore, agent_llm: AgentLLM,
                        n_reflections: int = 5) -> list[Path]:
        """
        LLMが実験結果を統合した追加集約図スクリプトを生成。
        反省ループで改善（最大 n_reflections 回）。
        AI-Scientist-v2 の plot aggregation に相当。
        """
```

**vlm_reviewer.py の実装**:
```python
class VLMReviewer:
    """§11.4 参照。VLM による図の視覚的レビュー。"""
    # ModelSpec.vlm が null の場合、PaperComposer は本クラスを None として扱う
    # describe_figures(), review_figure_caption_refs(), detect_duplicate_figures() を実装
```

**citation_searcher.py の実装**:
```python
class CitationSearcher:
    """§11.5 参照。Semantic Scholar API を使った自動引用検索ループ。"""
    # Phase 0 の SemanticScholarClient を再利用
    # 各ラウンドを citation_search_log.jsonl に記録（再現性）
```

**paper_evaluator.py のアンサンブル実装**:
```python
class PaperEvaluator:
    """§12.1 参照。以下の機能を実装:
    - 単体レビュー生成（Few-shot + bias_mode 対応）
    - レビュアー反省ループ（num_reviewer_reflections 回）
    - アンサンブル集約（num_reviews_ensemble > 1 の場合）
    - メタレビュー生成（Area Chair モード）
    """
```

**テスト**:
- ダミーデータで図が生成され、PNG ファイルが存在することを検証（画像内容は検証不要）
- CitationSearcher: Semantic Scholar API をモックし、検索ループが正しく動作することを検証
- VLMReviewer: VLM API をモックし、図記述・レビュー・重複検出の出力形式を検証
- PaperEvaluator: LLM をモックし、アンサンブル集約・メタレビュー・反省ループのフローを検証
- PaperComposer: ライティング内反省ループが最大回数で停止することを検証

---

### 22.13 Step 11: CLI（`src/sera/cli.py`）

**Typer で全コマンドを定義。各コマンドは対応するモジュールを呼び出すだけの薄いラッパー。**

```python
import typer
from pathlib import Path

app = typer.Typer(name="sera", help="Self-Evolving Research Agent")

@app.command()
def init(input1_path: Path, work_dir: Path = Path("./sera_workspace")):
    """Input-1 を読み込み、workspace を初期化"""
    # 1. work_dir 作成（§14 のディレクトリ構造）
    # 2. input1.yaml を specs/ にコピー
    # 3. 成功メッセージ

@app.command()
def phase0_related_work(
    work_dir: Path = Path("./sera_workspace"),
    topk: int = 10,
    teacher_papers: int = 5,
    citation_depth: int = 1,
    years_bias: int = 5,
    api_priority: str = "semantic_scholar,crossref,arxiv,web",
):
    """Phase 0: 先行研究収集"""
    # 1. Input-1 ロード
    # 2. AgentLLM 初期化
    # 3. RelatedWorkEngine.run()
    # 4. 結果を specs/ に保存

@app.command()
def freeze_specs(work_dir: Path = Path("./sera_workspace"), auto: bool = False):
    """Phase 1: 全Spec確定、ExecutionSpec固定"""
    # 1. Phase 0 出力ロード
    # 2. SpecBuilder で ProblemSpec, PlanSpec 生成
    # 3. auto=false なら specs/ を開いてユーザ確認待ち
    # 4. SpecFreezer.freeze()

@app.command()
def research(work_dir: Path = Path("./sera_workspace"), resume: bool = False):
    """Phase 2-6: 研究ループ"""
    # 1. AllSpecs ロード
    # 2. ExecutionSpec ハッシュ検証（失敗なら exit(2)）
    # 3. resume なら checkpoint ロード
    # 4. SearchManager.run()
    # 5. export_best 自動実行

@app.command()
def export_best(work_dir: Path = Path("./sera_workspace")):
    """best成果物を outputs/best/ に集約"""

@app.command()
def generate_paper(work_dir: Path = Path("./sera_workspace")):
    """Phase 7: 論文生成"""

@app.command()
def evaluate_paper(work_dir: Path = Path("./sera_workspace")):
    """Phase 8: 論文評価・改善ループ"""

@app.command()
def status(work_dir: Path = Path("./sera_workspace")):
    """現在の探索状態サマリ表示"""

@app.command()
def show_node(node_id: str, work_dir: Path = Path("./sera_workspace")):
    """ノード詳細表示"""

@app.command()
def replay(node_id: str, seed: int, work_dir: Path = Path("./sera_workspace")):
    """特定ノードの実験再実行"""

@app.command()
def validate_specs(work_dir: Path = Path("./sera_workspace")):
    """Spec整合性チェック"""
```

**テスト**: `typer.testing.CliRunner` で各コマンドの呼び出しテスト。`sera init` → `sera validate-specs` の最小フロー。

---

### 22.14 Step 12: 統合テスト + docs

**統合テスト**（`tests/test_integration/`）:
```python
def test_full_pipeline_mock():
    """
    全APIとLLMをモックして、以下のフローが完走することを検証:
    1. sera init（サンプル Input-1）
    2. sera phase0-related-work（モックAPI）
    3. sera freeze-specs --auto
    4. sera research（max_nodes=3, repeats=1 の最小設定、モック実験）
    5. sera export-best
    6. sera generate-paper（モックLLM）
    7. sera evaluate-paper（モックLLM）

    検証項目:
    - specs/ に全9ファイル + .lock が存在
    - logs/ に全 JSONL ファイルが存在し、各1エントリ以上
    - outputs/best/ に best_node.json, report.json が存在
    - paper/paper.md が存在し、空でない
    - exit code が全て 0
    """
```

**docs/**:
- §15 の内容に従い、各 .md ファイルを作成
- quickstart.md は実際にコマンドを実行できるチュートリアル形式
- architecture.md は Mermaid 図を含める

---

### 22.15 テスト戦略まとめ

| テスト種別 | 対象 | モック範囲 | テストファイル |
|-----------|------|----------|--------------|
| 単体 | utils/* | なし | tests/test_utils/ |
| 単体 | specs/* | なし | tests/test_specs/ |
| 単体 | evaluation/* | Executor | tests/test_evaluation/ |
| 単体 | lineage/* | なし（CPUテンソル） | tests/test_lineage/ |
| 単体 | search/priority.py | なし | tests/test_search/ |
| モック統合 | phase0/* | HTTP（respx） + LLM | tests/test_phase0/ |
| モック統合 | search/* | LLM + Executor | tests/test_search/ |
| モック統合 | paper/* | LLM | tests/test_paper/ |
| E2E | 全パイプライン | HTTP + LLM + Executor | tests/test_integration/ |
| CLI | cli.py | 全モック | tests/test_cli/ |

**テスト実行コマンド**:
```bash
# 全テスト（GPU不要テストのみ）
pytest -m "not gpu"

# GPU テスト含む
pytest

# カバレッジ
pytest --cov=sera --cov-report=html
```

**conftest.py に追加するマーカー**:
```python
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU")
    config.addinivalue_line("markers", "slow: slow integration tests")
    config.addinivalue_line("markers", "network: requires network access")
```

---

### 22.16 実装時の注意点・よくある落とし穴

```text
1. trl のバージョン依存:
   - trl の PPOTrainer API は頻繁に変わる。import エラーが出たら trl のバージョンを確認
   - 最悪自前PPOで対処（§9.3 の手順に従う）

2. peft と transformers のバージョン互換:
   - peft が新しすぎると transformers と非互換な場合がある
   - pip install 時にバージョン制約を確認

3. safetensors のキー名:
   - peft が保存するキー名と、本書で定義する delta キー名が異なる可能性
   - materialize 時にキー名マッピングが必要になることがある

4. httpx の async:
   - Phase 0 の API クライアントは async で実装するが、CLI から呼ぶ際は asyncio.run() でラップ
   - テストでは pytest-asyncio を使用

5. heapq は min-heap:
   - Best-First は max-priority なので、priority の負値を heapq に入れる
   - または heapq._heapify_max は非公開APIなので使わない

6. Ctrl+C ハンドリング:
   - signal.signal(signal.SIGINT, handler) で登録
   - handler 内で checkpoint 保存してから sys.exit(20)
   - PPO 更新中の中断は次回 resume で再実行

7. ExecutionSpec 改竄検知:
   - research コマンド開始時に必ず verify_spec_hash() を呼ぶ
   - 不一致なら即座に exit(2)

8. LLM JSON パース失敗:
   - LLM の出力が正しい JSON でないことがある
   - json.loads() の前に ```json ... ``` ブロックを抽出する前処理を入れる
   - 失敗時は最大3回リトライ（temperature += 0.1）

9. メモリ管理:
   - LoRA キャッシュ（cache_max_entries）を超えないよう LRU で管理
   - 大きなモデルは 4bit/8bit 量子化を推奨

10. ファイルパスの一貫性:
    - 全てのパスは work_dir からの相対パスとして管理
    - Path オブジェクトを使い、文字列結合は避ける
```

---

### 22.17 実装チェックリスト

実装完了後、以下を全て確認すること：

```text
[ ] pyproject.toml が正しく、pip install -e ".[dev]" が通る
[ ] sera コマンドが実行でき、--help が表示される
[ ] 全 Spec の Pydantic モデルが定義され、YAML ラウンドトリップが通る
[ ] ExecutionSpec のハッシュ固定と検証が動作する
[ ] Phase 0: モック API でフォールバック検索が動作し、queries.jsonl にログが記録される
[ ] Phase 1: LLM モックで ProblemSpec, PlanSpec が生成され、freeze-specs が完了する
[ ] Phase 2: Best-First 探索で子ノードが生成され、優先度でソートされる
[ ] Phase 3: LocalExecutor で実験スクリプトが実行され、metrics.json が出力される
[ ] Phase 4: μ, SE, LCB が正しく計算され、逐次評価で Top-k のみ追加実行される
[ ] Phase 5: PPO 更新で LoRA パラメータのみが更新され、delta が safetensors で保存される
[ ] Phase 6: materialize が正しく累積復元し、squash でスナップショットが生成される
[ ] Phase 6: Pareto 剪定が動作し、保護リストのノードは残る
[ ] Phase 7: PaperComposer が CI 付き図表を含む paper.md を生成する
[ ] Phase 7: 自動引用検索ループが Semantic Scholar API で動作する
[ ] Phase 7: ライティング内反省ループが構文エラー・未使用図等を自己修正する
[ ] Phase 7: VLM が有効な場合、図記述生成・キャプションレビュー・重複図検出が動作する
[ ] Phase 8: PaperEvaluator がアンサンブルレビュー（Few-shot + レビュアー反省ループ）で採点する
[ ] Phase 8: アンサンブル時に Area Chair メタレビューが生成される
[ ] Phase 8: 改善ループが Phase 7（内部反省）と Phase 8（外部評価）の二重構造で回る
[ ] outputs/best/ に best_node.json, adapter.safetensors, report.json が出力される
[ ] logs/ に全 JSONL ファイル（search, eval, ppo, paper, agent_llm）が記録される
[ ] docs/ に §15 の全ファイルが存在する
[ ] pytest が全テスト通過する（GPU不要テストのみでも可）
[ ] sera init → sera phase0-related-work → sera freeze-specs --auto → sera research → sera generate-paper → sera evaluate-paper の一連のフローがモック環境で完走する
```

---

## 23. 多言語実験サポート（Multi-Language Experiment Support）

### 23.1 概要

SERAは実験スクリプトをPython以外の言語（R, Julia, Go, C++, bash等）でも生成・実行できる。
言語設定は `ProblemSpec.language` で指定され、Phase 1で固定される。

### 23.2 LanguageConfig スキーマ

```yaml
# ProblemSpec 内
language:
  name: "python"              # 言語名（プロンプト生成に使用）
  interpreter_command: "python" # インタプリタコマンド
  file_extension: ".py"        # 実験スクリプトの拡張子
  seed_arg_format: "--seed {seed}" # シード引数のフォーマット文字列
  code_block_tag: "python"     # Markdownコードブロックのタグ
```

**例: R言語の設定**
```yaml
language:
  name: "R"
  interpreter_command: "Rscript"
  file_extension: ".R"
  seed_arg_format: "--seed {seed}"
  code_block_tag: "r"
```

**例: Julia の設定**
```yaml
language:
  name: "julia"
  interpreter_command: "julia"
  file_extension: ".jl"
  seed_arg_format: "-- --seed {seed}"
  code_block_tag: "julia"
```

### 23.3 影響範囲

| コンポーネント | 変更内容 |
|---------------|---------|
| `ExperimentGenerator` | スクリプトファイル名を `experiment{file_extension}` に動的生成。プロンプトに言語名とコードブロックタグを使用 |
| `LocalExecutor` | `interpreter_command` と `seed_arg_format` に基づきサブプロセスを起動 |
| `SlurmExecutor` | 同上（SLURM ジョブ内部で使用） |
| `DockerExecutor` | 同上（コンテナ内で使用、スタブ） |
| `TreeOps.debug` | デバッグプロンプトのコードブロックタグを動的に設定 |
| `replay_cmd` | `experiment.*` パターンでスクリプトを検索、LanguageConfigからインタプリタを決定 |
| `export_cmd` | `experiment.*` パターンで全実験スクリプトをコピー |
| `status_cmd` | `experiment.*` パターンでスクリプトを表示 |
| `StatisticalEvaluator` | `experiment.*` パターンで既存スクリプトを検索 |

### 23.4 metrics.json 出力契約（言語非依存）

全言語の実験スクリプトは同一の `metrics.json` スキーマに準拠する：

```json
{
  "primary": {
    "name": "<metric_name>",
    "value": 0.95,
    "higher_is_better": true
  },
  "constraints": [],
  "secondary": [],
  "raw": {},
  "seed": 42,
  "wall_time_sec": 120.5,
  "<metric_name>": 0.95
}
```

`metrics.json` は実験スクリプトのカレントディレクトリにファイルとして出力する（stdoutではない）。
最上位に `"<metric_name>": <float>` を含めることで後方互換性を維持する。

### 23.5 デフォルト動作

`ProblemSpec.language` が未指定の場合、Python がデフォルト：
- `name: "python"`, `interpreter_command: "python"`, `file_extension: ".py"`
- `seed_arg_format: "--seed {seed}"`, `code_block_tag: "python"`

既存のワークスペースは変更なしで動作する（後方互換）。

---

---

## 24. SLURM実行パイプライン（Local LLM + 実験実行）

SERAはSLURMクラスタ上で2つの異なる実行パターンをサポートする：

1. **実験のSLURM実行** — 生成された実験スクリプトをSLURMジョブとして計算ノードに投入
2. **Local LLM（vLLM）のGPU管理** — ヘッドノード上でvLLMを動作させ、PPO学習とGPUメモリを協調管理

### 24.1 全体アーキテクチャ

```text
┌─────────────── ヘッドノード / ログインノード ───────────────┐
│                                                              │
│  ┌─ AgentLLM ────────────────────────────────────────────┐  │
│  │  provider="local", inference.engine="vllm"            │  │
│  │  ┌─ VLLMInferenceEngine ──────────────────────────┐   │  │
│  │  │  vllm.LLM (offline mode)                       │   │  │
│  │  │  LoRA hot-swap via LoRARequest                  │   │  │
│  │  │  sleep(level=2) / wake_up()                     │   │  │
│  │  └────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
│                    │                                         │
│  SearchManager ────┤                                         │
│  TreeOps           │                                         │
│  PPOTrainer ───────┘  ← GPU共有: vLLM sleep → PPO → wake   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
          │
          │ submitit.AutoExecutor → sbatch
          ▼
┌─── SLURM 計算ノード ───┐
│  _run_experiment()      │
│  ・module load          │
│  ・experiment.py 実行   │
│  ・metrics.json 出力    │
└─────────────────────────┘
```

**重要な設計判断**: vLLMはSLURMジョブ内ではなくヘッドノード上で動作する。これによりジョブキュー待ちなしでの即座の仮説生成が可能。

### 24.2 実験のSLURM実行

#### 24.2.1 設定（ResourceSpec）

`resource_spec.yaml` で実行バックエンドとSLURM設定を指定する：

```yaml
# resource_spec.yaml
compute:
  executor_type: slurm        # "local" | "slurm" | "docker"
  gpu_required: true
  gpu_type: A100
  gpu_count: 2
  cpu_cores: 16
  memory_gb: 128

slurm:
  partition: gpu               # SLURMパーティション名
  account: my_project          # SLURMアカウント / プロジェクト
  time_limit: "04:00:00"       # 壁時計時間制限 (HH:MM:SS or D-HH:MM:SS)
  modules:                     # ジョブ内でロードするモジュール
    - cuda/12.1
    - pytorch/2.0
  sbatch_extra:                # 追加sbatchディレクティブ
    - "--gres=gpu:a100:2"
    - "--constraint=gpu80"
```

**Specモデル定義**（`src/sera/specs/resource_spec.py`）：

| クラス | フィールド | 型 | デフォルト | 説明 |
|--------|-----------|-----|-----------|------|
| `ComputeConfig` | `executor_type` | `str` | `"local"` | 実行バックエンド選択 |
| `SlurmConfig` | `partition` | `str` | `"gpu"` | SLURMパーティション |
| `SlurmConfig` | `account` | `str` | `""` | SLURMアカウント |
| `SlurmConfig` | `time_limit` | `str` | `"04:00:00"` | 壁時計時間制限 |
| `SlurmConfig` | `modules` | `list[str]` | `[]` | ロードする環境モジュール |
| `SlurmConfig` | `sbatch_extra` | `list[str]` | `[]` | 追加sbatchディレクティブ |

#### 24.2.2 Executor選択フロー

`research_cmd.py:64-90` でspec値に基づきExecutorが動的に選択される：

```python
executor_type = getattr(specs.resource.compute, "executor_type", "local")

if executor_type == "slurm":
    from sera.execution.slurm_executor import SlurmExecutor
    executor = SlurmExecutor(
        work_dir=workspace,
        slurm_config=specs.resource.slurm,
        interpreter_command=interpreter_cmd,  # 多言語対応
        seed_arg_format=seed_arg_fmt,
    )
```

#### 24.2.3 SlurmExecutor 実行フロー

`SlurmExecutor.run()`（`src/sera/execution/slurm_executor.py:108-256`）の処理手順：

```text
SlurmExecutor.run(node_id, script_path, seed, timeout_sec)
  │
  ├─ 1. ディレクトリ準備
  │     runs/{node_id}/ を作成
  │     stdout.log, stderr.log, metrics.json パスを設定
  │     slurm_logs/ サブディレクトリを作成
  │
  ├─ 2. submitit設定
  │     submitit.AutoExecutor(folder=slurm_logs/)
  │     slurm_partition, slurm_time, slurm_job_name を設定
  │     sbatch_extra を slurm_additional_parameters に変換
  │
  ├─ 3. ジョブ投入
  │     executor.submit(_run_experiment, interpreter, script, seed, run_dir, modules)
  │     → sbatchジョブとしてSLURMに投入される
  │
  ├─ 4. ポーリング（完了待ち）
  │     sacct利用可能 → submitit経由でjob.stateを確認
  │     sacct利用不可 → squeue -j <job_id> -h -o "%T" で確認
  │     timeout_sec超過 → scancel + TimeoutError
  │
  ├─ 5. ログ収集
  │     submitit出力を stdout.log / stderr.log にコピー（未出力の場合）
  │
  ├─ 6. OOM検出（多層アプローチ）
  │     ① SLURM job state == "OUT_OF_MEMORY"
  │     ② exit_code == 137 or -9 + stderrパターンマッチ
  │     ③ stderr内の "MemoryError" / "OutOfMemoryError" 検出
  │
  └─ 7. RunResult返却
        success, exit_code, stdout_path, stderr_path, metrics_path, wall_time_sec, seed
```

#### 24.2.4 SLURMジョブ内の実行

`_run_experiment()`（`src/sera/execution/slurm_executor.py:26-63`）はSLURMジョブ内部で実行されるcallable：

```python
def _run_experiment(interpreter_command, script_path, seed, run_dir, modules, seed_arg_format):
    # 1. 環境モジュールロード: module load <mod>
    # 2. コマンド構築: [interpreter, script_path, "--seed", str(seed)]
    # 3. subprocess.Popen で実行（stdout/stderrをファイルにリダイレクト）
    # 4. exit code を返却
```

#### 24.2.5 sbatch_extra のパース規則

`sbatch_extra` リスト内の各ディレクティブは以下の形式をサポート：

| 入力形式 | パース結果 |
|---------|-----------|
| `"--gres=gpu:1"` | `{"gres": "gpu:1"}` |
| `"--constraint A100"` | `{"constraint": "A100"}` |
| `"#SBATCH --mem=128G"` | `{"mem": "128G"}` |

これらは `submitit` の `slurm_additional_parameters` に渡される。

#### 24.2.6 タイムアウト制御（二重レイヤー）

| レイヤー | 制御元 | 動作 |
|---------|--------|------|
| **SLURM time_limit** | `SlurmConfig.time_limit` | スケジューラによる強制終了。`state="TIMEOUT"` |
| **Python timeout_sec** | `SlurmExecutor.run()` の引数 | ポーリングループで検出 → `scancel` → `exit_code=-9` |

Python側タイムアウトはSLURMのtime_limitより短い値を設定して早期終了に使用する。

#### 24.2.7 終了コードマッピング

| SLURM State | exit_code | SERAでの意味 | SearchNode.status |
|------------|-----------|-------------|-------------------|
| `COMPLETED` | `job.result()` (通常0) | 成功 | `"evaluated"` |
| `FAILED` | `1` | スクリプトエラー | `"failed"` |
| `TIMEOUT` | `-9` | 時間制限超過 | `"timeout"` |
| `OUT_OF_MEMORY` | `-7` (SERA独自センチネル) | メモリ不足 | `"oom"` |
| `CANCELLED` | `-15` | ユーザーまたは自動キャンセル | `"failed"` |

#### 24.2.8 依存ライブラリ

`submitit`（Meta Research製）をSLURMジョブ管理に使用。オプション依存：

```bash
pip install "sera[slurm]"  # or: pip install submitit
```

### 24.3 Local LLM（vLLM）のSLURMクラスタ上での動作

#### 24.3.1 設定（ModelSpec）

`model_spec.yaml` でvLLMエンジンと推論設定を指定する：

```yaml
# model_spec.yaml
base_model:
  id: meta-llama/Llama-3.1-70B
  revision: ""
  dtype: bf16
  load_method: auto
  max_seq_len: 8192

agent_llm:
  provider: local              # "local" | "openai" | "anthropic"
  temperature: 0.7
  max_tokens: 4096

inference:
  engine: vllm                 # "vllm" | "transformers"
  gpu_memory_utilization: 0.5  # vLLMのGPUメモリ使用率
  max_lora_rank: 64            # vLLMのLoRAプリアロケーション最大ランク
  adapter_cache_dir: /dev/shm/sera_adapters  # tmpfs上のアダプタキャッシュ
  swap_space_gb: 4.0           # vLLMのCPUスワップ領域
  enforce_eager: false         # CUDAグラフ無効化（デバッグ用）
```

**Specモデル定義**（`src/sera/specs/model_spec.py:60-68`）：

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `engine` | `str` | `"transformers"` | 推論エンジン選択 |
| `gpu_memory_utilization` | `float` | `0.5` | vLLMのGPUメモリ割当率 |
| `max_lora_rank` | `int` | `64` | LoRAプリアロケーション最大ランク |
| `adapter_cache_dir` | `str` | `"/dev/shm/sera_adapters"` | アダプタのtmpfsキャッシュ |
| `swap_space_gb` | `float` | `4.0` | CPUスワップ領域（GB） |
| `enforce_eager` | `bool` | `False` | Eagerモード強制（デバッグ） |

#### 24.3.2 vLLMエンジン初期化フロー

`AgentLLM`（`src/sera/agent/agent_llm.py`）は遅延初期化パターンを使用：

```text
AgentLLM.__init__()
  │ _inference_engine = model_spec.inference.engine  （"vllm" or "transformers"）
  │ _vllm_engine = None  （遅延初期化）
  │
AgentLLM.generate(prompt, purpose, adapter_node_id)
  │
  ├─ provider == "local" AND engine == "vllm"
  │   ├─ _init_vllm_engine()  （初回のみ）
  │   │   └─ VLLMInferenceEngine(model_spec)
  │   │       └─ vllm.LLM(model=..., enable_lora=True, ...)
  │   │
  │   └─ _vllm_engine.generate(prompt, temp, max_tok, adapter_node_id, lineage_manager)
  │
  └─ provider == "local" AND engine == "transformers"
      └─ transformers + peft による従来の推論パス
```

#### 24.3.3 VLLMInferenceEngine

`src/sera/agent/vllm_engine.py` の主要コンポーネント：

**初期化**（`__init__`）:
```python
self._llm = LLM(
    model=model_spec.base_model.id,
    revision=model_spec.base_model.revision,
    dtype=model_spec.base_model.dtype,
    enable_lora=True,                      # LoRA事前有効化
    max_lora_rank=inf.max_lora_rank,       # プリアロケーション
    gpu_memory_utilization=inf.gpu_memory_utilization,
    max_model_len=model_spec.base_model.max_seq_len,
    swap_space=inf.swap_space_gb,
    enforce_eager=inf.enforce_eager,
)
```

**LoRA Hot-Swap**（`_get_lora_request`）:
```text
_get_lora_request(adapter_node_id, lineage_manager)
  ├─ キャッシュ確認: adapter_cache_dir/{adapter_node_id}/adapter_model.safetensors
  │   存在しない場合:
  │   └─ lineage_manager.export_for_vllm(adapter_node_id, adapter_dir, model_spec)
  │       ├─ materialize(): root→nodeまでのデルタ重みを累積復元
  │       ├─ save_file(): adapter_model.safetensors を出力
  │       └─ adapter_config.json を出力（peft互換フォーマット）
  │
  ├─ vLLM用整数IDの割当: adapter_id_map[adapter_node_id] → int_id
  └─ LoRARequest(adapter_node_id, int_id, str(adapter_dir)) を返却
```

**生成**（`generate`）:
```python
outputs = self._llm.generate(
    [prompt],
    SamplingParams(temperature=temperature, max_tokens=max_tokens),
    lora_request=lora_request,  # リクエスト単位でアダプタを指定
)
```

#### 24.3.4 GPUメモリ協調管理（Sleep/Wake プロトコル）

**課題**: vLLMとPyTorch（PPO学習）は同一GPU上で共存できない。

**解決策**: vLLMの `sleep(level=2)` / `wake_up()` APIによる明示的なメモリ管理。

```text
SearchManager.run() ループ
  │
  ├─ Phase 2-4: vLLM使用中（通常推論）
  │   AgentLLM → VLLMInferenceEngine.generate()
  │   GPU: vLLMがメモリ占有
  │
  ├─ Phase 5: PPO更新トリガー
  │   PPOTrainer.update()
  │     │
  │     ├─ vllm_engine.sleep()          ← GPUメモリ解放
  │     │   └─ self._llm.sleep(level=2)   level=2 = 完全解放
  │     │
  │     ├─ _ppo_update_core()           ← PyTorchがGPUを使用
  │     │   ├─ GAE計算
  │     │   ├─ PPOクリッピング損失
  │     │   ├─ LoRAパラメータのみ更新
  │     │   └─ デルタ抽出 → lineage保存
  │     │
  │     └─ vllm_engine.wake()           ← GPUメモリ復帰（finally句で保証）
  │         └─ self._llm.wake_up()
  │
  └─ 次のイテレーションでvLLM再利用
```

`ppo_trainer.py:184-203` で `try/finally` パターンにより、PPO更新の成否にかかわらず `wake()` が必ず呼ばれる：

```python
vllm_engine = getattr(agent_llm, "_vllm_engine", None)
if vllm_engine is not None:
    vllm_engine.sleep()
try:
    return await self._ppo_update_core(rollouts, agent_llm, specs, ...)
finally:
    if vllm_engine is not None:
        vllm_engine.wake()
```

#### 24.3.5 アダプタキャッシュ戦略

| 要素 | 詳細 |
|------|------|
| **キャッシュ場所** | `/dev/shm/sera_adapters`（tmpfs = RAMディスク） |
| **キャッシュ単位** | `{adapter_node_id}/adapter_model.safetensors` + `adapter_config.json` |
| **キャッシュ判定** | safetensorsファイルの存在チェック |
| **材料化** | `LineageManager.materialize()`: root→nodeパスのデルタ累積 |
| **出力形式** | peft互換: `adapter_model.safetensors` + `adapter_config.json` |
| **vLLM ID管理** | `adapter_id_map: dict[str, int]` で文字列ID→整数IDマッピング |

#### 24.3.6 export_for_vllm の出力

`LineageManager.export_for_vllm()`（`src/sera/lineage/lineage_manager.py:331-381`）が生成するファイル：

```text
{adapter_cache_dir}/{adapter_node_id}/
  ├─ adapter_model.safetensors    # 材料化されたLoRA重み（safetensors形式）
  └─ adapter_config.json          # peft設定
      {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": <rank>,
        "lora_alpha": <alpha>,
        "target_modules": [...],
        "lora_dropout": <dropout>,
        "bias": "none",
        "base_model_name_or_path": "<model_id>"
      }
```

### 24.4 統合パイプライン：SLURMクラスタ上の全体フロー

```text
research_cmd.run_research()
  │
  ├─ 1. Spec読み込み・検証
  │     executor_type = specs.resource.compute.executor_type  ("slurm")
  │     engine = specs.model.inference.engine                 ("vllm")
  │
  ├─ 2. コンポーネント初期化
  │     AgentLLM(model_spec, resource_spec)      → vLLMエンジン（遅延初期化）
  │     SlurmExecutor(work_dir, slurm_config)    → SLURM実験実行器
  │     StatisticalEvaluator(executor, ...)       → 評価器（executor経由で実験実行）
  │     PPOTrainer(exec_spec, model_spec, ...)   → PPO学習器（オプション）
  │     LineageManager(lineage_dir)              → LoRA系譜管理
  │
  ├─ 3. 探索ループ（SearchManager.run()）
  │     │
  │     ├─ Phase 2: ノード生成
  │     │   AgentLLM.generate() → vLLMで仮説/コード生成（ヘッドノード上）
  │     │
  │     ├─ Phase 3: 実験実行
  │     │   StatisticalEvaluator → SlurmExecutor.run()
  │     │     → sbatchでSLURMジョブ投入（計算ノード）
  │     │     → ポーリングで完了待ち
  │     │     → RunResult返却
  │     │
  │     ├─ Phase 4: 統計評価
  │     │   mu, se, lcb 計算
  │     │   逐次評価（repeats回繰り返し、各回がSLURMジョブ）
  │     │
  │     ├─ Phase 5: PPO学習（条件付き）
  │     │   learning.enabled=True AND provider="local" の場合のみ
  │     │   vLLM sleep → PPO更新 → vLLM wake
  │     │
  │     └─ Phase 6: 系譜管理・剪定
  │         デルタsquash、深いノードの剪定
  │
  └─ 4. 結果出力
        best_node の情報表示
        export-best でアーティファクトエクスポート
```

### 24.5 PPO/Lineageの有効化条件

`research_cmd.py:113-136` において、PPOとLineageは以下の条件でのみ有効化：

```python
learning_enabled = getattr(specs.execution.learning, "enabled", True)
# AND agent_llm.provider == "local" （暗黙の前提：PPOはローカルモデルでのみ可能）
```

| 条件 | PPO | Lineage | vLLM Sleep/Wake |
|------|-----|---------|-----------------|
| `learning.enabled=True` + `provider="local"` | 有効 | 有効 | 有効 |
| `learning.enabled=True` + `provider="openai"` | 無効（例外でfallback） | 無効 | N/A |
| `learning.enabled=False` | 無効 | 無効 | N/A |

PPO/Lineage無効時でも探索ループ（Phase 2-4）は正常に動作する。

### 24.6 ファイルリファレンス

| ファイル | 主要クラス/関数 | 役割 |
|---------|---------------|------|
| `src/sera/execution/slurm_executor.py` | `SlurmExecutor`, `_run_experiment` | SLURMジョブ投入・ポーリング・OOM検出 |
| `src/sera/agent/vllm_engine.py` | `VLLMInferenceEngine` | vLLM推論 + LoRA Hot-Swap + sleep/wake |
| `src/sera/agent/agent_llm.py` | `AgentLLM` | LLMプロバイダ選択・vLLM遅延初期化 |
| `src/sera/learning/ppo_trainer.py` | `PPOTrainer` | PPO更新 + vLLM sleep/wake協調 |
| `src/sera/lineage/lineage_manager.py` | `LineageManager.export_for_vllm()` | アダプタ材料化 + peft形式エクスポート |
| `src/sera/commands/research_cmd.py` | `run_research()` | Executor選択・コンポーネント組み立て |
| `src/sera/specs/resource_spec.py` | `ComputeConfig`, `SlurmConfig` | SLURM設定スキーマ |
| `src/sera/specs/model_spec.py` | `InferenceConfig` | vLLM設定スキーマ |

---

（このTASK.mdは完全最終版 v12.2（SLURM実行パイプライン追加版）である。v12.1 からの追加：§24 SLURM実行パイプライン（SlurmExecutor詳細、vLLM GPUメモリ協調管理、Sleep/Wakeプロトコル、統合フロー図）。v12 からの追加：§23 多言語実験サポート（LanguageConfig, ExperimentGenerator/Executor/コマンドの多言語対応, metrics.json言語非依存契約）、§0.1 エージェント定義表の多言語対応更新）
