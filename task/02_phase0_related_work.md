# SERA 要件定義書 — Phase 0: 先行研究

> 本ファイルは TASK.md v12.4 を分割したものである。目次は [README.md](./README.md) を参照。

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
