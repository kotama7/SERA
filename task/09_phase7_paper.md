# SERA 要件定義書 — Phase 7: 論文生成

> 本ファイルは TASK.md v12.4 を分割したものである。目次は [README.md](./README.md) を参照。

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

### 11.3 ワークフロー（エージェント的反復改善：AI-Scientist-v2 スタイル）

Phase 7 は以下の **6ステップ** を順次実行する。エージェントが反復的に論文品質を改善し、
Phase 8 のマルチエージェントレビュー（§12）に行く前に内部品質を高める。

```python
class PaperComposer:
    """AI-Scientist-v2 スタイルの論文生成エージェント。
    エージェントが反復的に改善サイクルを回し、自律的に論文品質を高める。"""

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

        5c. エージェント自己修正サイクル（最大 n_writeup_reflections 回、既定3）
            エージェントが自律的に品質を評価し修正する反復ループ。各ラウンドで以下を実行:
            i.   LaTeX の場合: pdflatex 4-pass コンパイル + chktex 構文チェック
                 Markdown の場合: pandoc で整合性検証
            ii.  未使用図の検出（figures/ にあるが本文で参照されていない図）
            iii. 無効な図参照の検出（本文で参照されているが figures/ にない図）
            iv.  エージェントフィードバックループ — VLM 図・キャプション・本文参照レビュー（vlm が有効な場合）:
                 - VLM が各図について以下を評価:
                   * 図の内容とキャプションの整合性
                   * 本文中の参照箇所での説明の適切性
                   * 図の情報量と有用性（ページ制約下で残す価値があるか）
                 - 出力: 図ごとの {img_review, caption_review, figrefs_review}
            v.   VLM 重複図検出（vlm が有効な場合）:
                 - 全図を VLM に送り、内容が類似する図のペアを特定
                 - 本文とAppendixの間の重複も検出
            vi.  エージェント反省プロンプトを送信（上記i-vの結果を含む）:
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
