# SERA 要件定義書 — Phase 7: 論文生成

> 本ファイルは TASK.md v13.1 を分割したものである。目次は [README.md](./README.md) を参照。

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
    """Phase 2-6 の全結果を論文生成に必要な形式で保持。
    全フィールドがデフォルト値付きで、部分的なデータでも動作する。"""

    best_node: Any = None               # SearchNode（最良ノード）
    top_nodes: list = field(default_factory=list)            # LCB上位ノード群
    all_evaluated_nodes: list = field(default_factory=list)  # 評価済み全ノード
    search_log: list[dict] = field(default_factory=list)     # search_log.jsonl の全エントリ
    eval_log: list[dict] = field(default_factory=list)       # eval_log.jsonl の全エントリ
    ppo_log: list[dict] = field(default_factory=list)        # ppo_log.jsonl の全エントリ
    problem_spec: Any = None             # ProblemSpec
    related_work: Any = None             # RelatedWorkSpec
    execution_spec: Any = None           # ExecutionSpec

    @classmethod
    def from_workspace(cls, work_dir: str | Path) -> "EvidenceStore":
        """ワークスペースディレクトリから EvidenceStore を構築。
        logs/ 以下の search_log.jsonl, eval_log.jsonl, ppo_log.jsonl を読み込む。"""
        ...

    def to_json(self) -> dict:
        """デバッグ用のJSON互換dict。num_evaluated_nodes, search_log_len,
        eval_log_len, ppo_log_len, best_node_id を返す。"""
        ...

    def get_main_results_table(self) -> str:
        """主要結果の表をMarkdown形式で返す。
        - all_evaluated_nodes を LCB 降順でソート
        - 各ノードについて method（experiment_config.get("method") or hypothesis[:50]）、
          μ±SE、LCB、Feasible（Yes/No）の列を出力
        - ヘッダ: | Method | Metric (μ ± SE) | LCB | Feasible |
        """
        ...

    def get_ablation_data(self) -> dict:
        """アブレーション実験データを返す。
        - best_node の子ノードで branching_op == "improve" かつ
          experiment_config に best_node との差分があるノードを抽出
        - 差分の最初のキーを変数名として使用
        - 戻り値: {variable_name: {"mu": float, "se": float, "lcb": float, "config": dict}}
        """
        ...

    def get_convergence_data(self) -> list[tuple[int, float]]:
        """(step, best_lcb) の累積最大時系列データを返す。
        - search_log を順にスキャンし、各エントリの lcb を取得
        - best_lcb の累積最大値を追跡し、(index, best_lcb) のリストを返す
        """
        ...

    def get_experiment_summaries(self) -> dict[str, list[dict]]:
        """stage別の実験結果をJSON要約形式で返す。
        分類ロジック:
        - baseline: branching_op == "draft" かつ depth == 0 のノード
        - research: それ以外の全ノード（improve, debug 等を含む）
        - ablation: （現状は空リスト、get_ablation_data() で別途取得）
        各エントリ: {node_id, hypothesis, config, mu, se, lcb, feasible, op}
        """
        ...
```

### 11.3 ワークフロー（エージェント的反復改善：AI-Scientist-v2 スタイル）

Phase 7 は以下の **6ステップ** を順次実行する。エージェントが反復的に論文品質を改善し、
Phase 8 のマルチエージェントレビュー（§12）に行く前に内部品質を高める。

```python
@dataclass
class Paper:
    """生成された論文を表すデータクラス"""
    content: str = ""                                  # Markdown文字列
    figures: list[Path] = field(default_factory=list)  # 図のパスリスト
    bib_entries: list[dict] = field(default_factory=list)  # 引用エントリ
    metadata: dict = field(default_factory=dict)       # メタ情報（output_dir 等）


class PaperComposer:
    """AI-Scientist-v2 スタイルの論文生成エージェント。
    エージェントが反復的に改善サイクルを回し、自律的に論文品質を高める。"""

    def __init__(self, output_dir: str | Path, n_writeup_reflections: int = 3) -> None:
        """
        output_dir: 論文出力ディレクトリ（自動作成される）
        n_writeup_reflections: ステップ5c の反省ループ最大回数（既定3）
        """

    async def compose(
        self,
        evidence: EvidenceStore,
        paper_spec: PaperSpec | None = None,
        teacher_papers: TeacherPaperSet | None = None,
        agent_llm: AgentLLM,
        vlm: VLMReviewer | None = None,
        semantic_scholar_client: SemanticScholarClient | None = None,
    ) -> Paper:
        """
        ========================================
        ステップ 1: 実験ログ要約（Log Summarization）
        ========================================
        - evidence.get_experiment_summaries() から stage 別の構造化要約を生成
        - evidence.get_convergence_data(), get_ablation_data(), get_main_results_table() も収集
        - best_node の hypothesis, mu, se, lcb, config を抽出
        - 出力: paper/experiment_summaries.json に統合保存

        ========================================
        ステップ 2: 図生成・集約（Plot Aggregation）
        ========================================
        - FigureGenerator で基本図を生成:
          a. CI棒グラフ: 各手法の primary ± 95% CI（1.96 * SE）
          b. 収束曲線: evidence.get_convergence_data()
          c. アブレーション表: 棒グラフ画像（μ と LCB の並列表示）
          d. 探索木可視化: graphviz で全ノードのツリー（上位ノードをハイライト）
        - LLM が実験結果を統合した追加集約図スクリプトを生成（反省ループ最大2回）
        - 最大12枚、300 DPI
        - 出力: paper/figures/*.png

        ========================================
        ステップ 3: 自動引用検索（Citation Search Loop）
        ========================================
        - CitationSearcher を使用（semantic_scholar_client を渡す）
        - 最大 20 ラウンド:
          a. LLM が現在の論文コンテキストから「最も重要な不足引用」を特定
          b. LLM が Semantic Scholar 検索クエリを生成
          c. Semantic Scholar API で論文検索
          d. LLM が検索結果から関連論文を選択
          e. BibTeX エントリを生成
          f. LLM が "No more citations needed" と判断したら早期終了
        - 出力: paper/paper.bib（引用エントリ）、logs/citation_search_log.jsonl

        ========================================
        ステップ 4: VLM 図記述生成（VLM Figure Description）
        ========================================
        ※ vlm が None または vlm.enabled == False の場合はスキップ
        - VLM が paper/figures/ 内の各 PNG を視覚的に分析
        - 各図に対して科学的記述（軸、トレンド、パターン、統計的特徴）を生成
        - 出力: figure_descriptions: dict[str, str]（ファイル名→記述のマッピング）
        - この記述はステップ5のLLMコンテキストに注入される

        ========================================
        ステップ 5: 論文本体生成 + ライティング内反省ループ
        ========================================
        5a. アウトライン生成
            - LLM に evidence のベスト結果 + 結果テーブル + 図一覧 + 引用一覧を与え、
              paper_spec.sections_required に沿ったアウトライン（各章の箇条書き）を生成
            - teacher_papers のタイトルをスタイルガイダンスとして注入

        5b. 初版Markdown生成（1パス）
            - LLM に以下を統合入力:
              * アウトライン（ステップ5aの出力）
              * 結果テーブル
              * ベスト手法の詳細
              * 図ファイル一覧 + VLM記述（ステップ4の出力）
              * 引用一覧（ステップ3の出力）
            - 全セクションを1パスで生成（Markdown形式）

        5c. エージェント自己修正サイクル（最大 n_writeup_reflections 回、既定3）
            エージェントが自律的に品質を評価し修正する反復ループ。各ラウンドで以下を実行:

            i.   _check_paper_issues() による自動チェック:
                 - 未使用図の検出（figures/ にあるが本文で参照されていない図）
                 - 無効引用キーの検出（\cite{key} が bibliography に存在しない）
                 - 欠落セクションの検出（abstract, introduction, method, experiment,
                   result, conclusion のいずれかが本文に含まれない）
                 - 未閉じコードブロックの検出（``` の数が奇数）

            ii.  VLM 図レビュー（vlm が有効な場合）:
                 - 最大3図に対して review_figure_caption_refs を呼ぶ
                 - caption と text_refs は空で呼ばれる（caption="", text_refs=[]）
                 - suggestion フィールドがあればフィードバックとして収集

            iii. issues も vlm_feedback もなければ早期終了

            iv.  エージェント反省プロンプトを送信（上記 i-ii の結果を含む）:
                 - 検出された問題の修正
                 - VLMレビュー結果に基づく修正
                 - 現在のドラフト（先頭6000文字）を含む
            v.   LLM が修正版を返す（完全なMarkdownとして）

        ========================================
        ステップ 6: 統合・最終出力
        ========================================
        - 図番号の連番化（![caption](file) → ![Figure N: caption](file)）
        - 引用キーの整合（正規化）
        - References セクションの自動付与:
          本文に "# references" が含まれない場合、末尾に
          "## References" セクションを自動追加（各エントリを [key] Title by Authors (Year) 形式で列挙）
        - 出力ファイルの保存:
          * paper/paper.md — 論文本文
          * paper/paper.bib — BibTeX（@article 形式で各エントリを出力）
          * paper/figure_descriptions.json — VLM図記述（VLM有効時のみ）

        """
        pass
```

### 11.4 VLMReviewer（図の視覚レビュー）

```python
class VLMReviewer:
    """VLM（Vision Language Model）による図の視覚的レビュー。
    OpenAI と Anthropic の両プロバイダをサポート。"""

    def __init__(self, model: str | None = None, provider: str | None = None):
        """
        model: VLMモデル名（例: "gpt-4o", "claude-sonnet-4-20250514"）。None で無効化。
        provider: "openai" | "anthropic"。None で無効化。
        self.enabled: model と provider の両方が指定されている場合のみ True。
                      パッケージ未インストール時も False にフォールバック。
        ※ ModelSpec.vlm で設定。enabled=False の場合、全メソッドが空/no-op結果を返す。
        """

    def _encode_image(path: Path) -> str:
        """画像ファイルを base64 エンコードして返す（staticmethod）"""

    def _call_vlm(self, text_prompt: str, image_paths: list[Path]) -> str:
        """プロバイダに応じて VLM API を呼び出す。
        - OpenAI: image_url 形式（data:{mime};base64,{b64}）で送信
        - Anthropic: image source 形式（type=base64, media_type, data）で送信
        両プロバイダとも max_tokens=2048。
        """

    def describe_figures(self, figure_paths: list[Path]) -> dict[str, str]:
        """
        各図を視覚的に分析し、科学的記述を生成。
        戻り値: {filename: description} のマッピング
        enabled=False の場合は {filename: ""} を返す。
        """
        pass

    def review_figure_caption_refs(self, figure_path: Path, caption: str,
                                    text_refs: list[str]) -> dict[str, str]:
        """
        図・キャプション・本文参照の整合性をレビュー。

        戻り値（全フィールドが str 型）:
        {
            "img_review": "...",         # 図自体の品質評価
            "caption_review": "...",     # キャプションとの整合性
            "figrefs_review": "...",     # 本文参照の適切性
            "informative": "yes/no/unknown",  # str型（bool ではない）
            "suggestion": "..."          # 改善提案
        }
        enabled=False の場合は全フィールド空文字列（informative は "unknown"）を返す。
        """
        pass

    def detect_duplicate_figures(self, figure_paths: list[Path]) -> list[dict]:
        """
        内容が類似する図のペアを検出。
        - 全ペアを比較（最大15ペアまで）
        - VLM に similarity（0.0-1.0）と recommendation を尋ねる
        - similarity > 0.5 のペアのみ結果に含める
        戻り値: [{"fig_a": str, "fig_b": str, "similarity": float, "recommendation": str}]
        """
        pass
```

### 11.5 CitationSearcher（自動引用検索）

```python
class CitationSearcher:
    """Semantic Scholar API を使った自動引用検索（AI-Scientist-v2 参考）"""

    def __init__(
        self,
        semantic_scholar_client: SemanticScholarClient | None = None,
        agent_llm: AgentLLM | None = None,
        log_dir: str | Path | None = None,
    ):
        """
        log_dir: 指定した場合、citation_search_log.jsonl にラウンドごとのログを出力。
        各ラウンドのログエントリ: {round, action, query, claim, citation_key, title, year}
        """
        pass

    async def search_loop(
        self,
        context: str,
        existing_bibtex: str = "",
        max_rounds: int = 20,
    ) -> list[dict]:
        """
        自動引用検索ループ（非同期メソッド）:
        1. LLM が context（研究概要+実験要約）から最も重要な不足引用を特定
        2. LLM が Semantic Scholar 用の検索クエリを生成
        3. Semantic Scholar API で検索（agent_llm が None の場合はスキップ）
        4. LLM が検索結果から関連論文を選択（番号で指定）
        5. LLM が BibTeX エントリを生成
        6. LLM が "No more citations needed" と判断するか、max_rounds に達したら終了

        citation_key の生成: 第一著者の姓 + 年（重複時は a, b, c... を付与）

        戻り値: [{"citation_key": str, "title": str, "authors": list[str],
                  "year": int | None, "bibtex": str, "paper_id": str}]
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
- `paper/paper.bib`（BibTeX：@article 形式、citation_key / title / author / year / journal / doi）
- `paper/experiment_summaries.json`（ステップ1で生成、ベストノード・結果テーブル・収束データ等）
- `paper/figure_descriptions.json`（VLM による図記述、VLM有効時のみ）
- `logs/citation_search_log.jsonl`（引用検索の各ラウンドの記録）

---
