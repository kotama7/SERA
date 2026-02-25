# SERA 要件定義書 — Phase 8: 論文評価・改善

> 本ファイルは TASK.md v13.0 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 12. Phase 8：論文評価・改善ループ（AI-Scientist-v2 参考）

### 12.1 PaperScore 評価方法（マルチエージェントレビューアンサンブル）

AI-Scientist-v2 の査読方式を参考に、**Few-shot査読例**、**マルチエージェントレビューアンサンブル**、**エージェント自己改善ループ**を導入。
各レビュアーは独立したエージェントとして動作し、メタレビュー（Area Chair）がオーケストレーションエージェントとして統合判定を行う。

```python
@dataclass
class PaperScoreResult:
    """論文評価結果。全フィールドがデフォルト値付き。"""
    scores: dict[str, float] = field(default_factory=dict)  # {criteria_name: score}（justification なし）
    overall_score: float = 0.0
    confidence: float = 0.0              # 0.0-1.0（仕様の1-5ではなく正規化済み）
    summary: str = ""                    # 論文の要約
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    missing_items: list[str] = field(default_factory=list)                # list[str]（list[dict] ではない）
    improvement_instructions: list[str] = field(default_factory=list)     # list[str]（list[dict] ではない）
    decision: str = ""                   # "accept" | "revise" | "reject"（5段階ではなく3段階）
    passed: bool = False                 # overall_score >= passing_score
    individual_reviews: list[dict] = field(default_factory=list)   # アンサンブル時の個別レビュー（デフォルト空リスト）
    meta_review: str = ""                # アンサンブル時のメタレビュー（デフォルト空文字列）


class PaperEvaluator:
    """PaperScoreSpec に基づく LLM-as-Judge 論文評価（AI-Scientist-v2 スタイル）"""

    async def evaluate(self, paper_md: str, paper_score_spec: PaperScoreSpec,
                 agent_llm: AgentLLM) -> PaperScoreResult:
        """
        ============================================================
        ステップ 1: 単体レビュー生成（num_reviews_ensemble 回の独立レビュー）
        ============================================================
        各レビューは以下の手順で生成:

        a. バイアスモード（交互割当による多様性確保）:
           - bias_mode が "critical" の場合:
             偶数番目（0, 2, ...）のレビュアーは critical、奇数番目（1, 3, ...）は generous
           - bias_mode が "generous" の場合:
             偶数番目は generous、奇数番目は critical
           - それ以外: 全レビュアーが同じ bias_mode

        b. システムプロンプト:
           ---
           You are Reviewer #N for a scientific paper.
           Your reviewing style is {reviewer_bias}.
           Evaluate the paper carefully and provide structured feedback.
           ---

        c. Few-shot 査読例の注入:
           - paper_score_spec.few_shot_reviews から最大 num_fs_examples 件を選択
           - 各例をJSON形式で参考例として提示

        d. 評価プロンプト:
           ---
           ## 評価基準（PaperScoreSpec）
           {paper_score_spec の各 criteria を rubric + weight 付きで列挙}

           ## 参考: 査読例
           {few_shot_reviews（num_fs_examples 件）}

           ## 論文
           {paper_md（先頭8000文字まで）}

           ## 出力形式（構造化テキスト）
           SUMMARY: <1-2文の要約>
           STRENGTHS:
           - <強み1>
           WEAKNESSES:
           - <弱み1>
           QUESTIONS:
           - <質問1>
           LIMITATIONS:
           - <限界1>
           MISSING:
           - <欠落事項1>
           IMPROVEMENTS:
           - <改善指示1>
           SCORES:
           - <criterion_name>: <score 1-max_score>
           OVERALL: <score 1-max_score>
           CONFIDENCE: <0.0-1.0>
           DECISION: <accept|revise|reject>
           ---

        e. エージェント自己改善ループ（num_reviewer_reflections 回、既定2）:
           各レビュアーエージェントが自律的に評価品質を改善する。
           ※ 早期終了なし — 常に num_reviewer_reflections 回全て実行する。

           反省プロンプト:
           ---
           Reflect on your review below. Consider:
           1. Are your scores justified by the evidence?
           2. Have you missed any important strengths or weaknesses?
           3. Are your improvement suggestions actionable?

           Your current review:
           {review_text}

           Provide your REVISED review in the same format.
           If no changes are needed, reproduce the review as-is.
           ---

        f. レビュー解析:
           - 構造化テキストを行単位でパース
           - SUMMARY:/STRENGTHS:/WEAKNESSES:/QUESTIONS:/LIMITATIONS:/MISSING:/IMPROVEMENTS:/SCORES:/OVERALL:/CONFIDENCE:/DECISION: を検出
           - リスト項目は "- " または "* " で始まる行を収集
           - スコアは "- criterion_name: score" 形式でパース

        ============================================================
        ステップ 2: アンサンブル集約（num_reviews_ensemble > 1 の場合）
        ============================================================
        - 各スコア次元について: 全レビューのスコアを収集 → 単純平均
        - overall_score: 個別レビューの overall_score の単純平均
          （overall_score が全て0の場合は criterion スコアの平均にフォールバック）
        - confidence: 個別レビューの confidence の単純平均
        - strengths/weaknesses/questions/limitations: 全レビューからユニークな項目を統合
        - missing_items: 全レビューの "missing" をユニーク統合（list[str]）
        - improvement_instructions: 全レビューの "improvements" をユニーク統合（list[str]）
        - decision: 多数決で決定（Counter.most_common(1) を使用）
        - summary: 個別サマリーを " | " で連結

        アンサンブル集約後、メタレビュー（meta_review=True の場合）:
        ---
        You are an Area Chair at a top ML conference.
        Synthesize the following reviews into a meta-review.

        Paper (first 3000 chars):
        {paper_md[:3000]}

        Reviews:
        {individual_reviews の summary, overall, decision, strengths, weaknesses}

        Provide a meta-review that:
        1. Summarizes the consensus and disagreements
        2. Highlights the most important strengths and weaknesses
        3. Makes a final recommendation (accept/revise/reject)
        4. Provides specific improvement instructions if applicable
        ---

        ============================================================
        ステップ 3: 結果構築
        ============================================================
        1. overall_score = 個別 overall_score の単純平均
        2. passed = overall_score >= passing_score
        3. PaperScoreResult を返す（individual_reviews と meta_review を含む）
        """
        pass
```

### 12.2 改善ループ（具体）
```python
def run_evaluate_paper(work_dir: str) -> None:
    """
    Phase 8: 論文評価・改善ループ（paper_cmd.py の実装に基づく）

    前提:
    - Phase 7（sera generate-paper）で paper/paper.md が生成済みであること
    - paper.md が存在しない場合はエラー終了

    ループ構造:
    - compose() はループ外で1回のみ実行済み（Phase 7 で生成済みの paper.md を読み込む）
    - 改善時に再 compose はしない（LLM による直接テキスト修正のみ）

    1. paper_evaluator.evaluate() で採点（非同期、アンサンブル+レビュアー反省含む）
    2. paper_log.jsonl に記録:
       {event: "paper_evaluation", iteration, overall_score, passed, decision, scores}
    3. passed=true なら終了
    4. passed=false なら:
       a. improvement_instructions（list[str]）を順に適用
       b. 各 instruction について LLM に revision prompt を送信:
          "Revise the following paper based on this instruction:
           {instruction}
           Paper:
           {paper_md}"
       c. LLM の応答で paper_md を更新
       d. 修正版を paper/paper.md に上書き保存
    5. paper_revision_limit（specs.execution.paper.paper_revision_limit）回まで繰り返し
    6. 最終スコアをコンソールに出力

    注意:
    - improvement_instructions は list[str] であり、priority や requires_experiment による
      分岐は行わない（テキスト修正のみ）
    - 追加実験の自動実行は Phase 8 では行わない
    - Phase 7 のライティング内反省ループと Phase 8 の外部評価ループは二重構造:
      Phase 7（内部）: 構文・図・整合性の自己改善（未使用図、無効引用、欠落セクション等）
      Phase 8（外部）: 科学的品質の第三者的評価（査読シミュレーション）
    """
    workspace = Path(work_dir)
    paper_md = (workspace / "paper" / "paper.md").read_text()

    for iteration in range(revision_limit):
        result = await evaluator.evaluate(paper_md, specs.paper_score, agent_llm)

        paper_logger.log({
            "event": "paper_evaluation",
            "iteration": iteration + 1,
            "overall_score": result.overall_score,
            "passed": result.passed,
            "decision": result.decision,
            "scores": result.scores,
        })

        if result.passed:
            break

        # 改善（テキスト修正のみ）
        for instruction in result.improvement_instructions:
            revision_prompt = (
                f"Revise the following paper based on this instruction:\n"
                f"{instruction}\n\nPaper:\n{paper_md}"
            )
            paper_md = await agent_llm.generate(
                revision_prompt, purpose="paper_revision"
            )

        # 修正版を上書き保存
        (workspace / "paper" / "paper.md").write_text(paper_md)
```

---
