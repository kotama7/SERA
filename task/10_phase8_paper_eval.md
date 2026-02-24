# SERA 要件定義書 — Phase 8: 論文評価・改善

> 本ファイルは TASK.md v12.4 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 12. Phase 8：論文評価・改善ループ（AI-Scientist-v2 参考）

### 12.1 PaperScore 評価方法（マルチエージェントレビューアンサンブル）

AI-Scientist-v2 の査読方式を参考に、**Few-shot査読例**、**マルチエージェントレビューアンサンブル**、**エージェント自己改善ループ**を導入。
各レビュアーは独立したエージェントとして動作し、メタレビュー（Area Chair）がオーケストレーションエージェントとして統合判定を行う。

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

        d. エージェント自己改善ループ（最大 num_reviewer_reflections 回、既定2）:
           各レビュアーエージェントが自律的に評価品質を改善する。各ラウンドで:
           - レビュアーエージェントが自身の評価の正確性・公正性を再検討
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

        アンサンブル集約が有効な場合、オーケストレーションエージェント（Area Chair）がメタレビューを生成:
        ---
        あなたはオーケストレーションエージェント（エリアチェア）です。以下の {n} 件の独立したレビュアーエージェントの査読結果を統合し、
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
