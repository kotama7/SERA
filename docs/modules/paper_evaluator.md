# PaperEvaluator / PaperScoreResult

Phase 8 のアンサンブル LLM-as-Judge 論文評価モジュール。複数の独立したレビューアによるレビュー、リフレクションループ、スコア集約、メタレビューを実行する。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `PaperEvaluator` / `PaperScoreResult` | `src/sera/paper/paper_evaluator.py` |

## 依存関係

- `sera.agent.agent_llm` (`AgentLLM`) -- レビュー生成用 LLM

---

## PaperScoreResult (dataclass)

アンサンブルレビューの集約結果。

```python
@dataclass
class PaperScoreResult:
    scores: dict[str, float]                     # 基準別平均スコア
    overall_score: float = 0.0                    # 全体スコア（レビューア平均）
    confidence: float = 0.0                       # 信頼度（レビューア平均）
    summary: str = ""                             # サマリ（" | " 区切り）
    strengths: list[str]                          # 強み（重複排除済み）
    weaknesses: list[str]                         # 弱み（重複排除済み）
    questions: list[str]                          # 質問（重複排除済み）
    limitations: list[str]                        # 制限事項（重複排除済み）
    missing_items: list[str]                      # 不足項目（重複排除済み）
    improvement_instructions: list[str]           # 改善指示（重複排除済み）
    decision: str = ""                            # "accept" | "revise" | "reject"
    passed: bool = False                          # overall_score >= passing_score
    individual_reviews: list[dict]                # 個別レビューの生データ
    meta_review: str = ""                         # エリアチェアスタイルのメタレビュー
```

---

## PaperEvaluator

### コンストラクタ

```python
def __init__(self) -> None
```

ステートレスなクラス。設定は `evaluate()` の引数として渡される。

### evaluate(paper_md, paper_score_spec, agent_llm) -> PaperScoreResult [async]

論文をアンサンブルレビューで評価する。

**パラメータ:**

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `paper_md` | `str` | Markdown 形式の論文内容 |
| `paper_score_spec` | `PaperScoreSpecModel` | 評価基準、ルーブリック、アンサンブル設定 |
| `agent_llm` | `AgentLLM` | レビュー生成用 LLM |

**アンサンブル設定（`paper_score_spec.ensemble` から取得）:**

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `num_reviews_ensemble` | `3` | レビューア数 |
| `num_reviewer_reflections` | `2` | リフレクションラウンド数 |
| `num_fs_examples` | `2` | Few-shot 例の数 |
| `bias_mode` | `"critical"` | バイアスモード |
| `meta_review` | `True` | メタレビューの有効化 |
| `temperature` | `0.75` | サンプリング温度 |

**処理フロー:**

1. **独立レビュー生成**: `num_reviews_ensemble` 回のレビューを並行生成
2. **アンサンブル集約**: スコア平均、テキスト統合、多数決
3. **メタレビュー**: 複数レビューがある場合にエリアチェアスタイルのメタレビューを生成

---

### _generate_review(...) -> dict [async]

単一のレビューをリフレクションループ付きで生成する。

**パラメータ:**

| パラメータ | 説明 |
|-----------|------|
| `paper_md` | 論文（先頭 8000 文字に制限） |
| `criteria` | 評価基準リスト |
| `max_score` | 基準あたりの最大スコア |
| `bias_mode` | `"critical"` / `"generous"`（偶数/奇数レビューアで交互） |
| `num_reflections` | リフレクションラウンド数 |
| `few_shot_reviews` | Few-shot 例 |
| `temperature` | サンプリング温度 |

**処理フロー:**

1. システムプロンプト構築（レビューア番号、バイアスモード）
2. ルーブリックテキスト構築（各基準の名前、説明、重み、スコア記述）
3. Few-shot 例の追加
4. 評価プロンプトの構築（構造化フォーマット指定）
5. 初回レビュー生成（`agent_llm.generate()`）
6. リフレクションループ: `num_reflections` 回繰り返し
   - スコアの妥当性確認
   - 見落としの確認
   - 改善提案の実行可能性確認
7. レビューをパース（`_parse_review()`）

### バイアスモード交互化

レビューの多様性を確保するため、バイアスモードを交互に割り当てる:

| レビューア | `bias_mode="critical"` 時 | `bias_mode="generous"` 時 |
|-----------|-------------------------|--------------------------|
| #0 (偶数) | `critical` | `generous` |
| #1 (奇数) | `generous` | `critical` |
| #2 (偶数) | `critical` | `generous` |

---

### _parse_review(review_text, criteria, max_score) -> dict

構造化されたレビューテキストを辞書にパースする。

**パース対象セクション:**

| セクション | キー | 型 |
|-----------|------|-----|
| `SUMMARY:` | `summary` | `str` |
| `STRENGTHS:` | `strengths` | `list[str]` |
| `WEAKNESSES:` | `weaknesses` | `list[str]` |
| `QUESTIONS:` | `questions` | `list[str]` |
| `LIMITATIONS:` | `limitations` | `list[str]` |
| `MISSING:` | `missing` | `list[str]` |
| `IMPROVEMENTS:` | `improvements` | `list[str]` |
| `SCORES:` | `scores` | `dict[str, float]` |
| `OVERALL:` | `overall_score` | `float` |
| `CONFIDENCE:` | `confidence` | `float` |
| `DECISION:` | `decision` | `str` |

リストセクション内の項目は `- ` または `* ` プレフィックスで区切られる。スコアは `max_score` でクリッピングされる。

---

### _aggregate_reviews(reviews, criteria, max_score, passing_score) -> PaperScoreResult

複数のレビューを単一の `PaperScoreResult` に集約する。

**集約ルール:**

| フィールド | 集約方法 |
|-----------|---------|
| `scores` | 基準別の平均 |
| `overall_score` | レビューア全体の平均（フォールバック: 基準スコアの平均） |
| `confidence` | レビューア全体の平均 |
| `summary` | `" \| "` で結合 |
| `strengths`, `weaknesses` 等 | 重複排除して統合 |
| `decision` | 多数決（`Counter.most_common(1)`） |
| `passed` | `overall_score >= passing_score` |

---

### _generate_meta_review(paper_md, reviews, agent_llm) -> str [async]

エリアチェアスタイルのメタレビューを生成する。

**プロンプトに含まれる情報:**

- 論文（先頭 3000 文字）
- 各レビューアのサマリ、全体スコア、決定、強み、弱み

**メタレビューの内容:**

1. コンセンサスと意見の相違のまとめ
2. 最も重要な強みと弱みのハイライト
3. 最終推薦（accept / revise / reject）
4. 改善指示（該当する場合）
