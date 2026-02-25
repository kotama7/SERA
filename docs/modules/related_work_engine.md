# RelatedWorkEngine / API Clients / Ranking / Clustering

Phase 0 の関連研究収集を担当するモジュール群のドキュメント。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `RelatedWorkEngine` | `src/sera/phase0/related_work_engine.py` |
| `BaseScholarClient` | `src/sera/phase0/api_clients/base.py` |
| `SemanticScholarClient` | `src/sera/phase0/api_clients/semantic_scholar.py` |
| `CrossRefClient` | `src/sera/phase0/api_clients/crossref.py` |
| `ArxivClient` | `src/sera/phase0/api_clients/arxiv.py` |
| `WebSearchClient` | `src/sera/phase0/api_clients/web_search.py` |
| `rank_papers` 等 | `src/sera/phase0/ranking.py` |
| `cluster_papers` 等 | `src/sera/phase0/clustering.py` |

## 依存関係

- `httpx` -- 非同期 HTTP クライアント
- `tenacity` -- リトライ機構（Semantic Scholar）
- `structlog` -- 構造化ログ
- `sera.utils.logging` (`JsonlLogger`)
- `sera.specs.phase0` -- 出力スペックモデル（`RelatedWorkSpec`, `PaperSpec`, `PaperScoreSpec`, `ClusterSpec`, `TeacherPaperSet`, `BaselineCandidate`, `OpenProblem`）
- `sera.specs.input1` (`Input1Model`)

---

## Phase0Config (dataclass)

Phase 0 パイプラインのチューニングパラメータ。

| フィールド | デフォルト | 説明 |
|-----------|----------|------|
| `top_k_papers` | 10 | 最終的に返す論文数 |
| `recent_years_bias` | 5 | 現在の年から何年前までを検索対象とするか |
| `citation_graph_depth` | 1 | 引用グラフの展開深度 |
| `teacher_papers` | 5 | ティーチャー論文の数 |
| `citation_weight` | 0.6 | ランキングにおける引用スコアの重み |

## Phase0Output (dataclass)

Phase 0 で生成される 4 つのアーティファクトを保持するコンテナ。

| フィールド | 型 |
|-----------|-----|
| `related_work_spec` | `RelatedWorkSpec` |
| `paper_specs` | `list[PaperSpec]` |
| `paper_scores` | `list[PaperScoreSpec]` |
| `teacher_paper_set` | `TeacherPaperSet` |

---

## RelatedWorkEngine

Phase 0 関連研究パイプラインのオーケストレータ。

### コンストラクタ

```python
def __init__(
    self,
    clients: list[BaseScholarClient],    # API クライアントのリスト（優先度順）
    agent_llm: Callable[[str], Awaitable[str]] | None = None,  # 非同期 LLM コーラブル
    logger: JsonlLogger | None = None,
)
```

**注意:** `agent_llm` は `AgentLLM` オブジェクトではなく、プロンプト文字列を受け取って応答文字列を返す非同期コーラブル。

### run(input1, config=None) -> Phase0Output

Phase 0 パイプライン全体を実行する非同期メソッド。

**処理フロー:**

1. **クエリ生成** (`_build_queries`):
   - LLM が利用可能な場合: Input-1 の task, domain, goal から 3 つの多様な検索クエリを生成するよう LLM に依頼。応答を改行で分割してクエリリストを取得
   - LLM が利用不可または失敗時: フォールバックヒューリスティック `"{task_brief} {field}"` + `" {subfield}"` を使用

2. **検索** (`_search_with_fallback`):
   - 各クエリに対して、`clients` リスト内のクライアントを優先度順に試行
   - 最初に結果を返したクライアントの結果を採用
   - 失敗時は次のクライアントにフォールバック
   - `year_from = current_year - recent_years_bias` で最近の論文にバイアス

3. **重複排除** (`_deduplicate`):
   - `paper_id` で重複を除去、最初の出現を保持
   - **クエリログ**: 各 API 呼び出しの結果を `queries.jsonl` に記録（後述）

4. **引用グラフの展開** (`_expand_citations`):
   - `citation_graph_depth > 0` の場合に実行
   - 上位 `top_k_papers` 件の論文に対して、各クライアントで `get_references()` と `get_citations()` を呼び出し
   - 結果を追加した後、再度重複排除

5. **ランキング** (`rank_papers`):
   - `ranking_weight = citation_weight` でランキングスコアを計算
   - スコア降順でソートし、上位 `top_k_papers` 件を取得

6. **クラスタリング** (`cluster_papers`):
   - LLM ベースのテーマ別クラスタリング

7. **出力スペック構築**:
   - 各論文を `PaperSpec` と `PaperScoreSpec` に変換
   - `ClusterSpec` リストを生成
   - `RelatedWorkSpec`（papers, clusters, scores, baseline_candidates, common_metrics, open_problems）を構築
   - `TeacherPaperSet` として上位 `teacher_papers` 件を設定

### RelatedWorkSpec の追加フィールド

`run()` メソッドは `RelatedWorkSpec` に以下の追加フィールドを含める:

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `baseline_candidates` | `list[BaselineCandidate]` | 被引用数上位 5 件の論文を構造化データとして保持 |
| `common_metrics` | `list[str]` | `input1.goal.metric` から抽出された主要メトリクス名のリスト |
| `open_problems` | `list[OpenProblem]` | クラスタの `description` から抽出された未解決問題を構造化データとして保持 |

**BaselineCandidate** (dataclass):

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `name` | `str` | 論文タイトル |
| `paper_id` | `str` | 論文 ID |
| `reported_metric` | `str` | 報告されたメトリクス名（`input1.goal.metric` から取得） |
| `method_summary` | `str` | アブストラクトの先頭 200 文字 |

**OpenProblem** (dataclass):

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `description` | `str` | クラスタの説明文 |
| `related_paper_ids` | `list[str]` | 関連する論文 ID のリスト |
| `severity` | `str` | 深刻度（デフォルト `"medium"`） |

これらは Phase 2 のルートドラフト生成時にプロンプトへ注入され、`baseline`/`open_problem`/`novel` カテゴリ分割に活用される。

### クエリログ（queries.jsonl）

`_search_with_fallback()` は各 API 呼び出しの結果をログに記録する:

```json
{
  "event": "api_query",
  "query_id": "<UUID>",
  "api": "SemanticScholarClient",
  "endpoint": "SemanticScholarClient",
  "params": {"query": "...", "limit": 20, "year_from": 2021},
  "timestamp_utc": "2026-01-15T12:34:56+00:00",
  "http_status": 200,
  "result_count": 15,
  "paper_ids_returned": ["id1", "id2", "..."],
  "retry_count": 0,
  "error": null
}
```

パイプライン完了時には `phase0_complete` イベントも記録される:

```json
{
  "event": "phase0_complete",
  "total_searched": 120,
  "top_k": 10,
  "num_clusters": 5,
  "teacher_papers": 5
}
```

---

## API Clients

### BaseScholarClient (ABC)

全 API クライアントが実装する抽象基底クラス。

```python
class BaseScholarClient(ABC):
    @abstractmethod
    async def search(self, query: str, limit: int = 20, year_from: int | None = None) -> list[PaperResult]: ...

    @abstractmethod
    async def get_references(self, paper_id: str, limit: int = 20) -> list[PaperResult]: ...

    @abstractmethod
    async def get_citations(self, paper_id: str, limit: int = 20) -> list[PaperResult]: ...
```

### PaperResult (dataclass)

全 API クライアントが返す統一論文表現。

| フィールド | 型 | デフォルト |
|-----------|-----|----------|
| `paper_id` | `str` | -- |
| `title` | `str` | -- |
| `authors` | `list[str]` | `[]` |
| `year` | `int \| None` | `None` |
| `venue` | `str` | `""` |
| `abstract` | `str` | `""` |
| `citation_count` | `int` | `0` |
| `url` | `str` | `""` |
| `doi` | `str` | `""` |
| `arxiv_id` | `str` | `""` |
| `source_api` | `str` | `""` |
| `relevance_score` | `float` | `0.5` |

### SemanticScholarClient

Semantic Scholar Graph API v1 のクライアント。

- ベース URL: `https://api.semanticscholar.org/graph/v1`
- リトライ: `tenacity` で最大 5 回、指数バックオフ（1-60 秒）、`HTTPStatusError` でリトライ
- API キー対応（`x-api-key` ヘッダ）
- 引用/参照グラフ対応（`get_references`, `get_citations` が実装済み）
- `_parse_paper()`: `citedPaper` / `citingPaper` ネストの解決、`externalIds` から DOI / ArXiv ID を取得

### CrossRefClient

CrossRef REST API のクライアント。

- DOI ベースの検索
- アブストラクト内の JATS XML タグを正規表現で除去
- 年の取得: `published-print` -> `published-online` -> `created` の優先順位
- `get_references()` / `get_citations()` は空リストを返す（CrossRef は引用グラフエンドポイントを提供しない）

### ArxivClient

arXiv Atom API のクライアント。

- Atom XML パーシング（`xml.etree.ElementTree`）
- レート制限: リクエスト間 3 秒の最小遅延（`_MIN_DELAY_SECONDS = 3.0`）
- 年フィルタ: arXiv API はサーバーサイドの年フィルタを提供しないためクライアントサイドで実装
- 引用数は常に 0（arXiv は引用数を提供しない）
- `get_references()` / `get_citations()` は空リストを返す

### WebSearchClient

SerpAPI 経由の Google Scholar 検索クライアント。

- SerpAPI URL: `https://serpapi.com/search`
- `engine: "google_scholar"`
- 著者情報: `publication_info.authors` から取得、フォールバックとして `summary` の先頭部分をパース
- 引用数: `inline_links.cited_by.total` から取得
- `get_references()` / `get_citations()` は空リストを返す

---

## Ranking (sera.phase0.ranking)

### citation_norm(citations, max_citations) -> float

対数スケールによる引用数の正規化。

```
citation_norm = log(1 + c) / log(1 + max_c)
```

`max_citations <= 0` の場合は `0.0` を返す。

### compute_ranking_score(citation_count, max_citations, relevance_score, citation_weight=0.6) -> float

統合ランキングスコアの計算。

```
score = citation_weight * citation_norm + (1 - citation_weight) * relevance_score
```

### rank_papers(papers, ranking_weight=0.6) -> list[PaperResult]

論文を統合ランキングスコアの降順でソートする。各論文の `relevance_score` 属性を使用（存在しない場合は 0.5）。入力リストは変更せず、新しいリストを返す。

---

## Clustering (sera.phase0.clustering)

### Cluster (dataclass)

論文のテーマ別クラスタ。

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `label` | `str` | クラスタ名 |
| `description` | `str` | クラスタの説明 |
| `paper_ids` | `list[str]` | クラスタに属する論文 ID |
| `keywords` | `list[str]` | クラスタのキーワード（LLM が JSON レスポンスで返す） |

### cluster_papers(papers, agent_llm=None) -> list[Cluster]

非同期関数。LLM ベースで論文をテーマ別クラスタに分類する。

- LLM が利用可能な場合: 論文リスト（ID, タイトル, 年, 引用数）を提示し、JSON 形式（`label`, `description`, `keywords`, `paper_ids`）でクラスタリングを依頼
- LLM レスポンスのパース: Markdown コードフェンスを除去後に JSON パース。`keywords` フィールドも抽出。存在する `paper_id` のみを含むクラスタを構築
- LLM が利用不可、またはパース失敗時: 全論文を含む単一の `"All Papers"` クラスタにフォールバック
- 論文リストが空の場合は空リストを返す
