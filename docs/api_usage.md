# SERA API 利用ガイド

このドキュメントでは、SERA の外部 API 連携システムについて説明します。Phase 0（先行研究収集）と Phase 7（引用検索）で使用される API クライアントの実装詳細を記載しています。

## プロバイダ優先順位

`sera.phase0.related_work_engine.RelatedWorkEngine._search_with_fallback()` は、登録されたクライアントを順番に試行し、**最初に成功した結果**を返します:

```python
async def _search_with_fallback(self, query, limit, year_from):
    for client in self._clients:
        try:
            results = await client.search(query, limit=limit, year_from=year_from)
            if results:
                return results
        except Exception as exc:
            logger.warning("search_client_failed", client=type(client).__name__, ...)
    return []
```

クライアントのリストはエンジン初期化時に渡される順序で試行されます。デフォルトの優先順位は:

1. **Semantic Scholar** -- 最も包括的な学術データベース
2. **CrossRef** -- DOI ベースのメタデータ
3. **arXiv** -- プレプリントの検索
4. **Web Search (SerpAPI)** -- 上位 3 つで不十分な場合のフォールバック

## 共通インターフェース

すべての API クライアントは `sera.phase0.api_clients.base.BaseScholarClient` ABC を実装しています:

```python
class BaseScholarClient(ABC):
    @abstractmethod
    async def search(
        self, query: str, limit: int = 20, year_from: int | None = None
    ) -> list[PaperResult]: ...

    @abstractmethod
    async def get_references(
        self, paper_id: str, limit: int = 20
    ) -> list[PaperResult]: ...

    @abstractmethod
    async def get_citations(
        self, paper_id: str, limit: int = 20
    ) -> list[PaperResult]: ...
```

ソースファイル: `src/sera/phase0/api_clients/base.py`

## PaperResult データクラス

すべてのクライアントが返す統一的な論文表現です:

```python
@dataclass
class PaperResult:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    abstract: str = ""
    citation_count: int = 0
    url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    source_api: str = ""
    relevance_score: float = 0.5
```

ソースファイル: `src/sera/phase0/api_clients/base.py`

## API クライアント詳細

### 1. SemanticScholarClient

学術論文検索の主要プロバイダです。検索、参照取得、被引用取得をすべてサポートします。

| 項目 | 値 |
|---|---|
| ソースファイル | `src/sera/phase0/api_clients/semantic_scholar.py` |
| ベース URL | `https://api.semanticscholar.org/graph/v1` |
| 認証 | `x-api-key` ヘッダー（`SEMANTIC_SCHOLAR_API_KEY` 環境変数から取得） |
| HTTP クライアント | `httpx.AsyncClient`（タイムアウト: 30 秒） |

**取得フィールド:**

```
paperId, title, abstract, year, citationCount, authors, venue, externalIds, url
```

**サポートするメソッド:**

| メソッド | エンドポイント | 説明 |
|---|---|---|
| `search()` | `GET /paper/search` | キーワード検索。`year_from` パラメータで `year={year_from}-` フィルタを適用 |
| `get_references()` | `GET /paper/{paper_id}/references` | 論文の参照先一覧を取得 |
| `get_citations()` | `GET /paper/{paper_id}/citations` | 論文の被引用一覧を取得 |

**レスポンス解析:**

`references` / `citations` エンドポイントのレスポンスでは、論文データが `citedPaper` または `citingPaper` キーの下にネストされています。`_parse_paper()` 関数がこれらの形式を統一的に処理します。

`externalIds` から DOI（`ExternalIds.DOI`）と arXiv ID（`ExternalIds.ArXiv`）を抽出します。

**リトライ戦略:**

```python
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    reraise=True,
)
```

- 最大 5 回試行
- 指数バックオフ: 初期 1 秒、最大 60 秒
- `httpx.HTTPStatusError` でリトライ（429 や 5xx を含む）
- 上記以外のエラーは即座に失敗

### 2. CrossRefClient

DOI ベースの書誌メタデータプロバイダです。

| 項目 | 値 |
|---|---|
| ソースファイル | `src/sera/phase0/api_clients/crossref.py` |
| ベース URL | `https://api.crossref.org/works` |
| 認証 | `mailto` クエリパラメータ（`CROSSREF_EMAIL` 環境変数から取得）で polite pool を利用 |
| HTTP クライアント | `httpx.AsyncClient`（タイムアウト: 30 秒） |

**サポートするメソッド:**

| メソッド | 動作 |
|---|---|
| `search()` | `GET /works` でキーワード検索。`sort=relevance`、`year_from` は `filter=from-pub-date:{year}` で適用 |
| `get_references()` | 常に `[]` を返す（CrossRef API は参照グラフを提供しない） |
| `get_citations()` | 常に `[]` を返す（CrossRef API は被引用グラフを提供しない） |

**paper_id のフォーマット:** `"crossref:{doi}"`

**アブストラクトの処理:**

CrossRef のアブストラクトには JATS XML タグが含まれる場合があります。`_parse_item()` で `re.sub(r"<[^>]+>", "", abstract)` により XML タグが除去されます。

**年の解決順序:**

`published-print` → `published-online` → `created` の順で `date-parts` を検索します。

**被引用数:**

`is-referenced-by-count` フィールドから取得します。

### 3. ArxivClient

プレプリント検索用のクライアントです。

| 項目 | 値 |
|---|---|
| ソースファイル | `src/sera/phase0/api_clients/arxiv.py` |
| ベース URL | `http://export.arxiv.org/api/query` |
| 認証 | 不要（IP ベースのレート制限） |
| HTTP クライアント | `httpx.AsyncClient`（タイムアウト: 30 秒） |
| レート制限 | リクエスト間で最低 3 秒の間隔を確保 |
| XML パーサ | `xml.etree.ElementTree`（Atom フォーマット） |

**サポートするメソッド:**

| メソッド | 動作 |
|---|---|
| `search()` | `search_query=all:{query}` でクエリ。`sortBy=relevance`。`year_from` はクライアント側でフィルタ（arXiv API はサーバー側の年フィルタを提供しない） |
| `get_references()` | 常に `[]` を返す（arXiv API は参照/被引用グラフを提供しない） |
| `get_citations()` | 常に `[]` を返す |

**paper_id のフォーマット:** `"arxiv:{arxiv_id}"`（例: `"arxiv:2301.12345"`）

**制限事項:**
- `citation_count` は常に `0`（arXiv API は引用数を提供しない）
- `venue` は `journal_ref` フィールドから取得（空の場合が多い）

**レート制限の実装:**

```python
async def _rate_limit(self) -> None:
    elapsed = time.monotonic() - self._last_request_time
    if elapsed < 3.0 and self._last_request_time > 0:
        await asyncio.sleep(3.0 - elapsed)
    self._last_request_time = time.monotonic()
```

### 4. WebSearchClient (SerpAPI)

Google Scholar 検索のラッパーです。上位 3 プロバイダで十分な結果が得られない場合のフォールバックとして使用されます。

| 項目 | 値 |
|---|---|
| ソースファイル | `src/sera/phase0/api_clients/web_search.py` |
| URL | `https://serpapi.com/search` |
| 検索エンジン | `engine=google_scholar` |
| 認証 | `api_key` クエリパラメータ（`SERPAPI_API_KEY` 環境変数から取得） |
| HTTP クライアント | `httpx.AsyncClient`（タイムアウト: 30 秒） |

**サポートするメソッド:**

| メソッド | 動作 |
|---|---|
| `search()` | `q={query}`, `num={limit}` でクエリ。`year_from` は `as_ylo` パラメータで適用 |
| `get_references()` | 常に `[]` を返す |
| `get_citations()` | 常に `[]` を返す |

**レスポンス解析:**

`organic_results` 配列の各要素から以下を抽出:
- `title`: 論文タイトル
- `snippet`: アブストラクトの代替として使用
- `publication_info.authors`: 著者リスト（存在しない場合は `summary` から解析）
- `inline_links.cited_by.total`: 被引用数
- 年は `publication_info.summary` から正規表現 `\b(19|20)\d{2}\b` で抽出
- `paper_id` は `result_id` フィールド、存在しない場合は `"serpapi:{index}"` を使用

## Phase 7 の引用検索

`sera.paper.citation_searcher.CitationSearcher` は Phase 7（論文生成）で使用され、論文ドラフトに必要な引用を反復的に検索します。

### 動作フロー

最大 `max_rounds` ラウンド（デフォルト: 20）の反復ループで動作します:

1. **引用特定**: LLM にドラフトの文脈を提示し、不足している引用を 1 件特定させる
2. **早期終了判定**: LLM が `"No more citations needed"` と応答した場合、ループを終了
3. **検索クエリ抽出**: LLM の応答から `CLAIM:` と `QUERY:` を解析
4. **Semantic Scholar 検索**: 抽出したクエリで `SemanticScholarClient.search()` を呼び出し（上限 10 件）
5. **最適一致選択**: 検索結果を LLM に提示し、最も適切な論文を番号で選択させる
6. **BibTeX 生成**: 選択した論文の情報を LLM に渡し、BibTeX エントリを生成させる

### 引用キーの命名規則

`{第一著者の姓(小文字)}{年}` の形式で生成されます。重複する場合はサフィックス `a`, `b`, `c`, ... が付加されます。

```python
citation_key = f"{first_author.lower()}{year or 'nd'}"
# 重複時: base_key + 'a', base_key + 'b', ...
```

### ログ

各ラウンドの結果は `paper/citation_search_log.jsonl` に JSONL 形式で記録されます:

```json
{
  "round": 0,
  "action": "citation_found",
  "query": "attention mechanism transformer",
  "claim": "Self-attention enables parallel processing of sequences",
  "citation_key": "vaswani2017",
  "title": "Attention Is All You Need",
  "year": 2017
}
```

`action` の値:
- `"citation_found"`: 引用が見つかり追加された
- `"no_results"`: 検索結果が空だった
- `"no_selection"`: LLM が適切な論文を選択しなかった
- `"early_exit"`: LLM がこれ以上の引用は不要と判断した

ソースファイル: `src/sera/paper/citation_searcher.py`

## リトライ戦略

### Semantic Scholar のリトライ（tenacity ライブラリ使用）

Semantic Scholar クライアントの全メソッド（`search`, `get_references`, `get_citations`）に `tenacity` デコレータが適用されています:

| 設定 | 値 |
|---|---|
| 最大試行回数 | 5 |
| バックオフ方式 | 指数バックオフ（`multiplier=1`） |
| 最小待機時間 | 1 秒 |
| 最大待機時間 | 60 秒 |
| リトライ対象 | `httpx.HTTPStatusError`（429 や 5xx を含む） |
| `reraise` | `True`（最終試行失敗時に例外を再送出） |

### CrossRef / arXiv / SerpAPI のリトライ

これら 3 つのクライアントには tenacity によるリトライデコレータは適用されていません。例外は `RelatedWorkEngine._search_with_fallback()` で捕捉され、次のプロバイダにフォールバックします。

### フォールバック動作の詳細

`_search_with_fallback()` は以下の動作をします:

1. クライアントリストの先頭から順に `search()` を呼び出す
2. 結果が空でなければ即座に返す
3. 例外が発生した場合はログに警告を記録し、次のクライアントに進む
4. すべてのクライアントが失敗した場合は空のリストを返す

## 引用グラフ展開

`RelatedWorkEngine._expand_citations()` は、初期検索結果の上位論文に対して参照・被引用グラフを展開します:

1. 各論文に対して、登録されたクライアントの順に `get_references()` と `get_citations()` を呼び出す
2. いずれかのクライアントからグラフデータが得られた時点で、その論文の展開を完了
3. Semantic Scholar のみが参照・被引用グラフをサポートするため、実質的には Semantic Scholar から取得される
4. CrossRef、arXiv、SerpAPI の `get_references()` / `get_citations()` は常に空のリストを返す

展開の深さは `Phase0Config.citation_graph_depth`（デフォルト: 1）で制御されます。
