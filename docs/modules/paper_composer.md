# PaperComposer / FigureGenerator / CitationSearcher / VLMReviewer

Phase 7 の論文生成を担当するモジュール群のドキュメント。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `PaperComposer` | `src/sera/paper/paper_composer.py` |
| `FigureGenerator` | `src/sera/paper/figure_generator.py` |
| `CitationSearcher` | `src/sera/paper/citation_searcher.py` |
| `VLMReviewer` | `src/sera/paper/vlm_reviewer.py` |

## 依存関係

- `matplotlib` (Agg バックエンド) -- グラフ生成
- `numpy` -- 数値計算
- `graphviz` (オプション) -- 探索木の可視化
- `openai` / `anthropic` (オプション) -- VLM プロバイダ

---

## Paper (dataclass)

生成された研究論文を表すデータクラス。

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `content` | `str` | Markdown 形式の論文本文 |
| `figures` | `list[Path]` | 図のファイルパスリスト |
| `bib_entries` | `list[dict]` | 参考文献エントリ |
| `metadata` | `dict` | メタデータ（output_dir, paper_path 等） |

出力先: `paper/paper.md`, `paper/paper.bib`, `paper/figure_descriptions.json`

---

## PaperComposer

6 ステップの論文作成パイプラインを統括するクラス。

### コンストラクタ

```python
def __init__(self, output_dir: str | Path, n_writeup_reflections: int = 3)
```

- `output_dir`: 論文の出力先ディレクトリ（自動作成）
- `n_writeup_reflections`: 執筆リフレクションループの最大回数（デフォルト 3）

### compose(evidence, paper_spec, teacher_papers, agent_llm, vlm, semantic_scholar_client) -> Paper

論文生成パイプライン全体を実行する非同期メソッド。`agent_llm` は必須（None の場合は `ValueError`）。

### 6 ステップパイプライン

#### Step 1: ログ要約 (_step1_log_summarization)

`EvidenceStore` から実験結果を構造化データとして抽出する。

- `evidence.get_experiment_summaries()` -- 実験サマリー
- `evidence.get_convergence_data()` -- 収束データ
- `evidence.get_ablation_data()` -- アブレーションデータ
- `evidence.get_main_results_table()` -- メイン結果テーブル
- 最良ノードの情報（hypothesis, mu, se, lcb, config）

出力: `experiment_summaries.json` としてディスクに保存。

#### Step 2: プロット集約 (_step2_plot_aggregation)

`FigureGenerator` でベース図を生成し、LLM で追加図を生成する。

生成される図:
1. **CI バーチャート** (`ci_bar_chart`): 全評価済みノードの mu +/- 1.96*SE
2. **収束曲線** (`convergence_curve`): ステップ vs best_lcb
3. **アブレーションテーブル** (`ablation_table`): アブレーションデータのグループ化棒グラフ（mu vs LCB）
4. **探索木** (`search_tree`): graphviz による DAG 可視化（gold=top, lightcoral=failed, lightblue=others）
5. **LLM 生成の追加プロット** (`aggregate_plots`): 最大 3 つ、リフレクションループ付き

#### Step 3: 引用検索 (_step3_citation_search)

`CitationSearcher.search_loop()` を呼び出す。

- コンテキスト: `summaries` の JSON 文字列の先頭 5000 文字
- 最大ラウンド数: 20

#### Step 4: VLM 図の説明 (_step4_vlm_descriptions)

VLM が有効な場合のみ実行。各図に対して `vlm.describe_figures(figures)` を呼び出し、`{filename: description}` の辞書を返す。VLM が無効（`None` または `enabled == False`）の場合は空辞書。

結果は `figure_descriptions.json` としてディスクに保存される（indent=2）。

#### Step 5: 論文本文生成 + リフレクション (_step5_paper_body)

3 段階で論文本文を生成する。

**Step 5a: アウトライン生成**
- 必須セクション（paper_spec から取得）、ティーチャー論文のスタイルガイダンス、結果テーブル、最良結果、利用可能な図と引用をコンテキストとして提示
- LLM にセクション別のアウトラインを生成させる

**Step 5b: 1 パス全文生成**
- アウラインと全エビデンスを用いて、Markdown 形式の完全な論文を 1 パスで生成

**Step 5c: リフレクションループ**（最大 `n_writeup_reflections` 回）

各ラウンドで以下のチェックを実行:
1. **未参照の図**: 論文本文に図のファイル名が含まれていない場合
2. **無効な `\cite{key}` 参照**: 参考文献リストにないキーの検出（カンマ区切りの複数キーにも対応）
3. **欠落セクション**: abstract, introduction, method, experiment, result, conclusion のいずれかが本文に含まれていない場合
4. **閉じていないコードブロック**: ` ``` ` の数が奇数の場合

VLM が有効な場合: 最初の 3 図に対して `vlm.review_figure_caption_refs()` を呼び出し、`suggestion` があればフィードバックに追加。

問題が見つからない場合はループを早期終了。

#### Step 6: 最終統合 (_step6_final_integration)

1. **図の番号付け**: 各図の画像参照を `![Figure N: caption](filename)` 形式にリナンバリング
2. **引用キーの一貫性確認**: BibTeX エントリのキーと本文内の `\cite{}` の整合性を確認
3. **参考文献セクションの追加**: `# References` セクションが存在しない場合、末尾に `## References` を追加。各エントリのフォーマット: `[key] title by authors (year)`
4. **BibTeX ファイル出力**: `paper/paper.bib` に全 bib_entries を `@article` 形式で書き出し。著者は `" and "` で結合

---

## FigureGenerator

出版品質の図を生成するクラス。

### 定数

| 定数 | 値 | 説明 |
|------|-----|------|
| `_MAX_FIGURES` | 12 | 生成可能な図の最大数 |
| `_DPI` | 300 | PNG 出力の解像度 |

matplotlib は `Agg` バックエンドを使用（非対話型）。

### コンストラクタ

```python
def __init__(self, output_dir: str | Path)
```

### ci_bar_chart(nodes, output_name="ci_bar_chart.png") -> Path

各メソッドの mu +/- 95% CI（1.96 * SE）を示すエラーバー付き棒グラフを生成する。

- X 軸: メソッド名（`experiment_config.get("method")` またはhypothesis の先頭 40 文字）
- Y 軸: メトリクス値
- SE が `inf` の場合は 0 として扱う

### convergence_curve(data, output_name="convergence_curve.png") -> Path

探索ステップ vs best_lcb の折れ線グラフを生成する。

- `data`: `list[tuple[int, float]]` 形式の `(step, best_lcb)` リスト
- データが空の場合は "No convergence data" のテキストを表示

### search_tree(nodes, top_n=10, output_name="search_tree.png") -> Path | None

graphviz による探索木の DAG 可視化を生成する。

- 色分け: gold（top-n ノード）、lightcoral（failed ノード）、lightblue（その他）
- ラベル: ノード ID の先頭 8 文字、仮説の先頭 30 文字、mu の値
- graphviz パッケージが未インストールの場合は `None` を返す（graceful degradation）

### ablation_table(data, output_name="ablation_table.png") -> Path

アブレーション結果のグループ化棒グラフを生成する。

- `data`: `dict[str, dict]` 形式（変数名 -> `{mu, se, lcb, config}`）
- 各変数に対して mu と LCB の並列棒グラフを表示
- データが空の場合は "No ablation data" のテキストを表示

### aggregate_plots(evidence, agent_llm, n_reflections=2) -> list[Path]

LLM にmatplotlib コードを生成させて追加図を作成する非同期メソッド。

1. LLM に実験エビデンスのサマリーを提示し、最大 3 つの追加 matplotlib プロットの Python コードを JSON で提案させる
2. 各コードを `exec()` で実行
3. 失敗時はリフレクションループで LLM にコードの修正を依頼（最大 `n_reflections` 回）
4. `_MAX_FIGURES` の上限を超える場合はスキップ

---

## CitationSearcher

反復的な引用発見と BibTeX 生成を行うクラス。

### コンストラクタ

```python
def __init__(
    self,
    semantic_scholar_client=None,  # S2 クライアント（None で検索スキップ）
    agent_llm=None,                # LLM クライアント（None で全体スキップ）
    log_dir=None,                  # ログディレクトリ
)
```

### search_loop(context, existing_bibtex="", max_rounds=20) -> list[dict]

反復的引用検索ループを実行する非同期メソッド。

**各ラウンドの処理:**

1. LLM に欠落している引用を 1 つ特定させる（CLAIM と QUERY を返すよう指示）
2. **早期終了**: LLM が `"no more citations needed"` と回答した場合にループを終了
3. Semantic Scholar で検索（`limit=10`）
4. LLM に最も適合する結果を番号で選択させる
5. **引用キー生成**: `{lastname}{year}` 形式。重複時はサフィックス（a, b, c...）を追加
6. LLM に BibTeX エントリを生成させる

**戻り値:** `citation_key`, `title`, `authors`, `year`, `bibtex`, `paper_id` をキーとする dict のリスト。

**ログ:** 各ラウンドの結果を `citation_search_log.jsonl` に記録。

---

## VLMReviewer

Vision-Language Model を使った図のレビュー・分析を行うクラス。

### コンストラクタ

```python
def __init__(self, model: str | None = None, provider: str | None = None)
```

- `model`: モデル識別子（例: `"gpt-4o"`, `"claude-sonnet-4-20250514"`）。`None` で VLM を無効化
- `provider`: `"openai"` または `"anthropic"`
- `self.enabled`: `model` と `provider` の両方が `None` でない場合に `True`

**プロバイダ初期化:**
- OpenAI: `openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])`
- Anthropic: `anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])`
- パッケージ未インストール時は `enabled = False` に設定（graceful degradation）

### 画像処理

- `_encode_image(path)`: ファイルを base64 エンコード
- サポート形式: PNG, JPEG, GIF, WEBP
- MIME タイプは拡張子から自動判定

### describe_figures(figure_paths) -> dict[str, str]

各図の VLM による説明を返す。

- 無効時: 全図に対して空文字列を返す
- 各図に対して個別に VLM を呼び出し
- プロンプト: プロットの種類、軸の意味、トレンド/パターン、注目すべき特徴を記述するよう指示

### review_figure_caption_refs(figure_path, caption, text_refs) -> dict[str, str]

図、キャプション、テキスト参照の整合性をレビューする。

**戻り値のキー:**
- `img_review`: 画像品質と内容のレビュー
- `caption_review`: キャプションの正確性
- `figrefs_review`: テキスト参照の整合性
- `informative`: `"yes"` / `"no"` / `"unknown"` -- 図が情報的で必要か
- `suggestion`: 改善提案

無効時またはファイル未存在時は全フィールドが空文字列（`informative` は `"unknown"`）。

### detect_duplicate_figures(figure_paths) -> list[dict]

類似または重複する図をペアワイズ比較で検出する。

- 最大比較ペア数: 15（`max_pairs`）
- 各ペアに対して VLM で類似度（0.0-1.0）と推奨（`keep_both` / `merge` / `remove_one`）を取得
- 類似度 > 0.5 のペアを結果に含める
- 無効時または図が 2 枚未満の場合は空リストを返す

---

## 出力ファイル一覧

| ファイル | 生成ステップ | 内容 |
|---------|------------|------|
| `paper/paper.md` | Step 5-6 | 論文本文（Markdown 形式） |
| `paper/paper.bib` | Step 6 | 参考文献（BibTeX 形式） |
| `paper/figure_descriptions.json` | Step 4 | VLM による各図の説明（`{filename: description}` 形式） |
| `paper/experiment_summaries.json` | Step 1 | 実験サマリーと最良ノード情報 |
| `paper/figures/*.png` | Step 2 | 生成された図表（CI バーチャート、収束曲線等） |

### paper.bib フォーマット

```bibtex
@article{citation_key,
  title = {Title of the Paper},
  author = {Author1 and Author2 and Author3},
  year = {2024},
  journal = {Venue Name},
  doi = {10.xxxx/xxxxx},
}
```

各 bib_entry の `citation_key` は `CitationSearcher` が生成する `{lastname}{year}` 形式。著者は `" and "` で結合される。
