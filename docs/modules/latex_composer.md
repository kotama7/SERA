# LaTeXComposer / _escape_latex / _section_name_to_command

PaperComposer の Markdown 出力を LaTeX に変換するモジュールのドキュメント。論文セクションを受け取り、コンパイル可能な完全な LaTeX ドキュメントを生成する。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `LaTeXComposer` | `src/sera/paper/latex_composer.py` |
| `_escape_latex` | `src/sera/paper/latex_composer.py` |
| `_section_name_to_command` | `src/sera/paper/latex_composer.py` |

## 依存関係

- `re` -- 正規表現による Markdown パース
- `pathlib.Path` -- 図のパス管理
- `sera.paper.paper_composer` (`Paper` dataclass) -- `compose_from_paper()` の入力型

---

## LATEX_TEMPLATE

LaTeX ドキュメントのベーステンプレート。標準的な学術論文用パッケージを含む。

```latex
\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage{geometry}
\geometry{margin=1in}

\title{%(title)s}
\author{%(author)s}
\date{%(date)s}

\begin{document}

\maketitle

%(body)s

%(bibliography)s

\end{document}
```

テンプレート変数は Python の `%` フォーマットで置換される: `title`, `author`, `date`, `body`, `bibliography`。

---

## LaTeXComposer

Markdown 論文セクションを完全な LaTeX ドキュメントに変換するクラス。

### コンストラクタ

```python
def __init__(self, figures_dir: str | Path | None = None) -> None
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `figures_dir` | `str \| Path \| None` | `None` | 図ファイルのディレクトリ。`\includegraphics` のパス解決に使用。`None` の場合、ファイル名をそのまま使用 |

---

### compose(sections, metadata=None) -> str

セクション dict とメタデータから完全な LaTeX ドキュメントを生成するメインメソッド。

```python
def compose(
    self,
    sections: dict[str, str],
    metadata: dict | None = None,
) -> str
```

**パラメータ:**

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `sections` | `dict[str, str]` | セクション名 -> Markdown コンテンツのマッピング。`"content"` キーで全文を渡すことも可能 |
| `metadata` | `dict \| None` | オプションのメタデータ。キー: `"title"`, `"author"`, `"date"` |

**メタデータのデフォルト値:**

| キー | デフォルト |
|------|----------|
| `title` | `"Untitled"` |
| `author` | `"SERA"` |
| `date` | `\today` |

**処理フロー:**

1. メタデータ値を `_escape_latex()` でエスケープ
2. `_build_body(sections)` でセクションを LaTeX 本文に変換
3. `_extract_bibliography(body)` で References セクションを抽出し `thebibliography` 環境に変換
4. `LATEX_TEMPLATE` に値を埋め込んで返す

---

### compose_from_paper(paper) -> str

`Paper` dataclass からの変換用コンビニエンスメソッド。

```python
def compose_from_paper(self, paper: Any) -> str
```

- `paper.content` を `{"content": paper.content}` として `compose()` に渡す
- `paper.metadata` をメタデータとして使用

---

### 内部メソッド

#### _build_body(sections) -> str

セクション dict を LaTeX 本文テキストに変換する。

**2 つのモード:**

| 条件 | 動作 |
|------|------|
| `"content"` キーが存在 | 全文を `_markdown_to_latex()` で一括変換 |
| 個別セクション | 各セクションを個別に変換。`abstract` は `\begin{abstract}...\end{abstract}` 環境を使用。それ以外は `\section{}` を付与 |

セクション名の変換には `_section_name_to_command()` を使用（例: `"related_work"` -> `"Related Work"`）。

---

#### _markdown_to_latex(text) -> str

Markdown テキストを LaTeX に変換する中核メソッド。以下の要素を処理する。

**見出し変換:**

| Markdown | LaTeX |
|----------|-------|
| `# Title` | `\section{Title}` |
| `## Title` | `\subsection{Title}` |
| `### Title` | `\subsubsection{Title}` |
| `#### Title` | `\paragraph{Title}` |
| `# Abstract` | `\begin{abstract}`（特殊処理） |

**コードブロック:**

````
```python
code here
```
````

は以下に変換される:

```latex
\begin{verbatim}
code here
\end{verbatim}
```

コード言語指定（` ```python ` 等）は認識されるが、LaTeX 出力では `verbatim` 環境を統一的に使用する。

**画像参照:**

```markdown
![caption text](path/to/image.png)
```

は `_make_figure()` で LaTeX `figure` 環境に変換される。

**テーブル:**

`|` で始まり `|` で終わる行の連続はテーブルとして認識され、`_convert_table()` で `tabular` 環境に変換される。

**インラインフォーマット:**

`_convert_inline()` で処理される。

**状態管理:** コードブロック内とテーブル内をフラグで追跡し、コンテキストに応じた変換を行う。未閉じのコードブロックやテーブルはメソッド末尾でフラッシュされる。

---

#### _convert_inline(line) -> str

インライン Markdown フォーマットを LaTeX に変換する。

| Markdown | LaTeX | 正規表現パターン |
|----------|-------|----------------|
| `**text**` | `\textbf{text}` | `\*\*(.+?)\*\*` |
| `*text*` | `\textit{text}` | `\*(.+?)\*` |
| `` `text` `` | `\texttt{text}` | `` `([^`]+)` `` |
| `[text](url)` | `\href{url}{text}` | `\[([^\]]+)\]\(([^)]+)\)` |

**処理順序:** bold -> italic -> inline code -> links。`\cite` コマンドは既に LaTeX 形式のため変換しない。

---

#### _make_figure(img_path, caption) -> str

LaTeX `figure` 環境を生成する。

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{path/to/image.png}
\caption{Caption text}
\end{figure}
```

- `figures_dir` が設定されている場合: `figures_dir / filename` をパスとして使用
- パスのバックスラッシュはスラッシュに変換
- キャプションは `_escape_latex()` でエスケープ

---

#### _convert_table(table_lines) -> str

Markdown テーブルを LaTeX `table` + `tabular` 環境に変換する。

**処理フロー:**

1. 各行の `|` で区切られたセルをパース
2. セパレータ行（`|---|---|` 等）を検出してスキップ
3. 列数を最大セル数から決定
4. 列指定: 全列左揃え（`"l" * n_cols`）
5. `booktabs` パッケージの罫線を使用:
   - `\toprule`: テーブル上部
   - `\midrule`: ヘッダ行の下（最初の行の後、行数 > 1 の場合）
   - `\bottomrule`: テーブル下部

**出力例:**

```latex
\begin{table}[htbp]
\centering
\begin{tabular}{lll}
\toprule
Header 1 & Header 2 & Header 3 \\
\midrule
Cell 1 & Cell 2 & Cell 3 \\
Cell 4 & Cell 5 & Cell 6 \\
\bottomrule
\end{tabular}
\end{table}
```

各セルの内容は `_escape_latex()` でエスケープされる。列数が不揃いの行は空文字列でパディングされる。

---

#### _extract_bibliography(body) -> tuple[str, str]

本文から References セクションを抽出し、`thebibliography` 環境に変換する。

**処理フロー:**

1. `\section{References}` を正規表現で検索
2. マッチしない場合: `(body, "")` を返す
3. References セクション内の各行から `[key] description` 形式のエントリをパース
4. `\bibitem{key} description` 形式に変換

**出力例:**

```latex
\begin{thebibliography}{99}
\bibitem{smith2024} Title by Authors (2024)
\bibitem{jones2023} Another Title by Other Authors (2023)
\end{thebibliography}
```

エントリが見つからない場合は空文字列を返す。

---

## _escape_latex (モジュールレベル関数)

LaTeX の特殊文字をエスケープする。既存の LaTeX コマンド（`\cite`, `\textbf` 等）を壊さないよう、バックスラッシュはエスケープ対象外とする。

```python
def _escape_latex(text: str) -> str
```

**エスケープ対象:**

| 文字 | エスケープ後 |
|------|------------|
| `&` | `\&` |
| `%` | `\%` |
| `$` | `\$` |
| `#` | `\#` |
| `_` | `\_` |
| `{` | `\{` |
| `}` | `\}` |
| `~` | `\textasciitilde{}` |
| `^` | `\textasciicircum{}` |

空文字列の場合はそのまま返す。

---

## _section_name_to_command (モジュールレベル関数)

セクションキー名を表示用タイトルに変換する。

```python
def _section_name_to_command(name: str) -> str
```

- アンダースコアをスペースに置換
- 各単語の先頭を大文字化（`title()`）
- 例: `"related_work"` -> `"Related Work"`, `"introduction"` -> `"Introduction"`

---

## 使用例

```python
from sera.paper.latex_composer import LaTeXComposer

# セクション dict から LaTeX を生成
composer = LaTeXComposer(figures_dir="paper/figures")
sections = {
    "abstract": "We present a novel approach...",
    "introduction": "Background and motivation...",
    "method": "## Our Method\nWe propose...",
    "results": "| Method | Score |\n|--------|-------|\n| Ours | **0.95** |",
}
metadata = {"title": "My Research Paper", "author": "SERA Agent"}
latex_source = composer.compose(sections, metadata)

# Paper dataclass から LaTeX を生成
latex_source = composer.compose_from_paper(paper)
```
