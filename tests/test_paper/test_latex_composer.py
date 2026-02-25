"""Tests for LaTeXComposer."""

from __future__ import annotations

from pathlib import Path

import pytest

from sera.paper.latex_composer import LaTeXComposer, _escape_latex, _section_name_to_command
from sera.paper.paper_composer import Paper


class TestEscapeLatex:
    """Test the _escape_latex helper."""

    def test_ampersand(self):
        assert _escape_latex("A & B") == "A \\& B"

    def test_percent(self):
        assert _escape_latex("100%") == "100\\%"

    def test_dollar(self):
        assert _escape_latex("$10") == "\\$10"

    def test_hash(self):
        assert _escape_latex("#1") == "\\#1"

    def test_underscore(self):
        assert _escape_latex("my_var") == "my\\_var"

    def test_empty_string(self):
        assert _escape_latex("") == ""

    def test_no_specials(self):
        assert _escape_latex("Hello World") == "Hello World"

    def test_multiple_specials(self):
        result = _escape_latex("A & B & C")
        assert result == "A \\& B \\& C"


class TestSectionNameToCommand:
    """Test section name conversion."""

    def test_simple(self):
        assert _section_name_to_command("introduction") == "Introduction"

    def test_underscore(self):
        assert _section_name_to_command("related_work") == "Related Work"

    def test_already_titled(self):
        assert _section_name_to_command("Abstract") == "Abstract"


class TestLaTeXComposerCompose:
    """Test the compose method."""

    def test_basic_sections(self):
        composer = LaTeXComposer()
        sections = {
            "abstract": "We present a method.",
            "introduction": "Background and motivation.",
        }
        result = composer.compose(sections, {"title": "My Paper", "author": "Author"})

        assert "\\documentclass{article}" in result
        assert "\\title{My Paper}" in result
        assert "\\author{Author}" in result
        assert "\\begin{document}" in result
        assert "\\end{document}" in result
        assert "We present a method." in result
        assert "Background and motivation." in result

    def test_default_metadata(self):
        composer = LaTeXComposer()
        result = composer.compose({"introduction": "text"})

        assert "\\title{Untitled}" in result
        assert "\\author{SERA}" in result

    def test_full_content_key(self):
        """Sections dict with single 'content' key."""
        composer = LaTeXComposer()
        md_content = "# Introduction\n\nSome text.\n\n# Method\n\nOur approach."
        result = composer.compose({"content": md_content})

        assert "\\section{Introduction}" in result
        assert "Some text." in result
        assert "\\section{Method}" in result
        assert "Our approach." in result

    def test_standard_packages(self):
        composer = LaTeXComposer()
        result = composer.compose({"abstract": "test"})

        assert "\\usepackage{amsmath}" in result
        assert "\\usepackage{graphicx}" in result
        assert "\\usepackage{booktabs}" in result
        assert "\\usepackage{hyperref}" in result


class TestMarkdownToLatex:
    """Test Markdown to LaTeX conversion."""

    def test_headings(self):
        composer = LaTeXComposer()
        md = "# Section\n\n## Subsection\n\n### Subsubsection\n\n#### Paragraph"
        result = composer.compose({"content": md})

        assert "\\section{Section}" in result
        assert "\\subsection{Subsection}" in result
        assert "\\subsubsection{Subsubsection}" in result
        assert "\\paragraph{Paragraph}" in result

    def test_bold_italic(self):
        composer = LaTeXComposer()
        md = "This is **bold** and *italic* text."
        result = composer.compose({"content": md})

        assert "\\textbf{bold}" in result
        assert "\\textit{italic}" in result

    def test_inline_code(self):
        composer = LaTeXComposer()
        md = "Use `pip install` to install."
        result = composer.compose({"content": md})

        assert "\\texttt{pip install}" in result

    def test_code_block(self):
        composer = LaTeXComposer()
        md = "```python\nprint('hello')\n```"
        result = composer.compose({"content": md})

        assert "\\begin{verbatim}" in result
        assert "print('hello')" in result
        assert "\\end{verbatim}" in result

    def test_image_to_figure(self):
        composer = LaTeXComposer()
        md = "![My Figure](figures/plot.png)"
        result = composer.compose({"content": md})

        assert "\\begin{figure}" in result
        assert "\\includegraphics" in result
        assert "figures/plot.png" in result
        assert "\\caption{My Figure}" in result
        assert "\\end{figure}" in result

    def test_image_with_figures_dir(self):
        composer = LaTeXComposer(figures_dir="/paper/figures")
        md = "![caption](plot.png)"
        result = composer.compose({"content": md})

        assert "/paper/figures/plot.png" in result

    def test_links_to_href(self):
        composer = LaTeXComposer()
        md = "See [this paper](https://example.com)."
        result = composer.compose({"content": md})

        assert "\\href{https://example.com}{this paper}" in result

    def test_cite_preserved(self):
        """LaTeX \\cite commands should pass through unchanged."""
        composer = LaTeXComposer()
        md = "As shown by \\cite{smith2023}."
        result = composer.compose({"content": md})

        assert "\\cite{smith2023}" in result


class TestTableConversion:
    """Test Markdown table to LaTeX tabular conversion."""

    def test_basic_table(self):
        composer = LaTeXComposer()
        md = (
            "| Method | Score |\n"
            "|--------|-------|\n"
            "| A      | 0.9   |\n"
            "| B      | 0.8   |"
        )
        result = composer.compose({"content": md})

        assert "\\begin{tabular}" in result
        assert "\\toprule" in result
        assert "\\midrule" in result
        assert "\\bottomrule" in result
        assert "\\end{tabular}" in result
        assert "Method" in result
        assert "0.9" in result

    def test_table_with_special_chars(self):
        composer = LaTeXComposer()
        md = (
            "| Method | Score |\n"
            "|--------|-------|\n"
            "| A & B  | 95%   |"
        )
        result = composer.compose({"content": md})

        assert "A \\& B" in result
        assert "95\\%" in result


class TestBibliographyExtraction:
    """Test bibliography extraction from body."""

    def test_extracts_references(self):
        composer = LaTeXComposer()
        md = (
            "# Introduction\n\nSome text.\n\n"
            "# References\n\n"
            "[smith2023] A Great Paper by Smith (2023)\n"
            "[jones2024] Another Paper by Jones (2024)"
        )
        result = composer.compose({"content": md})

        assert "\\begin{thebibliography}" in result
        assert "\\bibitem{smith2023}" in result
        assert "\\bibitem{jones2024}" in result
        # References section should be removed from body
        assert "\\section{References}" not in result

    def test_no_references_section(self):
        composer = LaTeXComposer()
        md = "# Introduction\n\nJust text, no references."
        result = composer.compose({"content": md})

        assert "\\begin{thebibliography}" not in result


class TestComposeFromPaper:
    """Test the compose_from_paper convenience method."""

    def test_from_paper_dataclass(self):
        paper = Paper(
            content="# Abstract\n\nWe present results.\n\n# Introduction\n\nMotivation.",
            figures=[Path("fig1.png")],
            bib_entries=[{"citation_key": "a2023", "title": "Title"}],
            metadata={"title": "Test Paper", "author": "Test Author"},
        )
        composer = LaTeXComposer()
        result = composer.compose_from_paper(paper)

        assert "\\documentclass{article}" in result
        assert "\\title{Test Paper}" in result
        assert "\\author{Test Author}" in result
        assert "We present results." in result

    def test_from_paper_empty_metadata(self):
        paper = Paper(content="Some text.", metadata={})
        composer = LaTeXComposer()
        result = composer.compose_from_paper(paper)

        assert "\\title{Untitled}" in result
        assert "\\author{SERA}" in result
