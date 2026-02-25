"""LaTeXComposer -- converts PaperComposer Markdown output to LaTeX.

Takes paper sections (from PaperComposer output) and formats them into a
complete LaTeX document with standard academic paper packages.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Basic LaTeX document template with standard academic packages
LATEX_TEMPLATE = r"""\documentclass{article}

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
"""


class LaTeXComposer:
    """Convert Markdown paper sections to a complete LaTeX document.

    Takes the paper sections dict (from PaperComposer output) and optional
    metadata, and produces a compilable LaTeX source string.

    Parameters
    ----------
    figures_dir : str | Path | None
        Directory where figures are located (for \\includegraphics paths).
        If None, figure filenames are used as-is.
    """

    def __init__(self, figures_dir: str | Path | None = None) -> None:
        self.figures_dir = Path(figures_dir) if figures_dir else None

    def compose(self, sections: dict[str, str], metadata: dict | None = None) -> str:
        """Compose a complete LaTeX document from sections and metadata.

        Parameters
        ----------
        sections : dict[str, str]
            Mapping of section names to Markdown content.
            Example: {"abstract": "We present...", "introduction": "Background..."}
            Alternatively, a single key "content" may hold the full Markdown paper.
        metadata : dict | None
            Optional metadata with keys: "title", "author", "date".
            Defaults are provided for missing keys.

        Returns
        -------
        str
            Complete LaTeX source ready for compilation.
        """
        metadata = metadata or {}
        title = _escape_latex(metadata.get("title", "Untitled"))
        author = _escape_latex(metadata.get("author", "SERA"))
        date = _escape_latex(metadata.get("date", r"\today"))

        # Build body from sections
        body = self._build_body(sections)

        # Extract and format bibliography
        body, bibliography = self._extract_bibliography(body)

        return LATEX_TEMPLATE % {
            "title": title,
            "author": author,
            "date": date,
            "body": body,
            "bibliography": bibliography,
        }

    def compose_from_paper(self, paper: Any) -> str:
        """Compose LaTeX from a Paper dataclass (convenience method).

        Parameters
        ----------
        paper : Paper
            A Paper dataclass with ``content``, ``figures``, ``bib_entries``,
            and ``metadata`` fields.

        Returns
        -------
        str
            Complete LaTeX source.
        """
        sections = {"content": paper.content}
        metadata = dict(paper.metadata) if paper.metadata else {}
        return self.compose(sections, metadata)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_body(self, sections: dict[str, str]) -> str:
        """Convert sections dict into LaTeX body text."""
        parts: list[str] = []

        if "content" in sections:
            # Single full-document content -- parse sections from Markdown
            parts.append(self._markdown_to_latex(sections["content"]))
        else:
            # Individual sections
            for section_name, content in sections.items():
                latex_content = self._markdown_to_latex(content)
                section_cmd = _section_name_to_command(section_name)
                if section_name.lower() == "abstract":
                    parts.append(
                        "\\begin{abstract}\n"
                        + latex_content
                        + "\n\\end{abstract}"
                    )
                else:
                    parts.append(f"\\section{{{_escape_latex(section_cmd)}}}\n{latex_content}")

        return "\n\n".join(parts)

    def _markdown_to_latex(self, text: str) -> str:
        """Convert Markdown text to LaTeX.

        Handles headings, bold, italic, images, code blocks, inline code,
        Markdown tables, and citation references.
        """
        lines = text.split("\n")
        result: list[str] = []
        in_code_block = False
        code_lang = ""
        code_lines: list[str] = []
        table_lines: list[str] = []
        in_table = False

        for line in lines:
            # Code block boundaries
            if line.strip().startswith("```"):
                if in_code_block:
                    # End code block
                    code_content = "\n".join(code_lines)
                    result.append(
                        "\\begin{verbatim}\n"
                        + code_content
                        + "\n\\end{verbatim}"
                    )
                    code_lines = []
                    in_code_block = False
                else:
                    # Flush table if active
                    if in_table:
                        result.append(self._convert_table(table_lines))
                        table_lines = []
                        in_table = False
                    # Start code block
                    in_code_block = True
                    code_lang = line.strip()[3:].strip()
                continue

            if in_code_block:
                code_lines.append(line)
                continue

            # Detect table lines (contain | and are not headings)
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                if not in_table:
                    in_table = True
                table_lines.append(stripped)
                continue
            elif in_table:
                # End of table
                result.append(self._convert_table(table_lines))
                table_lines = []
                in_table = False

            # Headings (# to ####)
            heading_match = re.match(r"^(#{1,4})\s+(.+)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                if title_text.lower() == "abstract":
                    result.append("\\begin{abstract}")
                    # The abstract content follows in subsequent lines.
                    # We'll handle the end marker when we encounter the next heading.
                    continue
                latex_title = _escape_latex(title_text)
                if level == 1:
                    result.append(f"\\section{{{latex_title}}}")
                elif level == 2:
                    result.append(f"\\subsection{{{latex_title}}}")
                elif level == 3:
                    result.append(f"\\subsubsection{{{latex_title}}}")
                else:
                    result.append(f"\\paragraph{{{latex_title}}}")
                continue

            # Image references: ![caption](path)
            img_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line)
            if img_match:
                caption = img_match.group(1)
                img_path = img_match.group(2)
                result.append(self._make_figure(img_path, caption))
                continue

            # Convert inline formatting for normal text lines
            converted = self._convert_inline(line)
            result.append(converted)

        # Flush remaining table
        if in_table and table_lines:
            result.append(self._convert_table(table_lines))

        # Flush unclosed code block
        if in_code_block and code_lines:
            code_content = "\n".join(code_lines)
            result.append(
                "\\begin{verbatim}\n"
                + code_content
                + "\n\\end{verbatim}"
            )

        return "\n".join(result)

    def _convert_inline(self, line: str) -> str:
        """Convert inline Markdown formatting to LaTeX."""
        # Bold: **text** -> \textbf{text}
        line = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", line)
        # Italic: *text* -> \textit{text}
        line = re.sub(r"\*(.+?)\*", r"\\textit{\1}", line)
        # Inline code: `text` -> \texttt{text}
        line = re.sub(r"`([^`]+)`", r"\\texttt{\1}", line)
        # Markdown links: [text](url) -> \href{url}{text}
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\\href{\2}{\1}", line)
        # \cite commands are already LaTeX -- leave them as-is
        return line

    def _make_figure(self, img_path: str, caption: str) -> str:
        """Generate a LaTeX figure environment for an image.

        Parameters
        ----------
        img_path : str
            Path or filename of the image.
        caption : str
            Figure caption text.

        Returns
        -------
        str
            LaTeX figure environment.
        """
        # Determine the graphics path
        if self.figures_dir is not None:
            gfx_path = str(self.figures_dir / Path(img_path).name)
        else:
            gfx_path = img_path

        # Escape special characters in path for LaTeX
        gfx_path = gfx_path.replace("\\", "/")
        escaped_caption = _escape_latex(caption)

        return (
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.8\\textwidth]{{{gfx_path}}}\n"
            f"\\caption{{{escaped_caption}}}\n"
            "\\end{figure}"
        )

    def _convert_table(self, table_lines: list[str]) -> str:
        """Convert Markdown table lines to a LaTeX tabular environment.

        Parameters
        ----------
        table_lines : list[str]
            Lines of the Markdown table (each starts and ends with |).

        Returns
        -------
        str
            LaTeX table/tabular environment.
        """
        if not table_lines:
            return ""

        # Parse cells from each row
        rows: list[list[str]] = []
        separator_indices: list[int] = []
        for i, line in enumerate(table_lines):
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Check if this is a separator row (e.g., |---|---|)
            if all(re.match(r"^[-:]+$", c.strip()) for c in cells if c.strip()):
                separator_indices.append(i)
                continue
            rows.append(cells)

        if not rows:
            return ""

        n_cols = max(len(r) for r in rows)
        col_spec = "l" * n_cols

        parts: list[str] = []
        parts.append("\\begin{table}[htbp]")
        parts.append("\\centering")
        parts.append(f"\\begin{{tabular}}{{{col_spec}}}")
        parts.append("\\toprule")

        for i, row in enumerate(rows):
            # Pad row to n_cols
            padded = row + [""] * (n_cols - len(row))
            escaped = [_escape_latex(cell) for cell in padded]
            parts.append(" & ".join(escaped) + " \\\\")
            if i == 0 and len(rows) > 1:
                parts.append("\\midrule")

        parts.append("\\bottomrule")
        parts.append("\\end{tabular}")
        parts.append("\\end{table}")

        return "\n".join(parts)

    def _extract_bibliography(self, body: str) -> tuple[str, str]:
        """Extract References section from body and convert to bibliography.

        If the body contains a \\section{References} (or similar), extract it
        and convert to a thebibliography environment. Otherwise return empty
        bibliography.

        Parameters
        ----------
        body : str
            LaTeX body text.

        Returns
        -------
        tuple[str, str]
            (body_without_references, bibliography_latex)
        """
        # Look for \section{References} and everything after it
        ref_pattern = r"\\section\{References\}(.*?)(?=\\section\{|$)"
        match = re.search(ref_pattern, body, re.DOTALL)
        if not match:
            return body, ""

        ref_content = match.group(1).strip()
        body_clean = body[: match.start()].rstrip()

        # Parse reference entries (format: [key] Title by Authors (Year))
        bib_items: list[str] = []
        for line in ref_content.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Match entries like: [key] Title by Authors (Year)
            entry_match = re.match(r"\[([^\]]+)\]\s*(.+)", line)
            if entry_match:
                key = entry_match.group(1).strip()
                desc = entry_match.group(2).strip()
                bib_items.append(f"\\bibitem{{{key}}} {desc}")

        if not bib_items:
            return body_clean, ""

        bib_env = (
            "\\begin{thebibliography}{99}\n"
            + "\n".join(bib_items)
            + "\n\\end{thebibliography}"
        )
        return body_clean, bib_env


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text.

    Does NOT escape backslash-prefixed commands (like \\cite, \\textbf, etc.)
    to preserve existing LaTeX commands.

    Parameters
    ----------
    text : str
        Raw text to escape.

    Returns
    -------
    str
        Text with special characters escaped.
    """
    if not text:
        return text
    # Characters that need escaping in LaTeX (order matters)
    # We skip backslash to avoid breaking existing LaTeX commands
    specials = [
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for char, replacement in specials:
        text = text.replace(char, replacement)
    return text


def _section_name_to_command(name: str) -> str:
    """Convert a section key name to a display title.

    Parameters
    ----------
    name : str
        Section key (e.g. "introduction", "related_work").

    Returns
    -------
    str
        Display title (e.g. "Introduction", "Related Work").
    """
    return name.replace("_", " ").title()
