"""CitationSearcher per S11.5 - iterative citation discovery and BibTeX generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CitationSearcher:
    """Iteratively discovers missing citations for a paper draft.

    Each round:
      1. LLM identifies a missing citation from the context.
      2. LLM generates a search query.
      3. Semantic Scholar API is queried.
      4. LLM selects the best match.
      5. BibTeX entry is generated.

    Exits early when the LLM indicates no more citations are needed.
    """

    def __init__(
        self,
        semantic_scholar_client: Any | None = None,
        agent_llm: Any | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        self.ss_client = semantic_scholar_client
        self.agent_llm = agent_llm
        self.log_dir = Path(log_dir) if log_dir else None
        self._log_entries: list[dict] = []

    def _log_round(self, entry: dict) -> None:
        """Append a round entry and optionally write to JSONL."""
        self._log_entries.append(entry)
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / "citation_search_log.jsonl"
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    async def search_loop(
        self,
        context: str,
        existing_bibtex: str = "",
        max_rounds: int = 20,
    ) -> list[dict]:
        """Run the iterative citation search loop.

        Parameters
        ----------
        context:
            The paper draft text or summary for which citations are needed.
        existing_bibtex:
            Already-collected BibTeX entries as a string.
        max_rounds:
            Maximum number of search rounds.

        Returns
        -------
        List of dicts, each with keys: ``citation_key``, ``title``,
        ``authors``, ``year``, ``bibtex``, ``paper_id``.
        """
        if self.agent_llm is None:
            logger.warning("No agent_llm provided; citation search skipped.")
            return []

        found_citations: list[dict] = []
        collected_keys: set[str] = set()

        for round_idx in range(max_rounds):
            # -- Step 1: Ask LLM to identify a missing citation --------
            identify_prompt = (
                "You are a research paper citation expert.\n\n"
                "Paper context:\n"
                f"{context[:3000]}\n\n"
                "Existing citations:\n"
                f"{existing_bibtex[:2000]}\n\n"
                "Previously found in this session:\n"
                f"{json.dumps([c['title'] for c in found_citations], default=str)}\n\n"
                "Identify ONE specific citation that is missing and should be added. "
                "If no more citations are needed, respond EXACTLY with: "
                '"No more citations needed"\n\n'
                "Otherwise respond with:\n"
                "CLAIM: <the claim that needs a citation>\n"
                "QUERY: <a search query to find the paper>"
            )

            identify_response = await self.agent_llm.generate(prompt=identify_prompt, purpose="citation_identify")

            # -- Early exit check ------------------------------------
            if "no more citations needed" in identify_response.lower():
                self._log_round(
                    {
                        "round": round_idx,
                        "action": "early_exit",
                        "reason": "LLM indicated no more citations needed",
                    }
                )
                break

            # -- Parse claim and query --------------------------------
            claim = ""
            query = ""
            for line in identify_response.split("\n"):
                line = line.strip()
                if line.upper().startswith("CLAIM:"):
                    claim = line[len("CLAIM:") :].strip()
                elif line.upper().startswith("QUERY:"):
                    query = line[len("QUERY:") :].strip()

            if not query:
                # Fallback: use the entire response as query
                query = identify_response.strip()[:100]

            # -- Step 2: Search Semantic Scholar -----------------------
            search_results: list[Any] = []
            if self.ss_client is not None:
                try:
                    search_results = await self.ss_client.search(query=query, limit=10)
                except Exception as exc:
                    logger.warning(
                        "Semantic Scholar search failed (round %d): %s",
                        round_idx,
                        exc,
                    )

            if not search_results:
                self._log_round(
                    {
                        "round": round_idx,
                        "action": "no_results",
                        "query": query,
                        "claim": claim,
                    }
                )
                continue

            # -- Step 3: Ask LLM to select the best match -------------
            results_text = "\n".join(
                f"{i}. {getattr(r, 'title', str(r))} "
                f"({getattr(r, 'year', '?')}) "
                f"by {', '.join(getattr(r, 'authors', [])[:3])}"
                for i, r in enumerate(search_results[:10])
            )

            select_prompt = (
                f"The claim needing citation: {claim}\n"
                f"Search query: {query}\n\n"
                f"Search results:\n{results_text}\n\n"
                "Select the BEST matching result by number (0-indexed). "
                "Respond with just the number, or -1 if none are relevant."
            )

            select_response = await self.agent_llm.generate(prompt=select_prompt, purpose="citation_select")

            # Parse selection
            try:
                # Extract first integer from response
                import re

                match = re.search(r"-?\d+", select_response)
                selected_idx = int(match.group()) if match else -1
            except (ValueError, AttributeError):
                selected_idx = -1

            if selected_idx < 0 or selected_idx >= len(search_results):
                self._log_round(
                    {
                        "round": round_idx,
                        "action": "no_selection",
                        "query": query,
                        "claim": claim,
                    }
                )
                continue

            selected = search_results[selected_idx]
            title = getattr(selected, "title", "")
            authors = getattr(selected, "authors", [])
            year = getattr(selected, "year", None)
            paper_id = getattr(selected, "paper_id", "")

            # -- Step 4: Generate BibTeX ------------------------------
            # Create a citation key
            first_author = authors[0].split()[-1] if authors else "Unknown"
            citation_key = f"{first_author.lower()}{year or 'nd'}"
            # Deduplicate keys
            base_key = citation_key
            counter = 1
            while citation_key in collected_keys:
                citation_key = f"{base_key}{chr(ord('a') + counter - 1)}"
                counter += 1
            collected_keys.add(citation_key)

            bibtex_prompt = (
                f"Generate a BibTeX entry for this paper:\n"
                f"Title: {title}\n"
                f"Authors: {', '.join(authors)}\n"
                f"Year: {year}\n"
                f"Venue: {getattr(selected, 'venue', '')}\n"
                f"DOI: {getattr(selected, 'doi', '')}\n\n"
                f"Use citation key: {citation_key}\n"
                "Return ONLY the BibTeX entry, nothing else."
            )

            bibtex = await self.agent_llm.generate(prompt=bibtex_prompt, purpose="citation_bibtex")

            # Clean up BibTeX
            if "```" in bibtex:
                bibtex = bibtex.split("```")[1] if "```" in bibtex else bibtex
                if bibtex.startswith("bibtex"):
                    bibtex = bibtex[len("bibtex") :]
                bibtex = bibtex.strip()

            citation_entry = {
                "citation_key": citation_key,
                "title": title,
                "authors": authors,
                "year": year,
                "bibtex": bibtex,
                "paper_id": paper_id,
            }
            found_citations.append(citation_entry)

            # Update existing bibtex for next round
            existing_bibtex += f"\n{bibtex}"

            self._log_round(
                {
                    "round": round_idx,
                    "action": "citation_found",
                    "query": query,
                    "claim": claim,
                    "citation_key": citation_key,
                    "title": title,
                    "year": year,
                }
            )

        return found_citations
