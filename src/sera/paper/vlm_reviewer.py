"""VLMReviewer per S11.4 - Vision-Language Model figure review and analysis."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VLMReviewer:
    """Uses a Vision-Language Model to analyse figures, captions, and detect
    duplicate or low-quality visualisations.

    Supports OpenAI and Anthropic VLM providers.  When ``model`` is None the
    reviewer degrades gracefully -- every public method returns a sensible
    empty / no-op result.
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        """Initialise VLM client.

        Parameters
        ----------
        model:
            Model identifier, e.g. ``"gpt-4o"`` or ``"claude-sonnet-4-20250514"``.
            Pass ``None`` to disable VLM reviewing.
        provider:
            ``"openai"`` or ``"anthropic"``.
        """
        self.model = model
        self.provider = provider
        self._client: Any = None
        self.enabled = model is not None and provider is not None

        if not self.enabled:
            return

        if provider == "openai":
            try:
                import openai
                import os

                self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            except ImportError:
                logger.warning("openai package not installed; VLM disabled.")
                self.enabled = False
        elif provider == "anthropic":
            try:
                import anthropic
                import os

                self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
            except ImportError:
                logger.warning("anthropic package not installed; VLM disabled.")
                self.enabled = False
        else:
            logger.warning("Unknown VLM provider '%s'; VLM disabled.", provider)
            self.enabled = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(path: Path) -> str:
        """Read an image file and return its base64-encoded string."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_vlm(self, text_prompt: str, image_paths: list[Path]) -> str:
        """Send a prompt with images to the VLM and return the text response."""
        if not self.enabled or self._client is None:
            return ""

        if self.provider == "openai":
            return self._call_openai(text_prompt, image_paths)
        elif self.provider == "anthropic":
            return self._call_anthropic(text_prompt, image_paths)
        return ""

    def _call_openai(self, text_prompt: str, image_paths: list[Path]) -> str:
        content: list[dict] = [{"type": "text", "text": text_prompt}]
        for img_path in image_paths:
            b64 = self._encode_image(img_path)
            suffix = img_path.suffix.lstrip(".").lower()
            mime = f"image/{suffix}" if suffix in ("png", "jpeg", "jpg", "gif", "webp") else "image/png"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("OpenAI VLM call failed: %s", exc)
            return ""

    def _call_anthropic(self, text_prompt: str, image_paths: list[Path]) -> str:
        content: list[dict] = []
        for img_path in image_paths:
            b64 = self._encode_image(img_path)
            suffix = img_path.suffix.lstrip(".").lower()
            media_type = f"image/{suffix}" if suffix in ("png", "jpeg", "jpg", "gif", "webp") else "image/png"
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                }
            )
        content.append({"type": "text", "text": text_prompt})
        try:
            resp = self._client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
            )
            return resp.content[0].text
        except Exception as exc:
            logger.warning("Anthropic VLM call failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe_figures(self, figure_paths: list[Path]) -> dict[str, str]:
        """Analyse each figure and return a description.

        Parameters
        ----------
        figure_paths:
            List of paths to figure PNG files.

        Returns
        -------
        dict mapping figure filename to its VLM-generated description.
        """
        if not self.enabled:
            return {p.name: "" for p in figure_paths}

        descriptions: dict[str, str] = {}
        for fig_path in figure_paths:
            if not fig_path.exists():
                descriptions[fig_path.name] = ""
                continue
            prompt = (
                "Describe this scientific figure in detail. Cover:\n"
                "1. What type of plot/chart it is\n"
                "2. What the axes represent\n"
                "3. Key trends and patterns\n"
                "4. Any notable features or anomalies\n"
                "Be concise but thorough."
            )
            desc = self._call_vlm(prompt, [fig_path])
            descriptions[fig_path.name] = desc
        return descriptions

    def review_figure_caption_refs(
        self,
        figure_path: Path,
        caption: str,
        text_refs: list[str],
    ) -> dict[str, str]:
        """Review consistency between a figure, its caption, and text references.

        Parameters
        ----------
        figure_path:
            Path to the figure image.
        caption:
            The figure caption text.
        text_refs:
            List of text snippets that reference this figure.

        Returns
        -------
        dict with keys: img_review, caption_review, figrefs_review,
        informative, suggestion.
        """
        if not self.enabled or not figure_path.exists():
            return {
                "img_review": "",
                "caption_review": "",
                "figrefs_review": "",
                "informative": "unknown",
                "suggestion": "",
            }

        refs_text = "\n".join(f"- {r}" for r in text_refs) if text_refs else "(none)"
        prompt = (
            "Review this scientific figure for quality and consistency.\n\n"
            f"Caption: {caption}\n\n"
            f"Text references:\n{refs_text}\n\n"
            "Provide your review as follows:\n"
            "IMG_REVIEW: <your review of the image quality and content>\n"
            "CAPTION_REVIEW: <does the caption accurately describe the figure?>\n"
            "FIGREFS_REVIEW: <are the text references consistent with the figure?>\n"
            "INFORMATIVE: <yes/no - is this figure informative and necessary?>\n"
            "SUGGESTION: <any improvement suggestions>"
        )
        response = self._call_vlm(prompt, [figure_path])

        result = {
            "img_review": "",
            "caption_review": "",
            "figrefs_review": "",
            "informative": "unknown",
            "suggestion": "",
        }

        # Parse the structured response
        for line in response.split("\n"):
            line = line.strip()
            for key, field_name in [
                ("IMG_REVIEW:", "img_review"),
                ("CAPTION_REVIEW:", "caption_review"),
                ("FIGREFS_REVIEW:", "figrefs_review"),
                ("INFORMATIVE:", "informative"),
                ("SUGGESTION:", "suggestion"),
            ]:
                if line.upper().startswith(key):
                    result[field_name] = line[len(key) :].strip()

        return result

    def detect_duplicate_figures(self, figure_paths: list[Path]) -> list[dict[str, Any]]:
        """Detect potentially duplicate or highly similar figures.

        Parameters
        ----------
        figure_paths:
            List of paths to figure PNG files.

        Returns
        -------
        list of dicts with keys: fig_a, fig_b, similarity, recommendation.
        """
        if not self.enabled or len(figure_paths) < 2:
            return []

        duplicates: list[dict[str, Any]] = []

        # Compare pairs (limited to avoid excessive API calls)
        pairs_checked = 0
        max_pairs = 15  # reasonable limit

        for i in range(len(figure_paths)):
            for j in range(i + 1, len(figure_paths)):
                if pairs_checked >= max_pairs:
                    break
                if not figure_paths[i].exists() or not figure_paths[j].exists():
                    continue

                prompt = (
                    "Compare these two scientific figures. Are they duplicates or "
                    "very similar? Rate similarity from 0.0 (completely different) "
                    "to 1.0 (identical/duplicate).\n\n"
                    "Respond in this format:\n"
                    "SIMILARITY: <float>\n"
                    "RECOMMENDATION: <keep_both|merge|remove_one>"
                )
                response = self._call_vlm(prompt, [figure_paths[i], figure_paths[j]])
                pairs_checked += 1

                similarity = 0.0
                recommendation = "keep_both"
                for line in response.split("\n"):
                    line = line.strip()
                    if line.upper().startswith("SIMILARITY:"):
                        try:
                            similarity = float(line.split(":")[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif line.upper().startswith("RECOMMENDATION:"):
                        recommendation = line.split(":")[1].strip().lower()

                if similarity > 0.5:
                    duplicates.append(
                        {
                            "fig_a": figure_paths[i].name,
                            "fig_b": figure_paths[j].name,
                            "similarity": similarity,
                            "recommendation": recommendation,
                        }
                    )
            if pairs_checked >= max_pairs:
                break

        return duplicates
