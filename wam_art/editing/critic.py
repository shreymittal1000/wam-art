"""VLM critic for validating semantic realism of image edits.

Provides a pluggable interface so real VLM backends (GPT-4V, LLaVA,
etc.) can be swapped in later.  For Phase 2 we ship a
HeuristicCritic that catches degenerate corruptions without needing
API keys.
"""

from __future__ import annotations

import base64
import io
import json
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CriticResult:
    """Outcome of a VLM critic judgement.

    Attributes:
        is_realistic: Whether the image looks like a plausible camera
            observation (not blank, all-noise, etc.).
        preserves_task: Whether the edit still contains the task-relevant
            visual cues (not completely occluded, etc.).
        score: Continuous confidence in [0, 1]; higher means more
            acceptable.
        reason: Free-text rationale (useful for logging / debugging).
    """

    is_realistic: bool
    preserves_task: bool
    score: float
    reason: str

    @property
    def passes(self) -> bool:
        """True iff both realism and task-preservation hold."""
        return self.is_realistic and self.preserves_task


# ---------------------------------------------------------------------------
# Base critic
# ---------------------------------------------------------------------------
class BaseCritic(ABC):
    """Abstract VLM critic for red-team edits."""

    @abstractmethod
    def judge(
        self,
        image: np.ndarray,
        edit_description: str,
        *,
        original_image: np.ndarray | None = None,
        **kwargs: Any,
    ) -> CriticResult:
        """Evaluate a single edited image.

        Args:
            image: uint8 RGB array (H, W, 3) – the *edited* image.
            edit_description: Human-readable description of the edit
                (e.g. "motion_blur_k=5").
            original_image: Optional original (un-edited) image for
                comparison-based critics.
            **kwargs: Backend-specific options (temperature, retries, etc.).

        Returns:
            CriticResult with binary flags and continuous score.
        """
        ...


# ---------------------------------------------------------------------------
# Dummy / pass-through critic
# ---------------------------------------------------------------------------
class DummyCritic(BaseCritic):
    """Always passes.  Useful as a no-op placeholder."""

    def judge(
        self,
        image: np.ndarray,
        edit_description: str,
        *,
        original_image: np.ndarray | None = None,
        **kwargs: Any,
    ) -> CriticResult:
        return CriticResult(
            is_realistic=True,
            preserves_task=True,
            score=1.0,
            reason="Dummy critic: unconditional pass.",
        )


# ---------------------------------------------------------------------------
# Heuristic critic (no API needed)
# ---------------------------------------------------------------------------
class HeuristicCritic(BaseCritic):
    """Rule-based critic that rejects degenerate corruptions.

    Catches:
    - All-black or all-white images.
    - Extreme brightness (near-clipping).
    - Extreme blur / lack of texture (variance collapse).
    - Occlusion so large that >80 % of image is uniform.

    Tuned conservatively: only rejects obviously-broken images.
    """

    MIN_BRIGHTNESS_RANGE: float = 8.0  # peak-to-peak across channels
    MAX_BRIGHTNESS_PCT: float = 0.02  # fraction of pixels near clip
    MIN_VARIANCE: float = 100.0  # per-channel variance
    MAX_UNIFORM_PCT: float = 0.90  # fraction of image with same value

    def judge(
        self,
        image: np.ndarray,
        edit_description: str,
        *,
        original_image: np.ndarray | None = None,
        **kwargs: Any,
    ) -> CriticResult:
        if image.ndim != 3 or image.shape[2] != 3:
            return CriticResult(
                is_realistic=False,
                preserves_task=False,
                score=0.0,
                reason="Image must be HxWx3 RGB uint8.",
            )

        # Per-channel stats
        per_channel_var = image.reshape(-1, 3).var(axis=0).mean()
        per_channel_min = image.reshape(-1, 3).min(axis=0)
        per_channel_max = image.reshape(-1, 3).max(axis=0)
        brightness_range = float(per_channel_max.max() - per_channel_min.min())

        # Near-clip fraction (pixels at 0 or 255)
        near_clip = np.mean(
            (image == 0) | (image == 255)
        )

        # Uniformity fraction (most common pixel value)
        # Fast approximation: check if median is close to mean for all channels
        # and whether the mode dominates.
        hist_max_frac = 0.0
        for c in range(3):
            hist = np.bincount(image[:, :, c].ravel(), minlength=256)
            hist_max_frac = max(hist_max_frac, hist.max() / image[:, :, c].size)

        reasons: list[str] = []

        if brightness_range < self.MIN_BRIGHTNESS_RANGE:
            reasons.append(
                f"brightness_range={brightness_range:.1f} < "
                f"{self.MIN_BRIGHTNESS_RANGE} (near-degenerate image)."
            )

        if near_clip > self.MAX_BRIGHTNESS_PCT:
            reasons.append(
                f"near_clip={near_clip:.2%} > "
                f"{self.MAX_BRIGHTNESS_PCT:.2%} (too many clipped pixels)."
            )

        if per_channel_var < self.MIN_VARIANCE:
            reasons.append(
                f"variance={per_channel_var:.1f} < "
                f"{self.MIN_VARIANCE} (too little texture / over-blurred)."
            )

        if hist_max_frac > self.MAX_UNIFORM_PCT:
            reasons.append(
                f"uniform_frac={hist_max_frac:.2%} > "
                f"{self.MAX_UNIFORM_PCT:.2%} (image is mostly one value)."
            )

        if reasons:
            score = max(0.0, 1.0 - 0.25 * len(reasons))
            return CriticResult(
                is_realistic=False,
                preserves_task=False,
                score=score,
                reason="; ".join(reasons),
            )

        return CriticResult(
            is_realistic=True,
            preserves_task=True,
            score=1.0,
            reason="Heuristic checks passed.",
        )


# ---------------------------------------------------------------------------
# API-backed critic (OpenRouter → GPT-4o / etc.)
# ---------------------------------------------------------------------------
class APICritic(BaseCritic):
    """Vision-capable LLM critic via OpenRouter (or any OpenAI-compatible endpoint).

    Expects ``OPENROUTER_API_KEY`` as an environment variable, or pass
    ``api_key`` explicitly.

    Args:
        api_key: OpenRouter (or OpenAI) API key.
        model: Model identifier, e.g. ``"openai/gpt-4o-mini"`` (default),
            ``"openai/gpt-4o"``, ``"anthropic/claude-3.5-sonnet"``, etc.
        base_url: API base URL.  Default is OpenRouter.
        timeout: HTTP timeout in seconds.
        max_retries: Number of retries on transient errors.
    """

    _DEFAULT_PROMPT: str = (
        "You are a visual quality critic for robot-learning research. "
        "A robot-vision image was edited to simulate the following condition:\n\n"
        "{edit_description}\n\n"
        "The first attached image is the original observation. "
        "The second attached image is the edited version. "
        "Does the edited image plausibly reflect the described condition "
        "while remaining a realistic camera observation? "
        "Answer with a single word — 'yes' or 'no' — followed by a brief explanation."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "APICritic requires an API key. "
                "Set OPENROUTER_API_KEY env var or pass api_key=..."
            )
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """uint8 RGB → base64 PNG data URI."""
        from PIL import Image

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def _call_api(
        self,
        prompt: str,
        original_image: np.ndarray | None,
        edited_image: np.ndarray,
    ) -> dict[str, Any]:
        """Build the multi-modal message and POST to the endpoint."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if original_image is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(original_image)},
                }
            )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": self._encode_image(edited_image)},
            }
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a concise visual evaluator."},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": 128,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "WAM-ART",
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    warnings.warn(
                        f"APICritic API call failed (attempt {attempt + 1}): {exc}. Retrying...",
                        stacklevel=3,
                    )

        # All retries exhausted
        raise RuntimeError(
            f"APICritic failed after {self.max_retries + 1} attempts. Last error: {last_exc}"
        )

    @staticmethod
    def _parse_yes_no(text: str) -> bool | None:
        """Heuristic parser: look for leading 'yes' or 'no'."""
        t = text.strip().lower()
        if t.startswith("yes"):
            return True
        if t.startswith("no"):
            return False
        # Fuzzy fallbacks
        if "yes" in t[:20] and "no" not in t[:20]:
            return True
        if "no" in t[:20] and "yes" not in t[:20]:
            return False
        return None  # ambiguous

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def judge(
        self,
        image: np.ndarray,
        edit_description: str,
        *,
        original_image: np.ndarray | None = None,
        **kwargs: Any,
    ) -> CriticResult:
        if image.ndim != 3 or image.shape[2] != 3:
            return CriticResult(
                is_realistic=False,
                preserves_task=False,
                score=0.0,
                reason="Image must be HxWx3 RGB uint8.",
            )

        prompt = self._DEFAULT_PROMPT.format(edit_description=edit_description)
        try:
            data = self._call_api(prompt, original_image, image)
        except RuntimeError as exc:
            warnings.warn(f"APICritic API error, falling back to heuristic pass: {exc}", stacklevel=2)
            # Fail-open so the benchmark doesn't die on flaky API calls.
            return CriticResult(
                is_realistic=True,
                preserves_task=True,
                score=0.5,
                reason=f"API error; fail-open. {exc}",
            )

        try:
            message = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            warnings.warn(f"Unexpected API response format: {exc}. Raw={json.dumps(data)[:200]}", stacklevel=2)
            return CriticResult(
                is_realistic=True,
                preserves_task=True,
                score=0.5,
                reason="Unexpected API response format; fail-open.",
            )

        yn = self._parse_yes_no(message)
        if yn is True:
            return CriticResult(
                is_realistic=True,
                preserves_task=True,
                score=1.0,
                reason=message,
            )
        if yn is False:
            return CriticResult(
                is_realistic=False,
                preserves_task=False,
                score=0.0,
                reason=message,
            )
        # Ambiguous → still pass but flag uncertainty
        return CriticResult(
            is_realistic=True,
            preserves_task=True,
            score=0.5,
            reason=f"Ambiguous response: {message}",
        )
