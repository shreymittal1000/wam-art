"""VLM critic for validating semantic realism of image edits.

Provides a pluggable interface so real VLM backends (GPT-4V, LLaVA,
etc.) can be swapped in later.  For Phase 2 we ship a
HeuristicCritic that catches degenerate corruptions without needing
API keys.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


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
        self, image: np.ndarray, edit_description: str, **kwargs: Any
    ) -> CriticResult:
        """Evaluate a single edited image.

        Args:
            image: uint8 RGB array (H, W, 3).
            edit_description: Human-readable description of the edit
                (e.g. "motion_blur_k=5").
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
        self, image: np.ndarray, edit_description: str, **kwargs: Any
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
        self, image: np.ndarray, edit_description: str, **kwargs: Any
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
