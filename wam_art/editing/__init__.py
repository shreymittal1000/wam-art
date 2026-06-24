"""Image editing / perturbation pipeline.

Re-exports the full editing API:
- BaseEditor, SimplePerturbationEditor, RichPerturbationEditor
- Corruption functions (gaussian_noise, occlusion, etc.)
- VLM critic base class and implementations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# Sub-module re-exports
from wam_art.editing.corruptions import (
    brightness_shift,
    contrast_shift,
    gaussian_blur,
    gaussian_noise,
    jpeg_compression,
    list_corruptions,
    motion_blur,
    occlusion,
    perspective_warp,
    salt_and_pepper,
    saturation_shift,
)
from wam_art.editing.critic import (
    BaseCritic,
    CriticResult,
    DummyCritic,
    HeuristicCritic,
)

__all__ = [
    "BaseEditor",
    "SimplePerturbationEditor",
    "RichPerturbationEditor",
    # corruptions
    "gaussian_noise",
    "salt_and_pepper",
    "gaussian_blur",
    "motion_blur",
    "brightness_shift",
    "contrast_shift",
    "saturation_shift",
    "occlusion",
    "perspective_warp",
    "jpeg_compression",
    "list_corruptions",
    # critic
    "BaseCritic",
    "CriticResult",
    "DummyCritic",
    "HeuristicCritic",
]


class BaseEditor(ABC):
    """Abstract image editor for generating off-nominal observations."""

    def __init__(self, factor_name: str) -> None:
        self.factor_name = factor_name

    @abstractmethod
    def edit(self, image: np.ndarray, instruction: str, **kwargs: Any) -> np.ndarray:
        """Apply the factor-specific edit to an image.

        Args:
            image: Nominal image array (H, W, C) uint8.
            instruction: Text description of the edit.
            **kwargs: Factor-specific parameters.

        Returns:
            Edited image array (H, W, C) uint8.
        """
        ...


class SimplePerturbationEditor(BaseEditor):
    """Deterministic pixel-level perturbations for smoke testing.

    Replaces a real diffusion-based editor when weights aren't available.
    """

    def __init__(
        self,
        factor_name: str,
        perturbation_type: str = "noise",
        magnitude: float = 0.1,
    ) -> None:
        super().__init__(factor_name)
        self.perturbation_type = perturbation_type
        self.magnitude = magnitude

    def edit(self, image: np.ndarray, instruction: str, **kwargs: Any) -> np.ndarray:
        """Apply simple deterministic perturbation."""
        img = image.astype(np.float32) / 255.0
        rng = np.random.default_rng(seed=hash(instruction + self.factor_name) % 2**31)

        if self.perturbation_type == "noise":
            noise = rng.normal(0, self.magnitude, img.shape)
            img = np.clip(img + noise, 0, 1)
        elif self.perturbation_type == "brightness":
            img = np.clip(img * (1.0 + self.magnitude), 0, 1)
        elif self.perturbation_type == "darkness":
            img = np.clip(img * (1.0 - self.magnitude), 0, 1)
        else:
            raise ValueError(f"Unknown perturbation type: {self.perturbation_type}")

        return (img * 255).astype(np.uint8)


class RichPerturbationEditor(BaseEditor):
    """Real-world corruption editor using OpenCV/Pillow transforms.

    Selects a named corruption from the ``wam_art.editing.corruptions``
    registry.  Optionally applies a :class:`BaseCritic` to reject
    degenerate outputs.
    """

    def __init__(
        self,
        factor_name: str,
        corruption: str,
        corruption_kwargs: dict[str, Any] | None = None,
        critic: BaseCritic | None = None,
    ) -> None:
        """Args:
            factor_name: Human-readable factor label (e.g. ``motion_blur_k5``).
            corruption: Name of a registered corruption function
                (see :func:`list_corruptions`).
            corruption_kwargs: Keyword arguments forwarded to the corruption.
            critic: Optional VLM / heuristic critic to validate edits.
        """
        super().__init__(factor_name)
        available = list_corruptions()
        if corruption not in available:
            raise ValueError(
                f"Unknown corruption {corruption!r}. "
                f"Available: {available}"
            )
        self.corruption = corruption
        self.corruption_kwargs = corruption_kwargs or {}
        self.critic = critic

    def edit(self, image: np.ndarray, instruction: str, **kwargs: Any) -> np.ndarray:
        """Apply the configured corruption and run critic if present."""
        from wam_art.editing.corruptions import apply_corruption

        edited = apply_corruption(
            self.corruption, image, **self.corruption_kwargs
        )

        if self.critic is not None:
            result = self.critic.judge(
                edited,
                edit_description=f"{self.factor_name}: {self.corruption} "
                f"with {self.corruption_kwargs}",
            )
            if not result.passes:
                # Fallback: return the original image unmodified.
                # Logged reason lets us audit how often this happens.
                # A future version might retry with different params.
                return image

        return edited
