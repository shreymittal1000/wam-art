"""Image editing / perturbation pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image


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


# TODO: Add diffusion-based editor (InstructPix2Pix / SDEdit) later.
