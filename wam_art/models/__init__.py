"""Base model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import Tensor


class BaseWAMAdapter(ABC):
    """Abstract adapter for World Action Models.

    Each concrete adapter wraps a specific WAM implementation
    (FastWAM, DreamZero, etc.) and exposes a unified interface
    for latent extraction and policy rollout.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def load(self, checkpoint_path: str | None = None) -> None:
        """Load model weights and move to device."""
        ...

    @abstractmethod
    def extract_latent(self, observation: np.ndarray | Tensor) -> Tensor:
        """Extract latent representation ϕ_W(o) from an observation.

        Args:
            observation: RGB image array (H, W, C) or batch.

        Returns:
            Latent vector of shape (d,) or (B, d).
        """
        ...

    @abstractmethod
    def predict_action(
        self, observation: np.ndarray | Tensor, state: Any | None = None
    ) -> tuple[Tensor, Any]:
        """Run one-step action prediction.

        Args:
            observation: Current observation.
            state: Optional internal state for recurrent models.

        Returns:
            (action, next_state)
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (for episodic evaluation)."""
        ...

    def to(self, device: str) -> "BaseWAMAdapter":
        """Move model to device and return self."""
        self.device = device
        return self
