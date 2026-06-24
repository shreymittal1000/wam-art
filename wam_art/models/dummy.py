"""Dummy WAM adapter for smoke testing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from wam_art.models.base import BaseWAMAdapter


class DummyWAMAdapter(BaseWAMAdapter):
    """Random WAM for architecture validation.

    Produces deterministic latents given a fixed seed,
    useful for testing the pipeline without real model weights.
    """

    def __init__(
        self,
        model_name: str = "dummy",
        device: str = "cpu",
        latent_dim: int = 128,
        action_dim: int = 7,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed=42)
        # Fixed random projection for deterministic but input-dependent latents
        # For 64x64x3 = 12288 input dims (smoke-test default)
        self._projection = torch.randn(12288, latent_dim, device=device)

    def load(self, checkpoint_path: str | None = None) -> None:
        """No-op for dummy model."""
        pass

    def _obs_to_tensor(self, observation: np.ndarray | Tensor) -> Tensor:
        if isinstance(observation, np.ndarray):
            return torch.from_numpy(observation).float().to(self.device)
        return observation.float().to(self.device)

    def extract_latent(self, observation: np.ndarray | Tensor) -> Tensor:
        """Compute a hash-like latent from the image via fixed random projection."""
        x = self._obs_to_tensor(observation)
        if x.ndim == 3:  # (H, W, C)
            x = x.unsqueeze(0)
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        # Pad or truncate to projection matrix input size
        target_in = self._projection.shape[0]
        if x_flat.shape[1] < target_in:
            x_flat = torch.nn.functional.pad(x_flat, (0, target_in - x_flat.shape[1]))
        else:
            x_flat = x_flat[:, :target_in]
        latent = x_flat @ self._projection  # (B, latent_dim)
        # Normalize for cosine distance
        latent = latent / (latent.norm(dim=-1, keepdim=True) + 1e-8)
        return latent.squeeze(0) if b == 1 else latent

    def predict_action(
        self, observation: np.ndarray | Tensor, state: Any | None = None
    ) -> tuple[Tensor, Any]:
        """Return random action."""
        action = torch.randn(self.action_dim, device=self.device)
        return action, None

    def reset(self) -> None:
        """Reset dummy state."""
        pass
