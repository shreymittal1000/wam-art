"""RoboTwin glue that keeps FastWAM inputs, anomaly scores, and outcomes connected."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wam_art.editing.corruptions import apply_corruption
from wam_art.eval.online import OnlineWAMARTScorer, fastwam_vae_latent_extractor

_SEEDED_CORRUPTIONS = {
    "gaussian_noise",
    "occlusion",
    "perspective_warp",
    "salt_and_pepper",
}


class RobotTwinWAMARTSession:
    """Own one reproducible, episode-connected FastWAM + RoboTwin run."""

    def __init__(
        self,
        model: object,
        *,
        mode: str,
        checkpoint_path: str,
        task_name: str,
        output_path: str | Path,
        reference_path: str | Path,
        corruption: str | None = None,
        corruption_kwargs: dict[str, object] | None = None,
        policy_seed: int | None = None,
        corruption_seed: int = 0,
        k: int = 5,
        target_anomaly_rate: float = 0.05,
    ) -> None:
        if mode == "collect" and corruption is not None:
            raise ValueError("RoboTwin reference collection must use clean observations")

        self.model = model
        self.mode = mode
        self.checkpoint_path = str(checkpoint_path)
        self.task_name = str(task_name)
        self.output_path = Path(output_path)
        self.reference_path = Path(reference_path)
        self.corruption = corruption
        self.corruption_kwargs = dict(corruption_kwargs or {})
        self.policy_seed = policy_seed
        self.corruption_seed = int(corruption_seed)
        self._episode_idx = 0
        self._observation_idx = 0
        self.scorer = OnlineWAMARTScorer(
            fastwam_vae_latent_extractor(model),
            mode=mode,
            reference_path=reference_path if mode == "score" else None,
            k=k,
            target_anomaly_rate=target_anomaly_rate,
        )

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Corrupt one composite RGB image before both scoring and inference."""
        if self.corruption is None:
            return image

        kwargs = dict(self.corruption_kwargs)
        if self.corruption in _SEEDED_CORRUPTIONS and "seed" not in kwargs:
            kwargs["seed"] = (
                self.corruption_seed
                + self._episode_idx * 1_000_000
                + self._observation_idx
            )
        return apply_corruption(self.corruption, image, **kwargs)

    def observe(self, policy_image: torch.Tensor) -> None:
        self.scorer.observe(policy_image)
        self._observation_idx += 1

    def end_episode(
        self,
        measured_success: bool,
        *,
        environment_seed: int,
        policy_seed: int | None = None,
    ) -> None:
        self.scorer.end_episode(
            measured_success,
            metadata={
                "environment_seed": int(environment_seed),
                "policy_seed": self.policy_seed if policy_seed is None else policy_seed,
                "corruption_seed": self.corruption_seed,
                "corruption_seed_derivation": (
                    "base + episode_index * 1000000 + observation_index"
                ),
            },
        )
        self._episode_idx += 1
        self._observation_idx = 0
        self.save()

    def save(self) -> Path:
        """Checkpoint the reference/report after every completed episode."""
        if self.mode == "collect":
            if self.scorer.successful_observation_count >= 4:
                return self.scorer.save_reference(self.reference_path)
            return self.reference_path

        report = self.scorer.build_report(
            model_name=Path(self.checkpoint_path).stem,
            task_suite="robotwin",
            task_id=0,
            task_description=self.task_name,
            corruption=self.corruption,
            corruption_kwargs=self.corruption_kwargs,
        )
        report.save(self.output_path)
        return self.output_path
