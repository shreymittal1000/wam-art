"""Online WAM-ART scoring for real closed-loop policy evaluations.

This module deliberately does not own a simulator or policy loop.  Instead, the
real evaluator calls :meth:`OnlineWAMARTScorer.observe` with the exact visual
tensor passed to the WAM at each replan, and calls :meth:`end_episode` with the
simulator's measured success label.  This keeps prediction and measurement in
the same episode and prevents the disconnected evaluation that the original
benchmark harness allowed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from wam_art.anomaly import calibrate_threshold, compute_anomaly_rates
from wam_art.latents import knn_cosine_distance

LatentExtractor = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class EpisodePrediction:
    """Prediction and measured label for one closed-loop episode."""

    episode_idx: int
    measured_success: bool
    n_observations: int
    predicted_success_rate: float | None
    anomaly_rate: float | None
    mean_anomaly_score: float | None
    max_anomaly_score: float | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class OnlineRunReport:
    """Serializable report whose predictions and labels share episodes."""

    schema_version: int
    mode: str
    model_name: str
    task_suite: str
    task_id: int
    task_description: str
    corruption: str | None
    corruption_kwargs: dict[str, object]
    reference_path: str | None
    threshold: float | None
    k: int
    target_anomaly_rate: float
    successes: int
    total_episodes: int
    measured_success_rate: float
    episodes: list[EpisodePrediction]

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2) + "\n")


class OnlineWAMARTScorer:
    """Collect or score policy-input latents during real episodes.

    ``mode='collect'`` creates a nominal reference from clean successful
    episodes.  ``mode='score'`` loads that reference and computes an anomaly
    prediction for every episode.  The caller must pass the exact image tensor
    used for policy inference; scoring an independently generated image is an
    error in experimental design.
    """

    def __init__(
        self,
        latent_extractor: LatentExtractor,
        *,
        mode: str,
        reference_path: str | Path | None = None,
        k: int = 5,
        target_anomaly_rate: float = 0.05,
    ) -> None:
        if mode not in {"collect", "score"}:
            raise ValueError("mode must be 'collect' or 'score'")
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= target_anomaly_rate <= 1.0:
            raise ValueError("target_anomaly_rate must be in [0, 1]")
        if mode == "score" and reference_path is None:
            raise ValueError("score mode requires reference_path")

        self.latent_extractor = latent_extractor
        self.mode = mode
        self.reference_path = None if reference_path is None else Path(reference_path)
        self.k = k
        self.target_anomaly_rate = target_anomaly_rate
        self._current_latents: list[np.ndarray] = []
        self._episodes: list[tuple[bool, list[np.ndarray], dict[str, object]]] = []
        self._reference: np.ndarray | None = None
        self._threshold: float | None = None

        if self.mode == "score":
            self._load_reference(self.reference_path)

    def _load_reference(self, path: Path | None) -> None:
        assert path is not None
        if not path.exists():
            raise FileNotFoundError(f"WAM-ART reference not found: {path}")
        with np.load(path, allow_pickle=False) as payload:
            self._reference = np.asarray(payload["reference_latents"], dtype=np.float32)
            self._threshold = float(np.asarray(payload["threshold"]).item())
            reference_k = int(np.asarray(payload["k"]).item())
        if self._reference.ndim != 2 or len(self._reference) == 0:
            raise ValueError("reference_latents must be a non-empty [N, D] array")
        if reference_k != self.k:
            raise ValueError(
                f"Reference was calibrated with k={reference_k}, requested k={self.k}"
            )

    def observe(self, policy_image: Tensor) -> None:
        """Record the latent of the exact image tensor passed to the policy."""
        with torch.no_grad():
            latent = self.latent_extractor(policy_image)
        if not isinstance(latent, Tensor):
            latent = torch.as_tensor(latent)
        latent = latent.detach().to(device="cpu", dtype=torch.float32).flatten()
        norm = torch.linalg.vector_norm(latent)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("latent extractor returned a non-finite or zero latent")
        latent = latent / norm
        self._current_latents.append(latent.numpy())

    def end_episode(
        self,
        measured_success: bool,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Attach the simulator label to observations from the current episode."""
        if not self._current_latents:
            raise RuntimeError("end_episode called before any policy observations")
        self._episodes.append(
            (bool(measured_success), self._current_latents, dict(metadata or {}))
        )
        self._current_latents = []

    def save_reference(self, path: str | Path) -> Path:
        """Calibrate and save a nominal manifold from successful clean episodes."""
        if self.mode != "collect":
            raise RuntimeError("save_reference is only valid in collect mode")
        if self._current_latents:
            raise RuntimeError("cannot save reference with an unfinished episode")

        successful = [
            z for success, episode, _metadata in self._episodes if success for z in episode
        ]
        if len(successful) < 4:
            raise RuntimeError(
                "at least four observations from successful clean episodes are required"
            )
        latents = np.stack(successful).astype(np.float32, copy=False)
        split = max(1, min(len(latents) - 1, int(0.6 * len(latents))))
        train = latents[:split]
        calibration = latents[split:]
        scores = knn_cosine_distance(calibration, train, k=self.k)
        threshold = calibrate_threshold(scores, self.target_anomaly_rate)

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            reference_latents=train,
            calibration_latents=calibration,
            calibration_scores=scores.astype(np.float32),
            threshold=np.asarray(threshold, dtype=np.float64),
            k=np.asarray(self.k, dtype=np.int64),
            target_anomaly_rate=np.asarray(self.target_anomaly_rate, dtype=np.float64),
        )
        self.reference_path = output
        self._reference = train
        self._threshold = threshold
        return output

    @property
    def successful_observation_count(self) -> int:
        """Number of completed observations eligible for a clean reference."""
        return sum(
            len(episode)
            for success, episode, _metadata in self._episodes
            if success
        )

    def episode_predictions(self) -> list[EpisodePrediction]:
        """Return episode-level anomaly predictions paired with real labels."""
        if self._current_latents:
            raise RuntimeError("cannot report with an unfinished episode")

        predictions: list[EpisodePrediction] = []
        for idx, (success, episode, metadata) in enumerate(self._episodes):
            if self.mode == "collect":
                predictions.append(
                    EpisodePrediction(
                        idx, success, len(episode), None, None, None, None, metadata
                    )
                )
                continue

            assert self._reference is not None and self._threshold is not None
            scores = knn_cosine_distance(
                np.stack(episode), self._reference, k=self.k
            )
            _, anomaly_rate = compute_anomaly_rates(scores, self._threshold)
            predictions.append(
                EpisodePrediction(
                    episode_idx=idx,
                    measured_success=success,
                    n_observations=len(episode),
                    predicted_success_rate=1.0 - anomaly_rate,
                    anomaly_rate=anomaly_rate,
                    mean_anomaly_score=float(np.mean(scores)),
                    max_anomaly_score=float(np.max(scores)),
                    metadata=metadata,
                )
            )
        return predictions

    def build_report(
        self,
        *,
        model_name: str,
        task_suite: str,
        task_id: int,
        task_description: str,
        corruption: str | None,
        corruption_kwargs: dict[str, object] | None = None,
    ) -> OnlineRunReport:
        episodes = self.episode_predictions()
        if not episodes:
            raise RuntimeError("cannot build a report without completed episodes")
        successes = sum(int(ep.measured_success) for ep in episodes)
        return OnlineRunReport(
            schema_version=2,
            mode=self.mode,
            model_name=model_name,
            task_suite=task_suite,
            task_id=int(task_id),
            task_description=task_description,
            corruption=corruption,
            corruption_kwargs=corruption_kwargs or {},
            reference_path=(
                None if self.reference_path is None else str(self.reference_path)
            ),
            threshold=self._threshold,
            k=self.k,
            target_anomaly_rate=self.target_anomaly_rate,
            successes=successes,
            total_episodes=len(episodes),
            measured_success_rate=successes / len(episodes),
            episodes=episodes,
        )


def fastwam_vae_latent_extractor(model: object) -> LatentExtractor:
    """Build an extractor around the VAE already resident in FastWAM.

    The policy image must be the normalized ``[1,C,H,W]`` tensor that is
    passed to ``infer_action``.  Reusing it guarantees that WAM-ART scores the
    observation that actually caused the measured behavior.
    """

    encode = getattr(model, "_encode_input_image_latents_tensor", None)
    if encode is None:
        raise TypeError("FastWAM model does not expose its image-latent encoder")

    def extract(policy_image: Tensor) -> Tensor:
        if policy_image.ndim != 4 or policy_image.shape[0] != 1:
            raise ValueError("FastWAM policy image must have shape [1,C,H,W]")
        latent = cast(Tensor, encode(policy_image[0]))
        return latent.flatten()

    return extract
