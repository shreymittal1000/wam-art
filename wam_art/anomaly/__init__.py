"""Conformal prediction for anomaly thresholds."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def calibrate_threshold(
    scores: NDArray[np.float64],
    target_anomaly_rate: float,
) -> float:
    """Calibrate anomaly threshold via conformal prediction.

    Given held-out nominal scores, choose a threshold τ such that
    the empirical anomaly rate matches the nominal failure rate.

    Args:
        scores: (N,) anomaly scores on held-out nominal data.
        target_anomaly_rate: Desired fraction of nominal data flagged
            as anomalous (e.g., 1 - R_nom where R_nom is nominal success rate).

    Returns:
        Threshold τ.

    Raises:
        ValueError: If target_anomaly_rate is not in [0, 1].
    """
    if not (0.0 <= target_anomaly_rate <= 1.0):
        raise ValueError(f"target_anomaly_rate must be in [0, 1], got {target_anomaly_rate}")

    if len(scores) == 0:
        raise ValueError("Cannot calibrate threshold on empty scores.")

    sorted_scores = np.sort(scores)
    n = len(sorted_scores)

    if target_anomaly_rate >= 1.0:
        return float(sorted_scores[0] - 1e-9)
    if target_anomaly_rate <= 0.0:
        return float(sorted_scores[-1])

    idx = int(np.floor(n * (1.0 - target_anomaly_rate) - 1))
    idx = np.clip(idx, 0, n - 1)
    return float(sorted_scores[idx])


def compute_anomaly_rates(
    scores: NDArray[np.float64],
    threshold: float,
) -> tuple[NDArray[np.float64], float]:
    """Compute per-sample anomaly flags and overall rate.

    Args:
        scores: (N,) anomaly scores.
        threshold: Calibrated threshold τ.

    Returns:
        Binary flags (N,) and scalar rate.
    """
    flags = (scores > threshold).astype(np.float64)
    rate = float(flags.mean())
    return flags, rate


def split_scores(
    scores: NDArray[np.float64],
    calibration_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Randomly split scores into calibration and validation sets."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    perm = rng.permutation(n)
    split_idx = int(n * calibration_ratio)
    cal = scores[perm[:split_idx]]
    val = scores[perm[split_idx:]]
    return cal, val
