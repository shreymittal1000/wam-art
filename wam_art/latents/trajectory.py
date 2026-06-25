"""Trajectory-level latent distance metrics for Approach B (temporal dynamics)."""

from __future__ import annotations

import numpy as np
import torch
from numpy import ndarray
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


def _to_numpy(x: Tensor | ndarray) -> ndarray:
    if isinstance(x, Tensor):
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        return x.detach().cpu().numpy()
    return x


def trajectory_descriptor(sequence: Tensor | ndarray) -> ndarray:
    """Compute a trajectory-level descriptor capturing state distribution and dynamics.

    Descriptor concatenates:
      - mean latent vector  (d,)
      - std latent vector   (d,)
      - mean velocity       (d,)  (temporal difference)
      - std velocity        (d,)

    Args:
        sequence: (T, d) latent trajectory.

    Returns:
        (4*d,) descriptor vector.
    """
    s = _to_numpy(sequence)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    T, d = s.shape

    mean_vec = s.mean(axis=0)
    std_vec = s.std(axis=0)

    if T >= 2:
        vel = np.diff(s, axis=0)
        vel_mean = vel.mean(axis=0)
        vel_std = vel.std(axis=0)
    else:
        vel_mean = np.zeros(d, dtype=s.dtype)
        vel_std = np.zeros(d, dtype=s.dtype)

    return np.concatenate([mean_vec, std_vec, vel_mean, vel_std])


def sequence_manifold_distance(
    sequence: Tensor | ndarray,
    reference_sequences: list[Tensor | ndarray],
    k: int = 3,
) -> float:
    """Distance of a latent trajectory to a nominal manifold.

    Computes trajectory descriptors (state mean/std + velocity mean/std)
    for the query and all references, then returns the k-NN cosine distance
    from the query descriptor to the reference descriptor cloud.

    Args:
        sequence: (T, d) query latent trajectory.
        reference_sequences: List of (T_i, d) reference trajectories.
            Lengths may vary.
        k: Number of nearest neighbours to average over.

    Returns:
        Scalar distance (average cosine distance to k nearest ref descriptors).
    """
    reference_sequences = [_to_numpy(rs) for rs in reference_sequences]
    query_desc = trajectory_descriptor(sequence).reshape(1, -1)
    ref_descs = np.stack([trajectory_descriptor(rs) for rs in reference_sequences])

    if ref_descs.shape[0] == 0:
        raise ValueError("reference_sequences must not be empty")

    n_neighbors = min(k, len(reference_sequences))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(ref_descs)
    distances, _ = nn.kneighbors(query_desc)
    return float(distances.mean(axis=1)[0])


def soft_nearest_trajectory_score(
    query_sequence: Tensor | ndarray,
    reference_sequences: list[Tensor | ndarray],
    sigma: float = 1.0,
) -> float:
    """Soft-min distance to a set of reference trajectories.

    Useful for producing a continuous anomaly score that is robust to
    outlier reference trajectories.

    Args:
        query_sequence: (T, d) query trajectory.
        reference_sequences: List of (T_i, d) reference trajectories.
        sigma: Temperature for the soft-min (lower = closer to hard min).

    Returns:
        Scalar soft-min distance.
    """
    query_sequence = _to_numpy(query_sequence)
    reference_sequences = [_to_numpy(rs) for rs in reference_sequences]

    query_desc = trajectory_descriptor(query_sequence)
    ref_descs = np.stack([trajectory_descriptor(rs) for rs in reference_sequences])

    # Euclidean distances in descriptor space
    diffs = ref_descs - query_desc.reshape(1, -1)
    dists = np.linalg.norm(diffs, axis=1)

    # Soft-min: E[dists * weights] where weights ~ exp(-dists/sigma)
    weights = np.exp(-dists / sigma)
    weights /= weights.sum() + 1e-8
    return float(np.sum(dists * weights))
