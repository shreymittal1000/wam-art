"""Distance metrics in latent space."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


def knn_cosine_distance(
    query: Tensor | np.ndarray,
    reference: Tensor | np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """Compute k-NN distance using cosine similarity.

    Args:
        query: (N, d) or (d,) query latents.
        reference: (M, d) nominal reference latents.
        k: Number of nearest neighbors.

    Returns:
        (N,) array of average cosine distances.
    """
    if isinstance(query, Tensor):
        if query.dtype == torch.bfloat16:
            query = query.to(torch.float32)
        query = query.detach().cpu().numpy()
    if isinstance(reference, Tensor):
        if reference.dtype == torch.bfloat16:
            reference = reference.to(torch.float32)
        reference = reference.detach().cpu().numpy()

    if query.ndim == 1:
        query = query.reshape(1, -1)

    # Cosine distance = 1 - cosine similarity
    n_neighbors = min(k, len(reference))
    if n_neighbors == 0 or len(reference) == 0:
        return np.zeros(len(query)) if query.ndim > 1 else np.array([0.0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(reference)
    distances, _ = nn.kneighbors(query)
    return distances.mean(axis=1)

from wam_art.latents.trajectory import sequence_manifold_distance, soft_nearest_trajectory_score, trajectory_descriptor

