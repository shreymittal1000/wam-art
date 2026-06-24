"""Distance metrics in latent space."""

from __future__ import annotations

import numpy as np
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
        query = query.detach().cpu().numpy()
    if isinstance(reference, Tensor):
        reference = reference.detach().cpu().numpy()

    if query.ndim == 1:
        query = query.reshape(1, -1)

    # Cosine distance = 1 - cosine similarity
    nn = NearestNeighbors(n_neighbors=min(k, len(reference)), metric="cosine")
    nn.fit(reference)
    distances, _ = nn.kneighbors(query)
    return distances.mean(axis=1)


def sequence_manifold_distance(
    sequence: Tensor | np.ndarray,
    reference_sequences: list[Tensor | np.ndarray],
) -> float:
    """Average distance of a latent sequence to a nominal manifold.

    Placeholder for Approach B (temporal dynamics).
    Currently uses pointwise knn; can be upgraded to trajectory distance.

    Args:
        sequence: (T, d) latent sequence.
        reference_sequences: List of (T, d) reference sequences.

    Returns:
        Scalar distance.
    """
    # Flatten for now; proper manifold distance is future work
    raise NotImplementedError("Sequence manifold distance requires proper implementation.")
