"""Tests for temporal latent metrics (Approach B)."""

import numpy as np
import pytest
import torch
from wam_art.latents.trajectory import (
    trajectory_descriptor,
    sequence_manifold_distance,
    soft_nearest_trajectory_score,
)


class TestTrajectoryDescriptor:
    def test_descriptor_shape(self):
        seq = np.random.randn(10, 32).astype(np.float32)
        desc = trajectory_descriptor(seq)
        assert desc.shape == (128,)  # 4 * 32
        assert desc.dtype == np.float32

    def test_single_frame_trajectory(self):
        seq = np.random.randn(1, 16).astype(np.float32)
        desc = trajectory_descriptor(seq)
        assert desc.shape == (64,)
        # Velocity stats should be zero for single-frame
        assert np.allclose(desc[32:48], 0.0)
        assert np.allclose(desc[48:64], 0.0)

    def test_torch_input(self):
        seq = torch.randn(10, 32)
        desc = trajectory_descriptor(seq)
        assert desc.shape == (128,)

    def test_bfloat16_input(self):
        seq = torch.randn(10, 32, dtype=torch.bfloat16)
        desc = trajectory_descriptor(seq)
        assert desc.shape == (128,)
        assert desc.dtype == np.float32


class TestSequenceManifoldDistance:
    def test_identical_sequence_zero_distance(self):
        seq = np.random.randn(10, 32).astype(np.float32)
        refs = [seq.copy()]
        dist = sequence_manifold_distance(seq, refs, k=1)
        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_different_sequences_nonzero_distance(self):
        seq = np.random.randn(10, 32).astype(np.float32)
        refs = [np.random.randn(10, 32).astype(np.float32)]
        dist = sequence_manifold_distance(seq, refs, k=1)
        assert dist > 0.0

    def test_multiple_references(self):
        seq = np.random.randn(10, 32).astype(np.float32)
        refs = [np.random.randn(10, 32).astype(np.float32) for _ in range(5)]
        dist = sequence_manifold_distance(seq, refs, k=3)
        assert 0.0 <= dist <= 1.0  # cosine distance bounds

    def test_varying_length_sequences(self):
        seq = np.random.randn(8, 32).astype(np.float32)
        refs = [
            np.random.randn(10, 32).astype(np.float32),
            np.random.randn(12, 32).astype(np.float32),
        ]
        dist = sequence_manifold_distance(seq, refs, k=2)
        assert 0.0 <= dist <= 1.0


class TestSoftNearestTrajectoryScore:
    def test_close_to_reference_low_score(self):
        seq = np.random.randn(10, 32).astype(np.float32)
        refs = [seq + 0.01 * np.random.randn(10, 32).astype(np.float32)]
        score = soft_nearest_trajectory_score(seq, refs, sigma=1.0)
        assert score >= 0.0

    def test_torch_input(self):
        seq = torch.randn(10, 32)
        refs = [torch.randn(10, 32) for _ in range(3)]
        score = soft_nearest_trajectory_score(seq, refs)
        assert isinstance(score, float)
