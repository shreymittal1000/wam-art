"""Tests for conformal prediction logic."""

from __future__ import annotations

import numpy as np
import pytest

from wam_art.anomaly import calibrate_threshold, compute_anomaly_rates, split_scores


def test_calibrate_threshold_basic() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Want 10% anomalous <- > 90th percentile threshold
    tau = calibrate_threshold(scores, target_anomaly_rate=0.1)
    # 90th percentile of 10 elements is roughly index 8
    assert 0.8 <= tau <= 0.9


def test_calibrate_threshold_extremes() -> None:
    scores = np.arange(0.0, 1.01, 0.01)
    tau_0 = calibrate_threshold(scores, target_anomaly_rate=0.0)
    assert tau_0 >= 1.0  # Nothing flagged

    tau_1 = calibrate_threshold(scores, target_anomaly_rate=1.0)
    assert tau_1 <= 0.0  # Everything flagged


def test_compute_anomaly_rates() -> None:
    scores = np.array([0.1, 0.5, 0.9])
    flags, rate = compute_anomaly_rates(scores, threshold=0.6)
    expected_flags = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(flags, expected_flags)
    assert rate == pytest.approx(1 / 3)


def test_split_scores() -> None:
    scores = np.arange(100, dtype=np.float64)
    cal, val = split_scores(scores, calibration_ratio=0.5, seed=42)
    assert len(cal) == 50
    assert len(val) == 50
    assert not np.array_equal(cal, val)


def test_calibrate_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        calibrate_threshold(np.array([]), target_anomaly_rate=0.1)


def test_calibrate_invalid_rate_raises() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibrate_threshold(np.array([0.5]), target_anomaly_rate=1.5)
