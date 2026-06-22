"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr


def spearman_rank_correlation(
    predicted: NDArray[np.float64],
    measured: NDArray[np.float64],
) -> tuple[float, float]:
    """Spearman rank correlation with p-value.

    Args:
        predicted: (F,) predicted success rates.
        measured: (F,) measured (ground-truth) success rates.

    Returns:
        (correlation, p_value)
    """
    if len(predicted) != len(measured):
        raise ValueError("predicted and measured must have same length")
    if len(predicted) < 2:
        return 0.0, 1.0

    corr, p_value = spearmanr(predicted, measured)
    return float(corr), float(p_value)


def mean_absolute_error(
    predicted: NDArray[np.float64],
    measured: NDArray[np.float64],
) -> float:
    """Average absolute prediction error across factors.

    Args:
        predicted: (F,) predicted success rates.
        measured: (F,) measured success rates.

    Returns:
        Scalar MAE.
    """
    if len(predicted) != len(measured):
        raise ValueError("predicted and measured must have same length")
    return float(np.mean(np.abs(predicted - measured)))


def print_metric_report(
    factor_names: list[str],
    predicted: NDArray[np.float64],
    measured: NDArray[np.float64],
) -> None:
    """Pretty-print a factor-by-factor comparison."""
    corr, p = spearman_rank_correlation(predicted, measured)
    mae = mean_absolute_error(predicted, measured)

    print("\n" + "=" * 50)
    print("WAM-ART Evaluation Report")
    print("=" * 50)
    print(f"{'Factor':<25} {'Pred':>8} {'Meas':>8} {'Err':>8}")
    print("-" * 50)
    for name, pred, meas in zip(factor_names, predicted, measured):
        print(f"{name:<25} {pred:>8.4f} {meas:>8.4f} {abs(pred - meas):>8.4f}")
    print("-" * 50)
    print(f"Spearman correlation:  {corr:.4f} (p={p:.4f})")
    print(f"Mean absolute error:   {mae:.4f}")
    print("=" * 50 + "\n")
