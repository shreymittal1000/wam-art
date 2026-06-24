"""Visualization utilities for benchmark reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wam_art.eval.harness import BenchmarkReport

# Matplotlib is an optional dependency
try:
    import matplotlib
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False


def _require_matplotlib() -> None:
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )


def plot_factor_comparison(
    report: BenchmarkReport,
    output_path: str | Path | None = None,
) -> matplotlib.figure.Figure | None:
    """Bar chart comparing predicted success rate vs mean action divergence.

    Args:
        report: BenchmarkReport to visualise.
        output_path: If given, save the figure to this path.

    Returns:
        Matplotlib Figure, or None if matplotlib is not installed.
    """
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [fr.factor_name for fr in report.factor_results]
    x = np.arange(len(names))
    width = 0.35

    pred = [fr.predicted_success_rate for fr in report.factor_results]
    # Normalise mean divergence to [0,1] for plotting on same axis
    divergences = [fr.mean_action_divergence for fr in report.factor_results]
    max_div = max(divergences) if divergences else 1.0
    normed_div = [d / max_div if max_div > 0 else 0.0 for d in divergences]

    ax.bar(x - width / 2, pred, width, label="Predicted success rate", color="steelblue")
    ax.bar(x + width / 2, normed_div, width, label="Normed action divergence", color="coral")

    ax.set_ylabel("Score")
    ax.set_title(f"Factor Comparison — {report.model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.2)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_predicted_vs_measured(
    report: BenchmarkReport,
    output_path: str | Path | None = None,
) -> matplotlib.figure.Figure | None:
    """Scatter plot of predicted success rate vs measured divergence.

    Args:
        report: BenchmarkReport to visualise.
        output_path: If given, save the figure to this path.

    Returns:
        Matplotlib Figure, or None if matplotlib is not installed.
    """
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 6))

    pred = [fr.predicted_success_rate for fr in report.factor_results]
    meas = [fr.mean_action_divergence for fr in report.factor_results]
    names = [fr.factor_name for fr in report.factor_results]

    ax.scatter(pred, meas, s=120, edgecolors="k", facecolors="steelblue", zorder=3)

    for i, name in enumerate(names):
        ax.annotate(name, (pred[i], meas[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Predicted Success Rate (1 - anomaly rate)")
    ax.set_ylabel("Mean Action Divergence (proxy for failure)")
    ax.set_title(f"Predicted vs Measured — {report.model_name}")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def generate_report_plots(
    report: BenchmarkReport,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate all default plots for a benchmark report.

    Args:
        report: BenchmarkReport to visualise.
        output_dir: Directory to write PNG files into.

    Returns:
        Dict mapping plot names to file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {}

    fig1 = plot_factor_comparison(report, out / "factor_comparison.png")
    if fig1 is not None:
        paths["factor_comparison"] = str(out / "factor_comparison.png")

    fig2 = plot_predicted_vs_measured(report, out / "predicted_vs_measured.png")
    if fig2 is not None:
        paths["predicted_vs_measured"] = str(out / "predicted_vs_measured.png")

    return paths
