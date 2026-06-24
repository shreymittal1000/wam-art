"""Tests for benchmark harness and visualisation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wam_art.eval.harness import BenchmarkHarness, BenchmarkReport, FactorResult
from wam_art.eval.viz import (
    generate_report_plots,
    plot_factor_comparison,
    plot_predicted_vs_measured,
)
from wam_art.models import DummyWAMAdapter


# ---------------------------------------------------------------------------
# Harness tests
# ---------------------------------------------------------------------------
def _make_images(n: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for _ in range(n)]


FACTORS: list[tuple[str, str, dict]] = [
    ("blur_light", "gaussian_blur", {"kernel_size": 5, "sigma": 1.0}),
    ("occlusion_light", "occlusion", {"ratio": 0.1, "position": "center"}),
    ("noise_light", "gaussian_noise", {"sigma": 0.05}),
]


def test_harness_runs_with_dummy_adapter() -> None:
    adapter = DummyWAMAdapter(model_name="dummy", device="cpu", latent_dim=64)
    adapter.load()
    images = _make_images(n=10)
    harness = BenchmarkHarness(adapter, images, device="cpu")
    report = harness.run(
        factors=FACTORS,
        k=3,
        target_anomaly_rate=0.05,
        measure_action_divergence=True,
    )
    assert report.model_name == "dummy"
    assert report.n_nominal == 10
    assert report.n_factors == 3
    assert len(report.factor_results) == 3
    # Basic sanity: each factor has some metric
    for fr in report.factor_results:
        assert 0.0 <= fr.predicted_success_rate <= 1.0
        assert fr.latency_sec >= 0.0
        assert fr.n_samples == 10


def test_harness_no_action_divergence() -> None:
    adapter = DummyWAMAdapter(model_name="dummy", device="cpu", latent_dim=32)
    adapter.load()
    images = _make_images(n=8)
    harness = BenchmarkHarness(adapter, images, device="cpu")
    report = harness.run(
        factors=FACTORS,
        k=2,
        measure_action_divergence=False,
    )
    for fr in report.factor_results:
        assert fr.mean_action_divergence == 0.0
        assert fr.max_action_divergence == 0.0


def test_report_json_roundtrip() -> None:
    fr = FactorResult(
        factor_name="test",
        corruption="noise",
        corruption_kwargs={"sigma": 0.1},
        n_samples=10,
        predicted_success_rate=0.8,
        mean_action_divergence=0.5,
        max_action_divergence=1.0,
        mean_anomaly_score=0.3,
        max_anomaly_score=0.6,
        editor_reject_rate=0.0,
        latency_sec=1.0,
    )
    report = BenchmarkReport(
        model_name="dummy",
        n_nominal=10,
        n_factors=1,
        threshold=0.1,
        target_anomaly_rate=0.05,
        factor_results=[fr],
        overall_mae=0.05,
        overall_corr=0.9,
        overall_pvalue=0.01,
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "report.json"
        report.save(path)
        loaded = json.loads(path.read_text())
        assert loaded["model_name"] == "dummy"
        assert len(loaded["factor_results"]) == 1
        assert loaded["factor_results"][0]["factor_name"] == "test"


def test_harness_empty_images_raises() -> None:
    adapter = DummyWAMAdapter(model_name="dummy", device="cpu", latent_dim=4)
    with pytest.raises(ValueError, match="must not be empty"):
        BenchmarkHarness(adapter, [])


# ---------------------------------------------------------------------------
# Viz tests (skip if matplotlib missing)
# ---------------------------------------------------------------------------
_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib  # noqa: F401

    _MATPLOTLIB_AVAILABLE = True
except Exception:
    pass


def _dummy_report() -> BenchmarkReport:
    frs = [
        FactorResult(
            factor_name="blur",
            corruption="gaussian_blur",
            corruption_kwargs={},
            n_samples=10,
            predicted_success_rate=0.8,
            mean_action_divergence=0.3,
            max_action_divergence=0.5,
            mean_anomaly_score=0.2,
            max_anomaly_score=0.4,
            editor_reject_rate=0.0,
            latency_sec=1.0,
        ),
        FactorResult(
            factor_name="noise",
            corruption="gaussian_noise",
            corruption_kwargs={},
            n_samples=10,
            predicted_success_rate=0.4,
            mean_action_divergence=0.8,
            max_action_divergence=1.0,
            mean_anomaly_score=0.6,
            max_anomaly_score=0.9,
            editor_reject_rate=0.0,
            latency_sec=1.0,
        ),
    ]
    return BenchmarkReport(
        model_name="dummy",
        n_nominal=10,
        n_factors=2,
        threshold=0.1,
        target_anomaly_rate=0.05,
        factor_results=frs,
        overall_mae=0.1,
        overall_corr=-0.5,
        overall_pvalue=0.3,
    )


@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
def test_plot_factor_comparison() -> None:
    report = _dummy_report()
    fig = plot_factor_comparison(report)
    assert fig is not None


@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
def test_plot_predicted_vs_measured() -> None:
    report = _dummy_report()
    fig = plot_predicted_vs_measured(report)
    assert fig is not None


@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
def test_generate_report_plots() -> None:
    report = _dummy_report()
    with tempfile.TemporaryDirectory() as td:
        paths = generate_report_plots(report, td)
        assert "factor_comparison" in paths
        assert "predicted_vs_measured" in paths
        assert Path(paths["factor_comparison"]).exists()


def test_generate_report_plots_no_matplotlib() -> None:
    """If matplotlib is missing, raises ImportError."""
    if _MATPLOTLIB_AVAILABLE:
        pytest.skip("matplotlib is installed")
    report = _dummy_report()
    with pytest.raises(ImportError, match="matplotlib"):
        generate_report_plots(report, "/tmp")
