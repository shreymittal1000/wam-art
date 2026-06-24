"""Tests for simulator harness."""

from __future__ import annotations

import numpy as np
import pytest

from wam_art.eval.harness import BenchmarkHarness
from wam_art.eval.simulator import (
    MockSimulator,
)
from wam_art.models import DummyWAMAdapter


# ---------------------------------------------------------------------------
# MockSimulator tests
# ---------------------------------------------------------------------------
def test_mock_simulator_list_tasks() -> None:
    sim = MockSimulator()
    tasks = sim.list_tasks()
    assert len(tasks) == 3
    assert "mock_task_0" in tasks


def test_mock_simulator_reset_task() -> None:
    sim = MockSimulator()
    img = sim.reset_task(0, seed=42)
    assert img.shape == (128, 128, 3)
    assert img.dtype == np.uint8


def test_mock_simulator_run_episode_deterministic() -> None:
    adapter = DummyWAMAdapter(device="cpu", latent_dim=64)
    adapter.load()
    sim = MockSimulator(base_success_rate=0.75, seed=0)
    result = sim.run_episode(adapter, task_id=0, max_steps=20, seed=0)
    assert isinstance(result.success, bool)
    assert 0 <= result.steps <= 20
    assert result.total_reward in (0.0, 1.0)


def test_mock_simulator_heavy_factors_lower_success() -> None:
    adapter = DummyWAMAdapter(device="cpu", latent_dim=64)
    adapter.load()
    # Inject a fake factor_name onto the adapter so MockSimulator sees it
    adapter.factor_name = "occlusion_heavy"
    sim = MockSimulator(base_success_rate=0.9, seed=0)
    heavy_results = [sim.run_episode(adapter, 0, seed=i).success for i in range(50)]
    adapter.factor_name = "occlusion_light"
    light_results = [sim.run_episode(adapter, 0, seed=i).success for i in range(50)]
    # Heavy should have lower success rate on average (effect is large)
    assert np.mean(heavy_results) < np.mean(light_results)


# ---------------------------------------------------------------------------
# BenchmarkHarness + MockSimulator integration
# ---------------------------------------------------------------------------
def test_harness_with_mock_simulator() -> None:
    adapter = DummyWAMAdapter(device="cpu", latent_dim=32)
    adapter.load()
    images = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(8)]
    sim = MockSimulator(base_success_rate=0.8, seed=0)
    harness = BenchmarkHarness(adapter, images, device="cpu", simulator=sim)
    factors = [
        ("blur_light", "gaussian_blur", {"kernel_size": 5}),
        ("blur_heavy", "gaussian_blur", {"kernel_size": 15}),
    ]
    report = harness.run(
        factors=factors,
        k=2,
        target_anomaly_rate=0.05,
        measure_action_divergence=False,
        n_sim_episodes=10,
        task_id=0,
    )
    assert report.n_factors == 2
    for fr in report.factor_results:
        # measured_success_rate should be present and between 0 and 1
        assert 0.0 <= fr.measured_success_rate <= 1.0


# ---------------------------------------------------------------------------
# LiberoSimulator import / instantiation
# ---------------------------------------------------------------------------
_LIBERO_AVAILABLE = False
try:
    import libero  # noqa: F401

    from wam_art.eval.simulator import LiberoSimulator

    _LIBERO_AVAILABLE = True
except Exception:
    pass


@pytest.mark.skipif(not _LIBERO_AVAILABLE, reason="libero not installed")
def test_libero_simulator_valid_benchmarks() -> None:
    sim = LiberoSimulator("libero_spatial")
    tasks = sim.list_tasks()
    assert len(tasks) > 0


@pytest.mark.skipif(not _LIBERO_AVAILABLE, reason="libero not installed")
def test_libero_simulator_invalid_benchmark_raises() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark"):
        LiberoSimulator("not_a_real_benchmark")


@pytest.mark.skipif(not _LIBERO_AVAILABLE, reason="libero not installed")
def test_libero_simulator_reset_without_rendering() -> None:
    # In this headless environment LIBERO rendering will fail.
    # We verify the expected error class is raised so the caller knows
    # to fall back to MockSimulator.
    from wam_art.eval.simulator import _RenderingUnavailableError

    sim = LiberoSimulator("libero_spatial", render_gpu_device_id=0)
    with pytest.raises(_RenderingUnavailableError):
        sim.reset_task(0, seed=0)
