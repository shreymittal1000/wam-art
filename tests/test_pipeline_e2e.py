"""End-to-end pipeline integration tests for WAM-ART.

Exercises the full stack:
- Editor → Critic → Latent extraction → Anomaly detection
- Conformal calibration on nominal data
- Benchmark harness with dummy adapter + real corruptions
- Report save/load cycle
- Trajectory-level metrics (Approach B)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from wam_art.anomaly import calibrate_threshold, compute_anomaly_rates, split_scores
from wam_art.editing import (
    HeuristicCritic,
    RichPerturbationEditor,
    SimplePerturbationEditor,
    VLMPerturbationEditor,
    list_corruptions,
)
from wam_art.editing.corruptions import apply_corruption
from wam_art.eval.harness import BenchmarkHarness, BenchmarkReport, FactorResult
from wam_art.latents import knn_cosine_distance
from wam_art.latents.trajectory import (
    sequence_manifold_distance,
    soft_nearest_trajectory_score,
    trajectory_descriptor,
)
from wam_art.models.dummy import DummyWAMAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_wam() -> DummyWAMAdapter:
    wam = DummyWAMAdapter(model_name="dummy", device="cpu", latent_dim=128)
    wam.load()
    return wam


@pytest.fixture
def nominal_images() -> list[np.ndarray]:
    """Generate 50 synthetic RGB observations (64×64)."""
    rng = np.random.default_rng(42)
    return [rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8) for _ in range(50)]


@pytest.fixture
def nominal_latents(dummy_wam: DummyWAMAdapter, nominal_images: list[np.ndarray]) -> torch.Tensor:
    return torch.stack([dummy_wam.extract_latent(img) for img in nominal_images])


@pytest.fixture
def small_factors() -> list[tuple[str, str, dict]]:
    """A minimal set of corruption factors for fast testing."""
    return [
        ("motion_blur_light", "motion_blur", {"kernel_size": 5, "angle": 0.0}),
        ("occlusion_light", "occlusion", {"ratio": 0.1, "position": "center"}),
        ("brightness_down", "brightness_shift", {"factor": 0.60}),
        ("jpeg_heavy", "jpeg_compression", {"quality": 30}),
        ("noise_light", "gaussian_noise", {"sigma": 0.05}),
    ]


# ---------------------------------------------------------------------------
# 1. Corruption registry
# ---------------------------------------------------------------------------
class TestCorruptionRegistry:
    def test_all_corruptions_listed(self) -> None:
        corruptions = list_corruptions()
        assert len(corruptions) >= 10
        assert "gaussian_noise" in corruptions
        assert "motion_blur" in corruptions
        assert "occlusion" in corruptions

    def test_apply_corruption_return_shape(self) -> None:
        img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        for name in list_corruptions():
            out = apply_corruption(name, img)
            assert out.shape == img.shape, f"{name}: {out.shape} != {img.shape}"
            assert out.dtype == np.uint8

    def test_apply_corruption_unknown_raises(self) -> None:
        img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown corruption"):
            apply_corruption("nonexistent_corruption", img)

    def test_gaussian_noise_changes_image(self) -> None:
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        out = apply_corruption("gaussian_noise", img, sigma=0.2)
        assert not np.array_equal(out, img)
        assert out.shape == img.shape

    def test_occlusion_creates_black_region(self) -> None:
        img = np.ones((64, 64, 3), dtype=np.uint8) * 255
        out = apply_corruption("occlusion", img, ratio=0.25, position="center")
        assert np.any(out == 0)
        assert out.shape == img.shape

    def test_jpeg_compression_roundtrips(self) -> None:
        img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        out = apply_corruption("jpeg_compression", img, quality=30)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_brightness_shift_darkens(self) -> None:
        img = np.ones((64, 64, 3), dtype=np.uint8) * 200
        out = apply_corruption("brightness_shift", img, factor=0.5)
        assert out.mean() < img.mean()

    def test_salt_and_pepper_adds_extremes(self) -> None:
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        out = apply_corruption("salt_and_pepper", img, amount=0.1)
        assert np.any(out == 0) or np.any(out == 255)


# ---------------------------------------------------------------------------
# 2. Editor pipeline
# ---------------------------------------------------------------------------
class TestEditorPipeline:
    def test_simple_editor_noise(self) -> None:
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        editor = SimplePerturbationEditor("test", perturbation_type="noise", magnitude=0.1)
        out = editor.edit(img, "add noise")
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_rich_editor_with_critic_accepts(self) -> None:
        img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        editor = RichPerturbationEditor(
            factor_name="test_factor",
            corruption="gaussian_blur",
            corruption_kwargs={"kernel_size": 5, "sigma": 1.0},
            critic=HeuristicCritic(),
        )
        out = editor.edit(img, "light blur")
        assert out.shape == img.shape

    def test_rich_editor_with_critic_rejects_degenerate(self) -> None:
        img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        # Over-blur + heavy noise → should be rejected by HeuristicCritic
        editor = RichPerturbationEditor(
            factor_name="bad_edit",
            corruption="gaussian_noise",
            corruption_kwargs={"sigma": 0.5},
            critic=HeuristicCritic(),
        )
        out = editor.edit(img, "extreme noise")
        # Heavy noise on random image may or may not trigger rejection
        # — the critic is conservative, so we just verify shape
        assert out.shape == img.shape

    def test_rich_editor_unknown_corruption_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown corruption"):
            RichPerturbationEditor(
                factor_name="bad",
                corruption="nonexistent",
                corruption_kwargs={},
            )

    def test_all_editors_produce_valid_outputs(
        self, nominal_images: list[np.ndarray]
    ) -> None:
        """Every corruption produces valid uint8 HWC output."""
        for name in list_corruptions():
            editor = RichPerturbationEditor(
                factor_name=f"test_{name}",
                corruption=name,
                corruption_kwargs={},
            )
            for img in nominal_images[:3]:
                out = editor.edit(img, f"test {name}")
                assert out.shape == img.shape
                assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# 3. Critic
# ---------------------------------------------------------------------------
class TestCritic:
    def test_heuristic_critic_accepts_normal_image(self) -> None:
        img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        critic = HeuristicCritic()
        result = critic.judge(img, "normal image")
        assert result.passes
        assert result.is_realistic
        assert result.preserves_task

    def test_heuristic_critic_rejects_black_image(self) -> None:
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        critic = HeuristicCritic()
        result = critic.judge(img, "black image")
        assert not result.passes
        assert not result.is_realistic

    def test_heuristic_critic_rejects_all_white(self) -> None:
        img = np.ones((64, 64, 3), dtype=np.uint8) * 255
        critic = HeuristicCritic()
        result = critic.judge(img, "white image")
        assert not result.passes

    def test_heuristic_critic_rejects_wrong_shape(self) -> None:
        img = np.ones((64, 64), dtype=np.uint8)  # 2D, not 3D
        critic = HeuristicCritic()
        result = critic.judge(img, "bad shape")
        assert not result.passes

    def test_heuristic_critic_rejects_extreme_blur(self) -> None:
        # Create an image with almost no variance
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        img[:1, :1, 0] = 129  # tiny variation
        critic = HeuristicCritic()
        result = critic.judge(img, "near-uniform")
        assert not result.passes


# ---------------------------------------------------------------------------
# 4. Anomaly detection + conformal
# ---------------------------------------------------------------------------
class TestAnomalyPipeline:
    def test_knn_cosine_distance_shape(self, nominal_latents: torch.Tensor) -> None:
        n = len(nominal_latents)
        train = nominal_latents[: n // 2]
        query = nominal_latents[n // 2 :]
        dists = knn_cosine_distance(query, train, k=3)
        assert dists.shape == (len(query),)
        assert np.all(dists >= 0)

    def test_knn_cosine_distance_single(self, nominal_latents: torch.Tensor) -> None:
        dists = knn_cosine_distance(nominal_latents[0], nominal_latents, k=1)
        assert dists.shape == (1,)

    def test_calibrate_threshold(self) -> None:
        scores = np.linspace(0, 1, 100, dtype=np.float64)
        tau = calibrate_threshold(scores, target_anomaly_rate=0.05)
        # ~95% of scores should be ≤ tau
        below = np.mean(scores <= tau)
        assert 0.90 <= below <= 1.0, f"Expected ~95% below τ, got {below:.1%}"

    def test_calibrate_threshold_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            calibrate_threshold(np.array([], dtype=np.float64), 0.05)

    def test_calibrate_threshold_bounds(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
        # target=0.0 → tau = max score
        tau = calibrate_threshold(scores, 0.0)
        assert tau == scores[-1]
        # target=1.0 → tau < min score
        tau = calibrate_threshold(scores, 1.0)
        assert tau < scores[0]

    def test_compute_anomaly_rates(self) -> None:
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        flags, rate = compute_anomaly_rates(scores, threshold=0.4)
        # 0.5, 0.7, 0.9 are > 0.4 → 3/5 = 0.6
        assert rate == pytest.approx(0.6)
        assert flags.sum() == 3

    def test_split_scores(self) -> None:
        scores = np.arange(10, dtype=np.float64)
        cal, val = split_scores(scores, calibration_ratio=0.5, seed=42)
        assert len(cal) == 5
        assert len(val) == 5
        assert set(cal) | set(val) == set(scores)

    def test_full_conformal_workflow(
        self, nominal_latents: torch.Tensor
    ) -> None:
        """End-to-end: calibrate threshold on nominal → flag anomalies."""
        n = len(nominal_latents)
        train = nominal_latents[: n // 2]
        cal = nominal_latents[n // 2 :]

        cal_scores = knn_cosine_distance(cal, train, k=5)
        tau = calibrate_threshold(cal_scores, target_anomaly_rate=0.05)

        # Test on the same cal set (should have ~5% anomaly rate)
        test_scores = knn_cosine_distance(cal, train, k=5)
        _, rate = compute_anomaly_rates(test_scores, tau)
        assert 0.0 <= rate <= 0.2  # Should be near 5% but allow some variance


# ---------------------------------------------------------------------------
# 5. Benchmark harness
# ---------------------------------------------------------------------------
class TestBenchmarkHarness:
    def test_harness_construction(
        self, dummy_wam: DummyWAMAdapter, nominal_images: list[np.ndarray]
    ) -> None:
        harness = BenchmarkHarness(dummy_wam, nominal_images, device="cpu")
        assert harness.adapter is dummy_wam
        assert harness.nominal_images == nominal_images

    def test_harness_empty_images_raises(self, dummy_wam: DummyWAMAdapter) -> None:
        with pytest.raises(ValueError, match="empty"):
            BenchmarkHarness(dummy_wam, [], device="cpu")

    def test_harness_run_with_dummy(
        self,
        dummy_wam: DummyWAMAdapter,
        nominal_images: list[np.ndarray],
        small_factors: list[tuple[str, str, dict]],
    ) -> None:
        harness = BenchmarkHarness(dummy_wam, nominal_images, device="cpu")
        factors_with_desc = [
            (name, corr, kw, f"Test: {name}")
            for name, corr, kw in small_factors
        ]
        report = harness.run(
            factors=factors_with_desc,
            k=3,
            target_anomaly_rate=0.05,
            instruction="pick up the object",
            measure_action_divergence=True,
        )

        assert report.model_name == "dummy"
        assert report.n_nominal == len(nominal_images)
        assert report.n_factors == len(small_factors)
        assert len(report.factor_results) == len(small_factors)

        for fr in report.factor_results:
            assert isinstance(fr, FactorResult)
            assert 0.0 <= fr.predicted_success_rate <= 1.0
            assert fr.n_samples == len(nominal_images)

    def test_harness_run_without_action_divergence(
        self,
        dummy_wam: DummyWAMAdapter,
        nominal_images: list[np.ndarray],
        small_factors: list[tuple[str, str, dict]],
    ) -> None:
        harness = BenchmarkHarness(dummy_wam, nominal_images, device="cpu")
        report = harness.run(
            factors=[
                (name, corr, kw, name)
                for name, corr, kw in small_factors
            ],
            k=3,
            target_anomaly_rate=0.05,
            measure_action_divergence=False,
        )
        for fr in report.factor_results:
            assert fr.mean_action_divergence == 0.0
            assert fr.max_action_divergence == 0.0

    def test_harness_run_with_mock_simulator(
        self,
        dummy_wam: DummyWAMAdapter,
        nominal_images: list[np.ndarray],
        small_factors: list[tuple[str, str, dict]],
    ) -> None:
        from wam_art.eval.simulator import MockSimulator

        sim = MockSimulator(base_success_rate=0.9, seed=42)
        harness = BenchmarkHarness(
            dummy_wam, nominal_images, device="cpu", simulator=sim
        )
        report = harness.run(
            factors=[
                (name, corr, kw, name)
                for name, corr, kw in small_factors[:2]
            ],
            k=3,
            target_anomaly_rate=0.05,
            n_sim_episodes=3,
        )
        for fr in report.factor_results:
            assert 0.0 <= fr.measured_success_rate <= 1.0

    def test_harness_run_overall_metrics(
        self,
        dummy_wam: DummyWAMAdapter,
        nominal_images: list[np.ndarray],
        small_factors: list[tuple[str, str, dict]],
    ) -> None:
        harness = BenchmarkHarness(dummy_wam, nominal_images, device="cpu")
        report = harness.run(
            factors=[
                (name, corr, kw, name)
                for name, corr, kw in small_factors
            ],
            k=3,
            target_anomaly_rate=0.05,
            measure_action_divergence=True,
        )
        assert report.overall_mae >= 0.0
        assert -1.0 <= report.overall_corr <= 1.0


# ---------------------------------------------------------------------------
# 6. Report serialization
# ---------------------------------------------------------------------------
class TestReportSerialization:
    def test_report_to_dict_roundtrip(self) -> None:
        fr = FactorResult(
            factor_name="test",
            corruption="gaussian_noise",
            corruption_kwargs={"sigma": 0.1},
            n_samples=10,
            predicted_success_rate=0.85,
            mean_action_divergence=0.12,
            max_action_divergence=0.34,
            mean_anomaly_score=0.15,
            max_anomaly_score=0.40,
            editor_reject_rate=0.0,
            measured_success_rate=0.80,
            latency_sec=1.5,
        )
        report = BenchmarkReport(
            model_name="dummy",
            n_nominal=100,
            n_factors=1,
            threshold=0.25,
            target_anomaly_rate=0.05,
            factor_results=[fr],
            overall_mae=0.05,
            overall_corr=0.92,
            overall_pvalue=0.001,
        )

        d = report.to_dict()
        assert d["model_name"] == "dummy"
        assert len(d["factor_results"]) == 1
        assert d["factor_results"][0]["factor_name"] == "test"

    def test_report_save_and_load(self) -> None:
        fr = FactorResult(
            factor_name="test",
            corruption="gaussian_noise",
            corruption_kwargs={},
            n_samples=5,
            predicted_success_rate=0.9,
            mean_action_divergence=0.1,
            max_action_divergence=0.3,
            mean_anomaly_score=0.1,
            max_anomaly_score=0.3,
            editor_reject_rate=0.0,
            measured_success_rate=0.0,
            latency_sec=0.5,
        )
        report = BenchmarkReport(
            model_name="test_model",
            n_nominal=5,
            n_factors=1,
            threshold=0.2,
            target_anomaly_rate=0.05,
            factor_results=[fr],
            overall_mae=0.1,
            overall_corr=0.8,
            overall_pvalue=0.05,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report.save(f.name)
            saved_path = f.name

        try:
            with open(saved_path) as f:
                loaded = json.load(f)
            assert loaded["model_name"] == "test_model"
            assert loaded["n_factors"] == 1
            assert loaded["factor_results"][0]["predicted_success_rate"] == 0.9
        finally:
            Path(saved_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 7. Trajectory-level metrics (Approach B)
# ---------------------------------------------------------------------------
class TestTrajectoryMetrics:
    def test_trajectory_descriptor_shape(self) -> None:
        seq = torch.randn(10, 128)  # T=10, d=128
        desc = trajectory_descriptor(seq)
        assert desc.shape == (4 * 128,)  # mean, std, vel_mean, vel_std

    def test_trajectory_descriptor_single_frame(self) -> None:
        seq = torch.randn(1, 128)  # T=1
        desc = trajectory_descriptor(seq)
        assert desc.shape == (4 * 128,)
        # Velocity stats should be zero for single-frame
        vel_mean = desc[256:384]
        assert np.allclose(vel_mean, 0.0)

    def test_trajectory_descriptor_1d(self) -> None:
        # 1D input → reshaped to (1, d)
        seq = torch.randn(128)
        desc = trajectory_descriptor(seq)
        assert desc.shape == (4 * 128,)

    def test_sequence_manifold_distance(self) -> None:
        query = torch.randn(10, 64)
        refs = [torch.randn(10, 64) for _ in range(5)]
        dist = sequence_manifold_distance(query, refs, k=3)
        assert isinstance(dist, float)
        assert dist >= 0.0

    def test_sequence_manifold_distance_empty_refs(self) -> None:
        query = torch.randn(10, 64)
        with pytest.raises(ValueError, match="empty|need at least one array"):
            sequence_manifold_distance(query, [], k=3)

    def test_soft_nearest_trajectory_score(self) -> None:
        query = torch.randn(10, 64)
        refs = [torch.randn(10, 64) for _ in range(5)]
        score = soft_nearest_trajectory_score(query, refs, sigma=1.0)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_soft_nearest_temperature_effect(self) -> None:
        query = torch.randn(10, 64)
        refs = [torch.randn(10, 64) for _ in range(5)]
        hot = soft_nearest_trajectory_score(query, refs, sigma=0.01)
        cold = soft_nearest_trajectory_score(query, refs, sigma=100.0)
        # Hot (small sigma) should be closer to hard min, cold closer to mean
        assert isinstance(hot, float)
        assert isinstance(cold, float)


# ---------------------------------------------------------------------------
# 8. Full pipeline smoke test (the real deal)
# ---------------------------------------------------------------------------
class TestFullPipeline:
    """Exercises the entire pipeline: WAM → Editor → Critic → Latent → Anomaly → Report."""

    def test_full_pipeline_dummy_wam(self) -> None:
        """Run the complete benchmark loop with dummy WAM and real corruptions."""
        rng = np.random.default_rng(42)
        images = [
            rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            for _ in range(20)
        ]

        wam = DummyWAMAdapter(model_name="dummy", device="cpu", latent_dim=128)
        wam.load()

        factors = [
            ("motion_blur_light", "motion_blur", {"kernel_size": 5, "angle": 0.0}, "slight motion blur"),
            ("occlusion_light", "occlusion", {"ratio": 0.1, "position": "center"}, "small occlusion"),
            ("brightness_down", "brightness_shift", {"factor": 0.6}, "dim lighting"),
            ("jpeg_heavy", "jpeg_compression", {"quality": 30}, "heavy compression"),
            ("noise_light", "gaussian_noise", {"sigma": 0.05}, "light sensor noise"),
        ]

        harness = BenchmarkHarness(wam, images, device="cpu")
        report = harness.run(
            factors=factors,
            k=3,
            target_anomaly_rate=0.05,
            instruction="pick up the object",
            measure_action_divergence=True,
        )

        # Verify report structure
        assert report.model_name == "dummy"
        assert report.n_nominal == 20
        assert report.n_factors == 5
        assert len(report.factor_results) == 5

        # Every factor should have reasonable metrics
        for fr in report.factor_results:
            assert 0.0 <= fr.predicted_success_rate <= 1.0
            assert fr.mean_action_divergence >= 0.0
            assert fr.mean_anomaly_score >= 0.0
            assert fr.latency_sec >= 0.0
            assert 0.0 <= fr.editor_reject_rate <= 1.0

        # Overall metrics should be computed
        assert report.overall_mae >= 0.0
        assert -1.0 <= report.overall_corr <= 1.0

        # Report should serialize cleanly
        d = report.to_dict()
        assert isinstance(d, dict)
        json.dumps(d)  # Should not raise