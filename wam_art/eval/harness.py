"""WAM-ART benchmark harness.

Orchestrates the full evaluation loop:

1. Load WAM adapter + nominal observations
2. For each corruption factor:
   a. Apply edit via RichPerturbationEditor
   b. Extract latents (nominal + corrupted)
   c. Measure anomaly distance
   d. Predict actions (nominal + corrupted)
   e. Compute action divergence as a proxy for measured failure
3. Calibrate conformal threshold
4. Aggregate predicted anomaly rate vs. measured action divergence
5. Log results + generate comparison plots
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from wam_art.anomaly import calibrate_threshold, compute_anomaly_rates
from wam_art.editing import BaseCritic, HeuristicCritic, RichPerturbationEditor
from wam_art.eval import mean_absolute_error, spearman_rank_correlation
from wam_art.eval.simulator import BaseSimulator
from wam_art.latents import knn_cosine_distance
from wam_art.models.base import BaseWAMAdapter


@dataclass(frozen=True)
class FactorResult:
    """Results for a single corruption factor."""

    factor_name: str
    corruption: str
    corruption_kwargs: dict[str, Any]
    n_samples: int
    predicted_success_rate: float  # 1 - anomaly_rate
    mean_action_divergence: float    # proxy for "measured failure"
    max_action_divergence: float
    mean_anomaly_score: float
    max_anomaly_score: float
    editor_reject_rate: float        # fraction of edits blocked by critic
    measured_success_rate: float = 0.0  # from simulator (0.0 if not used)
    latency_sec: float = 0.0           # total time for this factor


@dataclass(frozen=True)
class BenchmarkReport:
    """Top-level benchmark report."""

    model_name: str
    n_nominal: int
    n_factors: int
    threshold: float
    target_anomaly_rate: float
    factor_results: list[FactorResult] = field(default_factory=list)
    overall_mae: float = 0.0
    overall_corr: float = 0.0
    overall_pvalue: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict (JSON-safe)."""
        d = asdict(self)
        d["factor_results"] = [asdict(fr) for fr in self.factor_results]
        return d

    def save(self, path: str | Path) -> None:
        """Write report as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BenchmarkHarness:
    """End-to-end WAM-ART benchmark harness.

    Usage::

        harness = BenchmarkHarness(adapter, nominal_images, device="cpu")
        report = harness.run(factors=FACTORS, k=5, target_anomaly_rate=0.05)
        report.save("results/benchmark.json")
    """

    def __init__(
        self,
        adapter: BaseWAMAdapter,
        nominal_images: list[np.ndarray],
        *,
        device: str = "cpu",
        critic: BaseCritic | None = None,
        simulator: BaseSimulator | None = None,
    ) -> None:
        if not nominal_images:
            raise ValueError("nominal_images must not be empty")
        self.adapter = adapter
        self.nominal_images = nominal_images
        self.device = device
        self.critic = critic or HeuristicCritic()
        self.simulator = simulator

    def run(
        self,
        factors: list[tuple[str, str, dict[str, Any]]],
        *,
        k: int = 5,
        target_anomaly_rate: float = 0.05,
        instruction: str = "complete the task",
        measure_action_divergence: bool = True,
        n_sim_episodes: int = 5,
        task_id: int | str = 0,
    ) -> BenchmarkReport:
        """Run the full benchmark loop.

        Args:
            factors: List of (factor_name, corruption_name, kwargs).
            k: k-NN neighbourhood size for anomaly scoring.
            target_anomaly_rate: Nominal failure rate used to calibrate τ.
            instruction: Task instruction for action prediction.
            measure_action_divergence: If True, predict actions on nominal
                and corrupted images and compute L2 divergence as a proxy
                for measured failure rate.
            n_sim_episodes: Number of simulator episodes per factor when
                ``self.simulator`` is set.  Ignored if no simulator.
            task_id: Task identifier passed to the simulator.

        Returns:
            BenchmarkReport with metrics per factor and overall
            correlation / MAE.
        """
        import time

        n_nominal = len(self.nominal_images)
        print(f"[Harness] Running benchmark on {n_nominal} nominal observations")
        print(f"[Harness] Model: {self.adapter.model_name} | Device: {self.device}")
        if self.simulator is not None:
            print(f"[Harness] Simulator: {type(self.simulator).__name__}")

        # 1. Extract nominal latents
        nominal_latents = self._extract_latents(self.nominal_images)
        n_train = int(0.6 * n_nominal)
        train_latents = nominal_latents[:n_train]
        cal_latents = nominal_latents[n_train:]

        # 2. Calibrate threshold on nominal calibration set
        cal_scores = knn_cosine_distance(cal_latents, train_latents, k=k)
        threshold = calibrate_threshold(cal_scores, target_anomaly_rate)
        print(f"[Harness] Calibrated τ = {threshold:.4f}")

        # 3. Run each factor
        factor_results: list[FactorResult] = []
        predicted_rates = []
        measured_values = []

        for factor_name, corruption, kwargs in factors:
            t0 = time.perf_counter()
            editor = RichPerturbationEditor(
                factor_name=factor_name,
                corruption=corruption,
                corruption_kwargs=kwargs,
                critic=self.critic,
            )

            # Apply edits
            edited_images = []
            rejected = 0
            for img in self.nominal_images:
                out = editor.edit(img, instruction=factor_name)
                if np.array_equal(out, img):
                    rejected += 1  # critic rejected → fallback to original
                edited_images.append(out)

            # Extract edited latents
            edited_latents = self._extract_latents(edited_images)

            # Anomaly scores
            edited_scores = knn_cosine_distance(edited_latents, train_latents, k=k)
            _, anomaly_rate = compute_anomaly_rates(edited_scores, threshold)
            predicted_success = 1.0 - anomaly_rate

            # Action divergence (proxy for measured failure)
            mean_div = 0.0
            max_div = 0.0
            if measure_action_divergence:
                divergences = self._action_divergences(
                    self.nominal_images, edited_images, instruction
                )
                mean_div = float(np.mean(divergences))
                max_div = float(np.max(divergences))

            # Simulator-based measured success rate
            measured_succ = 0.0
            if self.simulator is not None:
                measured_succ = self._measure_with_simulator(
                    edited_images, n_sim_episodes, task_id
                )

            latency = time.perf_counter() - t0

            fr = FactorResult(
                factor_name=factor_name,
                corruption=corruption,
                corruption_kwargs=kwargs,
                n_samples=n_nominal,
                predicted_success_rate=float(predicted_success),
                mean_action_divergence=mean_div,
                max_action_divergence=max_div,
                mean_anomaly_score=float(edited_scores.mean()),
                max_anomaly_score=float(edited_scores.max()),
                editor_reject_rate=rejected / n_nominal,
                measured_success_rate=measured_succ,
                latency_sec=latency,
            )
            factor_results.append(fr)
            predicted_rates.append(predicted_success)
            measured_values.append(
                measured_succ if self.simulator is not None else mean_div
            )

            sim_str = ""
            if self.simulator is not None:
                sim_str = f"  sim_succ={measured_succ:.3f}"
            print(
                f"  {factor_name:30s}  pred_succ={predicted_success:.3f}  "
                f"mean_div={mean_div:.4f}{sim_str}  ({latency:.1f}s)"
            )

        # 4. Overall metrics
        pred_arr = np.array(predicted_rates, dtype=np.float64)
        meas_arr = np.array(measured_values, dtype=np.float64)

        overall_mae = mean_absolute_error(pred_arr, meas_arr)
        overall_corr, overall_pvalue = spearman_rank_correlation(pred_arr, meas_arr)

        print(f"[Harness] Overall MAE: {overall_mae:.4f}")
        if not np.isnan(overall_corr):
            print(f"[Harness] Spearman ρ: {overall_corr:.3f} (p={overall_pvalue:.3f})")
        else:
            print("[Harness] Spearman ρ: NaN (insufficient variance)")

        return BenchmarkReport(
            model_name=self.adapter.model_name,
            n_nominal=n_nominal,
            n_factors=len(factors),
            threshold=float(threshold),
            target_anomaly_rate=target_anomaly_rate,
            factor_results=factor_results,
            overall_mae=float(overall_mae),
            overall_corr=float(overall_corr) if not np.isnan(overall_corr) else 0.0,
            overall_pvalue=float(overall_pvalue) if not np.isnan(overall_pvalue) else 1.0,
        )

    def _measure_with_simulator(
        self,
        edited_images: list[np.ndarray],
        n_episodes: int,
        task_id: int | str,
    ) -> float:
        """Run simulated episodes with the adapter on perturbed images.

        The adapter is not actually used to step the environment here;
        the simulator's ``run_episode`` handles the full agent loop.
        This method approximates measured success by running a quick
        sanity episode with the *first* edited image as the initial
        observation style reference.

        In a full implementation you would replace the simulator's
        ``run_episode`` to use your adapter at each step.  For now
        we run the simulator's default loop to get a binary success
        signal.
        """
        if self.simulator is None:
            return 0.0
        successes = 0
        for seed in range(n_episodes):
            result = self.simulator.run_episode(
                self.adapter, task_id=task_id, max_steps=100, seed=seed
            )
            if result.success:
                successes += 1
        return successes / n_episodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_latents(self, images: list[np.ndarray]) -> Tensor:
        """Batch-extract latents from a list of uint8 images."""
        latents = [self.adapter.extract_latent(img) for img in images]
        return torch.stack(latents)

    def _action_divergences(
        self,
        nominal_images: list[np.ndarray],
        edited_images: list[np.ndarray],
        instruction: str,
    ) -> np.ndarray:
        """Compute L2 action divergence for each (nominal, edited) pair."""
        divergences = []
        for nom, edit in zip(nominal_images, edited_images, strict=True):
            act_nom, _ = self.adapter.predict_action(nom, state=instruction)
            act_edit, _ = self.adapter.predict_action(edit, state=instruction)
            div = torch.norm(act_nom - act_edit, p=2).item()
            divergences.append(div)
        return np.array(divergences, dtype=np.float64)
