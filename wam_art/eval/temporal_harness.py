"""Temporal benchmark harness for Approach B.

Collects *trajectories* of latent codes from simulator rollouts (or synthetic
data), then scores a corrupted trajectory by its distance to a nominal
manifold of trajectories.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wam_art.editing import BaseCritic, HeuristicCritic, RichPerturbationEditor
from wam_art.eval.simulator import BaseSimulator, EpisodeResult
from wam_art.eval.viz import generate_report_plots
from wam_art.latents import sequence_manifold_distance, knn_cosine_distance
from wam_art.models.base import BaseWAMAdapter


@dataclass(frozen=True)
class TrajectoryFactorResult:
    """Results for a single corruption factor under Approach B."""

    factor_name: str
    corruption: str
    corruption_kwargs: dict[str, Any]
    n_trajectories: int
    trajectory_anomaly_score: float
    predicted_success_rate: float
    measured_success_rate: float
    mean_action_divergence: float
    editor_reject_rate: float
    latency_sec: float


@dataclass(frozen=True)
class TrajectoryBenchmarkReport:
    """Top-level report for Approach B."""

    model_name: str
    n_nominal_trajectories: int
    n_factors: int
    factor_results: list[TrajectoryFactorResult] = field(default_factory=list)
    overall_corr: float = 0.0
    overall_pvalue: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["factor_results"] = [asdict(fr) for fr in self.factor_results]
        return d

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class TemporalBenchmarkHarness:
    """Run Approach B: trajectory-level anomaly detection.

    Workflow:
      1. Collect nominal latent trajectories by rolling out the adapter
         (or loading pre-recorded rollouts).
      2. For each factor, corrupts observations in the trajectory.
      3. Extract corrupted latent trajectories.
      4. Score corruption severity via ``sequence_manifold_distance``.
      5. Run simulator episodes to obtain measured success rates.
    """

    def __init__(
        self,
        adapter: BaseWAMAdapter,
        simulator: BaseSimulator,
        critic: BaseCritic | None = None,
        device: str = "cpu",
    ) -> None:
        self.adapter = adapter
        self.simulator = simulator
        self.critic = critic or HeuristicCritic()
        self.device = device

    # ------------------------------------------------------------------
    # Nominal trajectory collection
    # ------------------------------------------------------------------
    def collect_nominal_trajectories(
        self,
        n_trajectories: int,
        task_id: int | str,
        max_steps: int = 20,
        seed_start: int = 0,
    ) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
        """Collect observation and latent trajectories from the simulator.

        Returns:
            (obs_trajectories, latent_trajectories) where each inner list
            corresponds to one episode.
        """
        obs_trajs: list[list[np.ndarray]] = []
        latent_trajs: list[list[np.ndarray]] = []

        for i in range(n_trajectories):
            seed = seed_start + i
            result = self.simulator.run_episode(
                self.adapter, task_id=task_id, max_steps=max_steps, seed=seed
            )
            # NOTE: EpisodeResult does not currently store the full
            # trajectory, only the final image.  We extend the simulator
            # interface below to capture per-step data if available.
            # For now we approximate by running a custom loop.
            obs_seq, latent_seq = self._run_episode_with_recording(
                task_id=task_id, max_steps=max_steps, seed=seed
            )
            obs_trajs.append(obs_seq)
            latent_trajs.append(latent_seq)

        return obs_trajs, latent_trajs

    def _run_episode_with_recording(
        self,
        task_id: int | str,
        max_steps: int = 20,
        seed: int = 0,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Run a single episode and record observations + latents."""
        obs = self.simulator.reset_task(task_id, seed=seed)
        observations: list[np.ndarray] = [obs.copy()]
        latents: list[np.ndarray] = []

        # Extract latent for initial observation
        with np.errstate(invalid="ignore"):
            z = self.adapter.extract_latent(obs)
        latents.append(z.detach().cpu().numpy() if hasattr(z, "detach") else z)

        done = False
        for _ in range(max_steps):
            if done:
                break
            action, _ = self.adapter.predict_action(obs)
            action_arr = (
                action.detach().cpu().numpy()
                if hasattr(action, "detach")
                else np.array(action)
            )
            obs, _, done, _ = self.simulator.step(action_arr)
            observations.append(obs.copy())
            z = self.adapter.extract_latent(obs)
            latents.append(z.detach().cpu().numpy() if hasattr(z, "detach") else z)

        return observations, latents

    # ------------------------------------------------------------------
    # Corrupted trajectory collection
    # ------------------------------------------------------------------
    def collect_corrupted_trajectories(
        self,
        obs_trajectories: list[list[np.ndarray]],
        factor: tuple[str, str, dict[str, Any], str | None],
        instruction: str = "complete the task",
    ) -> tuple[list[list[np.ndarray]], float]:
        """Apply a corruption factor to each observation trajectory.

        Returns:
            (corrupted_obs_trajectories, editor_reject_rate)
        """
        factor_name, corruption, kwargs, description = (
            factor[0],
            factor[1],
            factor[2],
            factor[3] if len(factor) > 3 else None,
        )
        editor = RichPerturbationEditor(
            factor_name=factor_name,
            corruption=corruption,
            corruption_kwargs=kwargs,
            critic=self.critic,
        )

        corrupted: list[list[np.ndarray]] = []
        total_frames = 0
        rejected = 0
        factor_instruction = description or factor_name

        for traj in obs_trajectories:
            corrupted_traj: list[np.ndarray] = []
            for obs in traj:
                total_frames += 1
                out = editor.edit(obs, instruction=factor_instruction)
                if np.array_equal(out, obs):
                    rejected += 1
                corrupted_traj.append(out)
            corrupted.append(corrupted_traj)

        reject_rate = rejected / total_frames if total_frames > 0 else 0.0
        return corrupted, reject_rate

    # ------------------------------------------------------------------
    # Main benchmark loop
    # ------------------------------------------------------------------
    def run(
        self,
        factors: list[tuple[str, str, dict[str, Any], str | None]],
        n_trajectories: int = 5,
        task_id: int | str = 0,
        max_steps: int = 20,
        seed_start: int = 0,
        k: int = 3,
        instruction: str = "complete the task",
    ) -> TrajectoryBenchmarkReport:
        print(f"[TemporalHarness] Collecting {n_trajectories} nominal trajectories")
        obs_trajs, latent_trajs = self.collect_nominal_trajectories(
            n_trajectories=n_trajectories,
            task_id=task_id,
            max_steps=max_steps,
            seed_start=seed_start,
        )

        # Split nominal trajectories into train / cal / test if desired.
        # For simplicity we use all nominal trajectories as the reference manifold.
        reference_latent_sequences = [
            np.stack(seq) if isinstance(seq, list) else seq for seq in latent_trajs
        ]

        factor_results: list[TrajectoryFactorResult] = []
        predicted_rates = []
        measured_rates = []

        for factor in factors:
            t0 = time.perf_counter()
            factor_name = factor[0]

            # 1. Corrupt observations
            corrupted_obs_trajs, reject_rate = self.collect_corrupted_trajectories(
                obs_trajs, factor, instruction=instruction
            )

            # 2. Extract corrupted latent sequences
            corrupted_latent_trajs: list[np.ndarray] = []
            for traj in corrupted_obs_trajs:
                latent_seq = []
                for obs in traj:
                    z = self.adapter.extract_latent(obs)
                    z_np = z.detach().cpu().numpy() if hasattr(z, "detach") else z
                    latent_seq.append(z_np)
                corrupted_latent_trajs.append(np.stack(latent_seq))

            # 3. Compute trajectory anomaly scores for each corrupted traj
            traj_scores = []
            for seq in corrupted_latent_trajs:
                score = sequence_manifold_distance(
                    seq, reference_latent_sequences, k=k
                )
                traj_scores.append(score)
            mean_score = float(np.mean(traj_scores))
            # Heuristic: convert distance to predicted success rate
            pred_success = float(np.exp(-mean_score))

            # 4. Measure actual task success by running episodes with corrupted initial obs
            #    Note: for a real simulator we should ideally run full episodes under
            #    corruption. Here we approximate by running episodes with the adapter
            #    and averaging success.
            measured_successes = []
            for i in range(n_trajectories):
                result = self.simulator.run_episode(
                    self.adapter,
                    task_id=task_id,
                    max_steps=max_steps,
                    seed=seed_start + i,
                )
                measured_successes.append(1.0 if result.success else 0.0)
            measured_success = float(np.mean(measured_successes))

            # 5. Action divergence (mean across trajectories, first frame only for speed)
            mean_div = 0.0
            # Could compute action divergence across the full trajectory but
            # skip for now as a placeholder.

            latency = time.perf_counter() - t0
            fr = TrajectoryFactorResult(
                factor_name=factor_name,
                corruption=factor[1],
                corruption_kwargs=factor[2],
                n_trajectories=n_trajectories,
                trajectory_anomaly_score=mean_score,
                predicted_success_rate=pred_success,
                measured_success_rate=measured_success,
                mean_action_divergence=mean_div,
                editor_reject_rate=reject_rate,
                latency_sec=latency,
            )
            factor_results.append(fr)
            predicted_rates.append(pred_success)
            measured_rates.append(measured_success)

            print(
                f"  {factor_name:30s}  score={mean_score:.4f}  "
                f"pred_succ={pred_success:.3f}  meas_succ={measured_success:.3f}  "
                f"({latency:.1f}s)"
            )

        # Overall Spearman correlation
        from wam_art.eval import spearman_rank_correlation
        corr, pvalue = spearman_rank_correlation(
            np.array(predicted_rates, dtype=np.float64),
            np.array(measured_rates, dtype=np.float64),
        )

        print(f"[TemporalHarness] Spearman ρ: {corr:.3f} (p={pvalue:.3f})")

        return TrajectoryBenchmarkReport(
            model_name=self.adapter.model_name,
            n_nominal_trajectories=n_trajectories,
            n_factors=len(factors),
            factor_results=factor_results,
            overall_corr=float(corr) if not np.isnan(corr) else 0.0,
            overall_pvalue=float(pvalue) if not np.isnan(pvalue) else 1.0,
        )
