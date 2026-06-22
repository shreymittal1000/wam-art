"""End-to-end smoke test for the WAM-ART pipeline.

This script runs a minimal version of the full method using:
- DummyWAMAdapter (no real model weights needed)
- SimplePerturbationEditor (deterministic pixel noise)
- k-NN anomaly detection + conformal calibration
- Basic metrics

Purpose: validate that the architecture is wired correctly
before swapping in real WAMs, editors, and simulators.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wam_art.anomaly import calibrate_threshold, compute_anomaly_rates, split_scores
from wam_art.editing import SimplePerturbationEditor
from wam_art.eval import mean_absolute_error, spearman_rank_correlation
from wam_art.latents import knn_cosine_distance
from wam_art.models.dummy import DummyWAMAdapter

# ---------------------------------------------------------------------------
# Config defaults (mirrors configs/smoke_test.yaml)
# ---------------------------------------------------------------------------
NOMINAL_SAMPLES = 100
EDITED_PER_FACTOR = 50
FACTORS = [
    ("noise_light", "noise", 0.05),
    ("noise_heavy", "noise", 0.15),
    ("bright", "brightness", 0.20),
    ("dark", "darkness", 0.20),
]
SEED = 42
DEVICE = "cpu"


def generate_nominal_observations(n: int, seed: int = 42) -> list[np.ndarray]:
    """Generate dummy RGB observations (H=64, W=64, C=3)."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8) for _ in range(n)]


def main() -> int:
    print("=" * 60)
    print("WAM-ART Smoke Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Nominal samples: {NOMINAL_SAMPLES}")
    print(f"Factors: {[f[0] for f in FACTORS]}")
    print()

    # ------------------------------------------------------------------
    # 1. Load dummy WAM
    # ------------------------------------------------------------------
    wam = DummyWAMAdapter(model_name="dummy", device=DEVICE, latent_dim=128)
    wam.load()
    print(f"[1/5] Loaded dummy WAM: {wam.model_name}")

    # ------------------------------------------------------------------
    # 2. Build nominal latent reference set
    # ------------------------------------------------------------------
    nominal_images = generate_nominal_observations(NOMINAL_SAMPLES, seed=SEED)
    nominal_latents = torch.stack([wam.extract_latent(img) for img in nominal_images])
    print(f"[2/5] Extracted {len(nominal_latents)} nominal latents (dim={nominal_latents.shape[-1]})")

    # Split into train (for kNN reference) and calibration (for threshold)
    n_train = int(0.6 * NOMINAL_SAMPLES)
    train_latents = nominal_latents[:n_train]
    cal_latents = nominal_latents[n_train:]

    # ------------------------------------------------------------------
    # 3. Compute anomaly scores on nominal calibration set (for threshold)
    # ------------------------------------------------------------------
    cal_scores = knn_cosine_distance(cal_latents, train_latents, k=5)
    # Assume nominal failure rate = 5%<- > target anomaly rate = 0.95
    target_anomaly_rate = 0.05
    threshold = calibrate_threshold(cal_scores, target_anomaly_rate)
    print(f"[3/5] Calibrated conformal threshold: τ={threshold:.4f}")

    # ------------------------------------------------------------------
    # 4. Run pipeline on perturbed observations per factor
    # ------------------------------------------------------------------
    factor_names = []
    predicted_rates = []
    # For smoke test, use synthetic "measured" success rates (anti-correlated with score)
    measured_rates = []

    rng = np.random.default_rng(SEED + 1)

    for factor_name, perturbation_type, magnitude in FACTORS:
        editor = SimplePerturbationEditor(
            factor_name=factor_name,
            perturbation_type=perturbation_type,
            magnitude=magnitude,
        )

        edited_images = [
            editor.edit(img, instruction=factor_name)
            for img in nominal_images[-EDITED_PER_FACTOR:]
        ]
        edited_latents = torch.stack([wam.extract_latent(img) for img in edited_images])
        edited_scores = knn_cosine_distance(edited_latents, train_latents, k=5)
        _, anomaly_rate = compute_anomaly_rates(edited_scores, threshold)
        predicted_success = 1.0 - anomaly_rate

        # Synthetic measured rate: larger perturbation <- > lower success
        synth_measured = max(0.0, 1.0 - magnitude * 3.0 + rng.normal(0, 0.05))

        factor_names.append(factor_name)
        predicted_rates.append(predicted_success)
        measured_rates.append(synth_measured)

    predicted_arr = np.array(predicted_rates, dtype=np.float64)
    measured_arr = np.array(measured_rates, dtype=np.float64)

    print(f"[4/5] Processed {len(FACTORS)} factors with {EDITED_PER_FACTOR} samples each")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    from wam_art.eval import print_metric_report

    print_metric_report(factor_names, predicted_arr, measured_arr)

    # Smoke-test assertion: correlation should not be random
    corr, _ = spearman_rank_correlation(predicted_arr, measured_arr)
    if np.isnan(corr):
        print("FAIL: NaN correlation")
        return 1

    # Naive assertion: this is synthetic data, so we only assert pipeline ran
    print("[5/5] Smoke test completed successfully.")
    print()
    print("Next steps:")
    print("  1. Install deps:  uv pip install -e '.[dev]'")
    print("  2. Run tests:     pytest tests/")
    print("  3. Swap DummyWAMAdapter for real FastWAM / DreamZero adapter")
    print("  4. Swap SimplePerturbationEditor for diffusion-based editor")
    print("  5. Add real simulator harness in wam_art/eval/harness.py")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAM-ART smoke test")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()
    SEED = args.seed
    DEVICE = args.device
    sys.exit(main())
