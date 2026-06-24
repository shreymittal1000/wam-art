"""End-to-end smoke test for the WAM-ART pipeline (Phase 2).

This script runs a minimal version of the full method using:
- DummyWAMAdapter (no real model weights needed)
- RichPerturbationEditor with real OpenCV/Pillow corruptions
- HeuristicCritic to sanity-check edits
- k-NN anomaly detection + conformal calibration
- Basic metrics

Purpose: validate that the architecture is wired correctly
before swapping in real WAMs and simulators.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wam_art.anomaly import calibrate_threshold, compute_anomaly_rates
from wam_art.editing import HeuristicCritic, RichPerturbationEditor
from wam_art.eval import spearman_rank_correlation
from wam_art.latents import knn_cosine_distance
from wam_art.models.dummy import DummyWAMAdapter

# ---------------------------------------------------------------------------
# Config defaults (mirrors configs/smoke_test.yaml)
# ---------------------------------------------------------------------------
NOMINAL_SAMPLES = 100
EDITED_PER_FACTOR = 50
SEED = 42
DEVICE = "cpu"

# Real corruptions for Phase 2 smoke test
FACTORS: list[tuple[str, str, dict]] = [
    ("motion_blur_light", "motion_blur", {"kernel_size": 5, "angle": 0.0}),
    ("motion_blur_heavy", "motion_blur", {"kernel_size": 15, "angle": 45.0}),
    ("occlusion_light", "occlusion", {"ratio": 0.1, "position": "center"}),
    ("occlusion_heavy", "occlusion", {"ratio": 0.35, "position": "center"}),
    ("brightness_up", "brightness_shift", {"factor": 1.30}),
    ("brightness_down", "brightness_shift", {"factor": 0.60}),
    ("contrast_up", "contrast_shift", {"factor": 1.40}),
    ("contrast_down", "contrast_shift", {"factor": 0.50}),
    ("jpeg_heavy", "jpeg_compression", {"quality": 30}),
    ("noise_heavy", "gaussian_noise", {"sigma": 0.15}),
]


def generate_nominal_observations(n: int, seed: int = 42) -> list[np.ndarray]:
    """Generate dummy RGB observations (H=64, W=64, C=3)."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8) for _ in range(n)]


def main() -> int:
    print("=" * 60)
    print("WAM-ART Smoke Test — Phase 2 (Real Corruptions)")
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
    print(
        f"[2/5] Extracted {len(nominal_latents)} nominal latents (dim={nominal_latents.shape[-1]})"
    )

    # Split into train (for kNN reference) and calibration (for threshold)
    n_train = int(0.6 * NOMINAL_SAMPLES)
    train_latents = nominal_latents[:n_train]
    cal_latents = nominal_latents[n_train:]

    # ------------------------------------------------------------------
    # 3. Compute anomaly scores on nominal calibration set (for threshold)
    # ------------------------------------------------------------------
    cal_scores = knn_cosine_distance(cal_latents, train_latents, k=5)
    # Assume nominal failure rate = 5% => target anomaly rate = 0.05
    target_anomaly_rate = 0.05
    threshold = calibrate_threshold(cal_scores, target_anomaly_rate)
    print(f"[3/5] Calibrated conformal threshold: τ={threshold:.4f}")

    # ------------------------------------------------------------------
    # 4. Run pipeline on perturbed observations per factor
    # ------------------------------------------------------------------
    factor_names = []
    predicted_rates = []
    measured_rates = []

    rng = np.random.default_rng(SEED + 1)
    critic = HeuristicCritic()

    for factor_name, corruption, kwargs in FACTORS:
        editor = RichPerturbationEditor(
            factor_name=factor_name,
            corruption=corruption,
            corruption_kwargs=kwargs,
            critic=critic,
        )

        edited_images = [
            editor.edit(img, instruction=factor_name)
            for img in nominal_images[-EDITED_PER_FACTOR:]
        ]
        edited_latents = torch.stack([wam.extract_latent(img) for img in edited_images])
        edited_scores = knn_cosine_distance(edited_latents, train_latents, k=5)
        _, anomaly_rate = compute_anomaly_rates(edited_scores, threshold)
        predicted_success = 1.0 - anomaly_rate

        # Synthetic measured rate: larger perturbation => lower success
        # We derive a heuristic "magnitude" from corruption kwargs for the
        # synthetic formula so the smoke-test correlation is non-random.
        magnitude = _extract_magnitude(corruption, kwargs)
        synth_measured = max(0.0, 1.0 - magnitude * 2.5 + rng.normal(0, 0.05))

        factor_names.append(factor_name)
        predicted_rates.append(predicted_success)
        measured_rates.append(synth_measured)

    predicted_arr = np.array(predicted_rates, dtype=np.float64)
    measured_arr = np.array(measured_rates, dtype=np.float64)

    print(
        f"[4/5] Processed {len(FACTORS)} factors "
        f"with {EDITED_PER_FACTOR} samples each"
    )

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    from wam_art.eval import print_metric_report

    print_metric_report(factor_names, predicted_arr, measured_arr)

    corr, _ = spearman_rank_correlation(predicted_arr, measured_arr)
    if np.isnan(corr):
        print("FAIL: NaN correlation")
        return 1

    print("[5/5] Smoke test completed successfully.")
    print()
    print("Next steps:")
    print("  1. Install OpenVLA deps:  uv pip install -e '.[openvla]'")
    print("  2. Run tests:             pytest tests/")
    print("  3. Swap DummyWAMAdapter for OpenVLAAdapter in smoke_test.py")
    print("  4. Add real simulator harness in wam_art/eval/harness.py")
    print("  5. Phase 3 → temporal latent dynamics (Approach B)")
    return 0


def _extract_magnitude(corruption: str, kwargs: dict) -> float:
    """Heuristic perturbation magnitude for synthetic measured-rate generation.

    Not used in the real method — only to make the smoke-test output
    non-random so we can eyeball whether correlation is sensible.
    """
    defaults: dict[str, float] = {
        "motion_blur": 0.10,
        "occlusion": 0.10,
        "brightness_shift": 0.15,
        "contrast_shift": 0.15,
        "jpeg_compression": 0.20,
        "gaussian_noise": 0.10,
    }
    base = defaults.get(corruption, 0.10)
    if corruption == "motion_blur":
        k = kwargs.get("kernel_size", 5)
        return base + (k / 100.0)
    if corruption == "occlusion":
        r = kwargs.get("ratio", 0.1)
        return base + r * 0.5
    if corruption in ("brightness_shift", "contrast_shift"):
        f = kwargs.get("factor", 1.0)
        return base + abs(f - 1.0) * 0.3
    if corruption == "jpeg_compression":
        q = kwargs.get("quality", 80)
        return base + (100 - q) / 200.0
    if corruption == "gaussian_noise":
        s = kwargs.get("sigma", 0.05)
        return base + s * 0.3
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAM-ART Phase 2 smoke test")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()
    SEED = args.seed
    DEVICE = args.device
    sys.exit(main())
