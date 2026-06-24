"""CLI benchmark runner for WAM-ART.

Examples:

    # Dummy WAM (fast, no downloads)
    python scripts/run_benchmark.py --model dummy --n-samples 20

    # OpenVLA on CPU (very slow)
    python scripts/run_benchmark.py --model openvla --n-samples 5 \
        --device cpu --no-action-divergence

    # OpenVLA on GPU with action divergence
    python scripts/run_benchmark.py --model openvla --n-samples 100 \
        --device cuda --output results/openvla_libero
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wam_art.eval.harness import BenchmarkHarness
from wam_art.eval.viz import generate_report_plots
from wam_art.models import DummyWAMAdapter

# ---------------------------------------------------------------------------
# Default factor suite (same as smoke_test.py but expanded)
# ---------------------------------------------------------------------------
DEFAULT_FACTORS: list[tuple[str, str, dict]] = [
    ("motion_blur_light", "motion_blur", {"kernel_size": 5, "angle": 0.0}),
    ("motion_blur_heavy", "motion_blur", {"kernel_size": 15, "angle": 45.0}),
    ("gaussian_blur_light", "gaussian_blur", {"kernel_size": 5, "sigma": 1.0}),
    ("gaussian_blur_heavy", "gaussian_blur", {"kernel_size": 11, "sigma": 3.0}),
    ("occlusion_light", "occlusion", {"ratio": 0.1, "position": "center"}),
    ("occlusion_heavy", "occlusion", {"ratio": 0.35, "position": "center"}),
    ("brightness_up", "brightness_shift", {"factor": 1.30}),
    ("brightness_down", "brightness_shift", {"factor": 0.60}),
    ("contrast_up", "contrast_shift", {"factor": 1.40}),
    ("contrast_down", "contrast_shift", {"factor": 0.50}),
    ("saturation_up", "saturation_shift", {"factor": 1.50}),
    ("saturation_down", "saturation_shift", {"factor": 0.50}),
    ("jpeg_heavy", "jpeg_compression", {"quality": 30}),
    ("jpeg_light", "jpeg_compression", {"quality": 70}),
    ("noise_light", "gaussian_noise", {"sigma": 0.05}),
    ("noise_heavy", "gaussian_noise", {"sigma": 0.15}),
    ("perspective_light", "perspective_warp", {"magnitude": 0.05}),
    ("perspective_heavy", "perspective_warp", {"magnitude": 0.15}),
    ("salt_and_pepper", "salt_and_pepper", {"amount": 0.02}),
]


def generate_synthetic_images(n: int, h: int = 224, w: int = 224, seed: int = 42) -> list[np.ndarray]:
    """Generate synthetic but natural-looking RGB images.

    Uses superposition of coloured blobs + Perlin-noise-like texture
    to create images that are more DINOv2-friendly than uniform noise.
    """
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(n):
        # Base gradient
        base = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            base[:, :, c] = rng.uniform(0.2, 0.8)

        # Add blobs
        for _ in range(rng.integers(3, 7)):
            cy, cx = rng.integers(0, h), rng.integers(0, w)
            radius = rng.integers(20, 60)
            y, x = np.ogrid[:h, :w]
            mask = ((x - cx) ** 2 + (y - cy) ** 2) < radius**2
            color = rng.uniform(0, 1, size=3)
            base[mask] = base[mask] * 0.5 + color * 0.5

        # Add noise texture
        noise = rng.normal(0, 0.05, size=(h, w, 3))
        img = np.clip(base + noise, 0, 1)
        images.append((img * 255).astype(np.uint8))
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description="WAM-ART benchmark runner")
    parser.add_argument(
        "--model",
        choices=["dummy", "openvla", "fastwam"],
        default="dummy",
        help="Which WAM adapter to benchmark",
    )
    parser.add_argument("--device", default="cpu", help="torch device string")
    parser.add_argument(
        "--n-samples", type=int, default=20, help="Number of nominal observations"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k-NN neighbourhood size for anomaly scoring",
    )
    parser.add_argument(
        "--target-anomaly-rate",
        type=float,
        default=0.05,
        help="Conformal target anomaly rate for threshold calibration",
    )
    parser.add_argument(
        "--instruction",
        default="pick up the object",
        help="Task instruction for VLA action prediction",
    )
    parser.add_argument(
        "--no-action-divergence",
        action="store_true",
        help="Skip action-divergence measurement (useful for slow CPU runs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark",
        help="Directory to write JSON report and PNG plots",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("WAM-ART Benchmark Runner")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Device:      {args.device}")
    print(f"Samples:     {args.n_samples}")
    print(f"k-NN:        k={args.k}")
    print(f"Action div:  {'OFF' if args.no_action_divergence else 'ON'}")
    print(f"Output:      {args.output}")
    print()

    # ------------------------------------------------------------------
    # 1. Load WAM adapter
    # ------------------------------------------------------------------
    if args.model == "dummy":
        adapter = DummyWAMAdapter(model_name="dummy", device=args.device, latent_dim=2176)
        adapter.load()
    elif args.model == "openvla":
        from wam_art.models.openvla import OpenVLAAdapter

        adapter = OpenVLAAdapter(device=args.device)
        print("Loading OpenVLA-7b weights (may download on first run)...")
        adapter.load()
    elif args.model == "fastwam":
        from wam_art.models.fastwam import FastWAMAdapter

        adapter = FastWAMAdapter(device=args.device)
        print("Loading FastWAM (requires repo cloned + checkpoint path)...")
        adapter.load()  # will raise if not properly configured
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # ------------------------------------------------------------------
    # 2. Prepare nominal observations
    # ------------------------------------------------------------------
    print(f"Generating {args.n_samples} synthetic observations...")
    nominal_images = generate_synthetic_images(
        args.n_samples, h=224, w=224, seed=args.seed
    )

    # ------------------------------------------------------------------
    # 3. Build harness & run
    # ------------------------------------------------------------------
    harness = BenchmarkHarness(
        adapter=adapter,
        nominal_images=nominal_images,
        device=args.device,
    )
    report = harness.run(
        factors=DEFAULT_FACTORS,
        k=args.k,
        target_anomaly_rate=args.target_anomaly_rate,
        instruction=args.instruction,
        measure_action_divergence=not args.no_action_divergence,
    )

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report.json"
    report.save(json_path)
    print(f"\n[Saved] JSON report → {json_path}")

    try:
        plot_paths = generate_report_plots(report, out_dir)
        for name, path in plot_paths.items():
            print(f"[Saved] Plot {name} → {path}")
    except ImportError:
        print("[Skipped] Plotting: install matplotlib to generate PNGs")

    print("\nBenchmark complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
