"""CLI benchmark runner for WAM-ART.

Examples:

    # Dummy WAM (fast, no downloads)
    python scripts/run_benchmark.py --model dummy --n-samples 20

    # OpenVLA on CPU (very slow)
    python scripts/run_benchmark.py --model openvla --n-samples 5 \
        --device cpu --no-action-divergence

    # With mock simulator (deterministic success rates)
    python scripts/run_benchmark.py --model dummy --n-samples 20 \
        --simulator mock --n-sim-episodes 10

    # With API-based VLM critic
    python scripts/run_benchmark.py --model openvla --n-samples 50 \
        --critic api
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wam_art.eval.harness import BenchmarkHarness
from wam_art.eval.viz import generate_report_plots
from wam_art.models import DummyWAMAdapter

# ---------------------------------------------------------------------------
# Default factor suite — parametric corruptions mapped to semantic descriptions
# ---------------------------------------------------------------------------
DEFAULT_FACTORS: list[tuple[str, str, dict, str | None]] = [
    ("motion_blur_light", "motion_blur", {"kernel_size": 5, "angle": 0.0},
     "slight motion blur simulating mild camera instability"),
    ("motion_blur_heavy", "motion_blur", {"kernel_size": 15, "angle": 45.0},
     "strong motion blur simulating heavy camera shake or fast motion"),
    ("gaussian_blur_light", "gaussian_blur", {"kernel_size": 5, "sigma": 1.0},
     "mild defocus blur simulating shallow depth of field"),
    ("gaussian_blur_heavy", "gaussian_blur", {"kernel_size": 11, "sigma": 3.0},
     "strong defocus blur simulating heavy out-of-focus conditions"),
    ("occlusion_light", "occlusion", {"ratio": 0.1, "position": "center"},
     "small occlusion simulating a minor distractor object partially blocking the view"),
    ("occlusion_heavy", "occlusion", {"ratio": 0.35, "position": "center"},
     "large occlusion simulating a big object (e.g. trash can) blocking most of the scene"),
    ("brightness_up", "brightness_shift", {"factor": 1.30},
     "bright lighting condition simulating strong overhead illumination"),
    ("brightness_down", "brightness_shift", {"factor": 0.60},
     "dim lighting condition simulating low ambient light"),
    ("contrast_up", "contrast_shift", {"factor": 1.40},
     "high contrast condition simulating harsh direct lighting"),
    ("contrast_down", "contrast_shift", {"factor": 0.50},
     "low contrast condition simulating fog or flat lighting"),
    ("saturation_up", "saturation_shift", {"factor": 1.50},
     "oversaturated colors simulating unusual coloured lighting (e.g. green / blue light)"),
    ("saturation_down", "saturation_shift", {"factor": 0.50},
     "desaturated colors simulating grayscale or washed-out lighting"),
    ("jpeg_heavy", "jpeg_compression", {"quality": 30},
     "heavy compression simulating low-quality camera feed or wireless transmission artefacts"),
    ("jpeg_light", "jpeg_compression", {"quality": 70},
     "mild compression simulating minor image degradation"),
    ("noise_light", "gaussian_noise", {"sigma": 0.05},
     "light noise simulating a low-quality sensor or dark environment"),
    ("noise_heavy", "gaussian_noise", {"sigma": 0.15},
     "heavy noise simulating extreme low-light or sensor malfunction"),
    ("perspective_light", "perspective_warp", {"magnitude": 0.05},
     "slight perspective warp simulating minor camera viewpoint change"),
    ("perspective_heavy", "perspective_warp", {"magnitude": 0.15},
     "strong perspective warp simulating large camera height or viewpoint change"),
    ("salt_and_pepper", "salt_and_pepper", {"amount": 0.02},
     "salt-and-pepper noise simulating sensor dead pixels or transmission errors"),
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
    # ---- Critic ----
    parser.add_argument(
        "--critic",
        choices=["dummy", "heuristic", "api"],
        default="heuristic",
        help="VLM critic backend for edit validation",
    )
    parser.add_argument(
        "--critic-model",
        type=str,
        default="openai/gpt-4o-mini",
        help="OpenRouter model identifier for API critic",
    )
    # ---- Simulator args ----
    parser.add_argument(
        "--simulator",
        choices=["mock", "libero"],
        default=None,
        help="Optional MuJoCo simulator for real success-rate evaluation",
    )
    parser.add_argument(
        "--n-sim-episodes",
        type=int,
        default=5,
        help="Number of simulated episodes per factor",
    )
    parser.add_argument(
        "--sim-task-id",
        type=int,
        default=0,
        help="LIBERO task index (or mock task ID)",
    )
    parser.add_argument(
        "--sim-benchmark",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10", "libero_100"],
        help="LIBERO sub-benchmark name",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load .env if present
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k, v)

    print("=" * 60)
    print("WAM-ART Benchmark Runner")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Device:      {args.device}")
    print(f"Samples:     {args.n_samples}")
    print(f"k-NN:        k={args.k}")
    print(f"Action div:  {'OFF' if args.no_action_divergence else 'ON'}")
    print(f"Output:      {args.output}")
    print(f"Critic:      {args.critic}")
    if args.simulator:
        print(f"Simulator:   {args.simulator} ({args.n_sim_episodes} episodes)")
    print()

    # ------------------------------------------------------------------
    # 1. Build critic
    # ------------------------------------------------------------------
    critic = None
    if args.critic == "dummy":
        from wam_art.editing import DummyCritic

        critic = DummyCritic()
    elif args.critic == "heuristic":
        from wam_art.editing import HeuristicCritic

        critic = HeuristicCritic()
    elif args.critic == "api":
        from wam_art.editing import APICritic

        print("[Critic] Initialising APICritic via OpenRouter...")
        critic = APICritic(model=args.critic_model)

    # ------------------------------------------------------------------
    # 2. Build simulator (if requested)
    # ------------------------------------------------------------------
    simulator = None
    if args.simulator == "mock":
        from wam_art.eval.simulator import MockSimulator

        simulator = MockSimulator(base_success_rate=0.8, seed=args.seed)
        print("[Simulator] MockSimulator ready")
    elif args.simulator == "libero":
        from wam_art.eval.simulator import LiberoSimulator

        print("[Simulator] Instantiating LiberoSimulator...")
        try:
            simulator = LiberoSimulator(benchmark_name=args.sim_benchmark)
        except Exception as exc:
            print(f"[Simulator] Libero init failed: {exc}")
            print("[Simulator] Falling back to MockSimulator")
            from wam_art.eval.simulator import MockSimulator

            simulator = MockSimulator(base_success_rate=0.8, seed=args.seed)
        else:
            print(f"[Simulator] LiberoSimulator loaded: {args.sim_benchmark}")

    # ------------------------------------------------------------------
    # 3. Load WAM adapter
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
    # 4. Prepare nominal observations
    # ------------------------------------------------------------------
    print(f"Generating {args.n_samples} synthetic observations...")
    nominal_images = generate_synthetic_images(
        args.n_samples, h=224, w=224, seed=args.seed
    )

    # ------------------------------------------------------------------
    # 5. Build harness & run
    # ------------------------------------------------------------------
    harness = BenchmarkHarness(
        adapter=adapter,
        nominal_images=nominal_images,
        device=args.device,
        critic=critic,
        simulator=simulator,
    )
    report = harness.run(
        factors=DEFAULT_FACTORS,
        k=args.k,
        target_anomaly_rate=args.target_anomaly_rate,
        instruction=args.instruction,
        measure_action_divergence=not args.no_action_divergence,
        n_sim_episodes=args.n_sim_episodes,
        task_id=args.sim_task_id,
    )

    # ------------------------------------------------------------------
    # 6. Save results
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
