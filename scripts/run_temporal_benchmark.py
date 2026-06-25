"""CLI runner for Approach B (temporal latent dynamics).

Examples:

    # With DummyWAMAdapter + MockSimulator (fast, no downloads)
    python scripts/run_temporal_benchmark.py --model dummy --n-trajectories 5

    # With OpenVLA (slow on CPU)
    python scripts/run_temporal_benchmark.py --model openvla --n-trajectories 3

    # With API critic
    python scripts/run_temporal_benchmark.py --model dummy --n-trajectories 5 --critic api
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wam_art.eval.simulator import MockSimulator
from wam_art.eval.temporal_harness import TemporalBenchmarkHarness
from wam_art.eval.viz import generate_report_plots
from wam_art.models import DummyWAMAdapter

# Reuse the default factor suite from run_benchmark.py
import run_benchmark as _rb

DEFAULT_FACTORS = _rb.DEFAULT_FACTORS


def main() -> int:
    parser = argparse.ArgumentParser(description="WAM-ART Temporal Benchmark Runner")
    parser.add_argument(
        "--model",
        choices=["dummy", "openvla", "fastwam"],
        default="dummy",
        help="Which WAM adapter to benchmark",
    )
    parser.add_argument("--device", default="cpu", help="torch device string")
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=5,
        help="Number of nominal trajectories (episodes) to collect",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="k-NN neighbourhood size for trajectory manifold distance",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/temporal_benchmark",
        help="Directory to write JSON report",
    )
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
    print("WAM-ART Temporal Benchmark Runner (Approach B)")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Device:      {args.device}")
    print(f"Trajectories: {args.n_trajectories}")
    print(f"Max steps:   {args.max_steps}")
    print(f"k-NN:        k={args.k}")
    print(f"Output:      {args.output}")
    print(f"Critic:      {args.critic}")
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
    # 2. Build simulator (MockSimulator for smoke tests; Libero available later)
    # ------------------------------------------------------------------
    simulator = MockSimulator(base_success_rate=0.8, seed=args.seed)
    print("[Simulator] MockSimulator ready")

    # ------------------------------------------------------------------
    # 3. Load WAM adapter
    # ------------------------------------------------------------------
    if args.model == "dummy":
        adapter = DummyWAMAdapter(
            model_name="dummy", device=args.device, latent_dim=2176
        )
        adapter.load()
    elif args.model == "openvla":
        from wam_art.models.openvla import OpenVLAAdapter

        adapter = OpenVLAAdapter(device=args.device)
        print("Loading OpenVLA-7b weights (may download on first run)...")
        adapter.load()
    elif args.model == "fastwam":
        from wam_art.models.fastwam import FastWAMAdapter

        adapter = FastWAMAdapter(device=args.device)
        print("Loading FastWAM...")
        adapter.load()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # ------------------------------------------------------------------
    # 4. Build harness & run
    # ------------------------------------------------------------------
    harness = TemporalBenchmarkHarness(
        adapter=adapter,
        simulator=simulator,
        critic=critic,
        device=args.device,
    )
    report = harness.run(
        factors=DEFAULT_FACTORS[:6],  # subset for speed on first run
        n_trajectories=args.n_trajectories,
        task_id=0,
        max_steps=args.max_steps,
        seed_start=args.seed,
        k=args.k,
    )

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "temporal_report.json"
    report.save(json_path)
    print(f"\n[Saved] JSON report → {json_path}")

    try:
        # Reuse viz helpers if possible; otherwise skip
        from wam_art.eval.viz import generate_report_plots

        # Temporal report has different fields; fallback to simple print
        print("\nTemporal benchmark complete.")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
