# Connected FastWAM + RoboTwin evaluation

RoboTwin is the primary WAM-ART behavioral benchmark. The integration patch at
`integrations/fastwam/robotwin_wamart.patch` connects the upstream simulator
loop to `RobotTwinWAMARTSession` without maintaining a fork of FastWAM.

## Evaluation contract

For every FastWAM replan, the policy builds its 384×320 three-camera composite,
applies any configured corruption, and passes the exact same normalized tensor
to both FastWAM and the Wan-VAE latent scorer. At episode termination, RoboTwin's
real success flag is attached to those observations.

Each episode records three separate sources of randomness:

- `environment_seed`: the accepted RoboTwin scene/initialization seed;
- `policy_seed`: the FastWAM diffusion seed, incremented each episode;
- `corruption_seed`: a fixed run base with a documented per-observation
  derivation for stochastic corruptions.

Matched clean and corrupted runs use the same base seed and episode count, so
their environment and policy seed sequences match.

## External setup

Use the complete RoboTwin repository at FastWAM's pinned commit; the compact
copy vendored by FastWAM omits runtime files required by the evaluator.

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
git -C /path/to/RoboTwin checkout bf44be51cf5717a5595ce59447f2cf5263d2aa95
git -C /path/to/FastWAM apply /path/to/wam-art/integrations/fastwam/robotwin_wamart.patch
git -C /path/to/RoboTwin apply /path/to/wam-art/integrations/robotwin/episode_callbacks.patch
git -C /path/to/RoboTwin apply /path/to/wam-art/integrations/robotwin/runtime_compat.patch
```

Install RoboTwin using its official instructions, download the `yuanty/fastwam`
`robotwin_uncond_3cam_384.pt` checkpoint and matching dataset stats, and make
WAM-ART importable in the FastWAM environment.

The pinned RoboTwin/CuRobo snapshot also needs `warp-lang==1.15.0`. The runtime
compatibility patch initializes planner attributes before reset and adapts
CuRobo's old `wp.torch` access to Warp's current internal torch adapter.

## Run modes

First collect a clean reference from successful episodes:

```text
EVALUATION.wamart_mode=collect
EVALUATION.wamart_reference_path=/path/to/reference.npz
EVALUATION.wamart_k=5
EVALUATION.wamart_target_anomaly_rate=0.05
```

Then score matched clean or corrupted episodes:

```text
EVALUATION.wamart_mode=score
EVALUATION.wamart_reference_path=/path/to/reference.npz
EVALUATION.wamart_corruption=gaussian_noise
+EVALUATION.wamart_corruption_kwargs={sigma:0.15}
EVALUATION.wamart_corruption_seed=0
```

Curated reports, references, exact configs, and compact rollout evidence belong
under `results/benchmarks/robotwin/<task>/<run-design>/`. Checkpoints, simulator
assets, caches, and installation trees stay outside Git.
