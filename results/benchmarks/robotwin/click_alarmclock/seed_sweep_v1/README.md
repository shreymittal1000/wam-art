# FastWAM RoboTwin `click_alarmclock`

This is a connected WAM-ART benchmark: the exact corrupted 384×320 three-camera
observation is passed to both FastWAM and the Wan-VAE scorer, and RoboTwin's real
task completion flag supplies measured success.

## Results

| condition | episode success | observation anomaly rate | episode mean anomaly scores |
| --- | ---: | ---: | --- |
| clean | 3/3 (100%) | 2/9 (22.2%) | 0.136, 0.152, 0.342 |
| Gaussian noise, σ=0.15 | 1/3 (33.3%) | 37/37 (100%) | 0.492, 0.494, 0.562 |

The threshold learned from the clean reference is 0.3966. Noise is cleanly
separated in latent space, but anomaly is a warning signal rather than a success
prediction: one noisy episode succeeded, and one clean episode contained two
anomalous replans while still succeeding.

## Matched design

- Task config: `demo_clean`; ALOHA-AgileX embodiment; D435 head and wrist cameras.
- FastWAM checkpoint: `yuanty/fastwam`, `robotwin_uncond_3cam_384.pt`.
- Three accepted environment seeds: 100000, 100002, 100003.
- Per-episode policy seeds: 0, 1, 2.
- Four diffusion inference steps; action horizon 32; replan interval 24.
- WAM k=5; target reference anomaly rate 0.05.
- Noise base seed 0, deterministically derived per episode and observation.

`reference/` contains the successful-observation reference bank. Each scored
condition contains `wamart.json`, the official RoboTwin result and rollout
videos under `rollouts/`, and the complete evaluator log. The `collect/` run is
retained as provenance for the reference bank.

Run date: 2026-07-19. GPU: NVIDIA A10G (AWS g5.2xlarge class).
