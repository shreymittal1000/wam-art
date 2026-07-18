# Connected FastWAM + LIBERO evaluation

This is the canonical behavioral evaluation for WAM-ART. Predictions and task
labels must come from the same episode.

## Integration points

The FastWAM evaluator must perform four operations:

1. After loading the model, construct an `OnlineWAMARTScorer` using
   `fastwam_vae_latent_extractor(model)`.
2. Immediately after constructing the normalized multi-camera `image` tensor,
   call `scorer.observe(image)` before `model.infer_action(...)`.
3. At episode termination, call `scorer.end_episode(success)` with LIBERO's
   actual success signal.
4. Save either a clean reference (`collect`) or an episode report (`score`).

The tensor passed to `observe` must be the exact `[1,C,H,W]` tensor passed to
FastWAM. Do not re-render, regenerate, or separately corrupt an image for the
scorer.

## Clean reference

Run successful clean episodes with:

```text
+EVALUATION.wamart_mode=collect
+EVALUATION.wamart_reference_path=/path/to/libero_spatial_task0.npz
+EVALUATION.wamart_k=5
+EVALUATION.wamart_target_anomaly_rate=0.05
```

Only observations from successful episodes enter the nominal reference. The
artifact contains training latents, held-out calibration latents, calibration
scores, the threshold, and its calibration parameters.

Use several clean episodes for scientific runs. A single successful episode is
acceptable only as an integration check because adjacent replans are strongly
correlated.

## Corrupted scoring

Run the matched task and initial states with:

```text
+EVALUATION.wamart_mode=score
+EVALUATION.wamart_reference_path=/path/to/libero_spatial_task0.npz
+EVALUATION.wamart_k=5
+EVALUATION.corruption=occlusion
+EVALUATION.corruption_kwargs={ratio:0.35,position:center}
```

The output `*_wamart.json` contains, per episode:

- the number of real policy observations scored;
- mean and maximum WAM-latent anomaly scores;
- anomaly rate and predicted success rate;
- the measured LIBERO success label;
- task, corruption, reference, and calibration metadata.

## Experimental requirements

- Pair clean and corrupted runs by task and initial state.
- Keep model checkpoint, prompt, inference steps, action horizon, and seed fixed.
- Apply corruption before both policy inference and latent extraction.
- Never interpret `null` measured values from offline reports as failures.
- Calibrate on clean episodes that are disjoint from evaluation episodes.
- Report episode counts alongside rates.

The current extractor uses FastWAM's Wan VAE latent. Richer MoT/video-expert
representations can be added later behind the same scorer interface.
