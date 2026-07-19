# LIBERO spatial task 0 — fixed policy seed 42

Connected FastWAM/WAM-ART evaluation of “pick up the black bowl between the
plate and the ramekin and place it on the plate.” Every anomaly score and
measured success label came from the same closed-loop episode.

| Condition | Parameters | Successes | Mean latent anomaly score by episode |
|---|---|---:|---|
| clean | none | 3/3 | reference collection |
| gaussian blur, heavy | kernel 11, sigma 3 | 3/3 | 0.415, 0.414, 0.438 |
| motion blur, heavy | kernel 15, angle 45° | 3/3 | 0.394, 0.401, 0.410 |
| gaussian noise, heavy | sigma 0.15 | 0/3 | 0.685, 0.688, 0.695 |
| occlusion, light | centered, area ratio 0.10 | 3/3 | 0.373, 0.388, 0.417 |
| occlusion, heavy | centered, area ratio 0.35 | 0/3 | 0.666, 0.669, 0.673 |

Important limitation: the three trials vary LIBERO initial states, but all use
policy seed 42. They are not three independent policy-seed runs. The original
Gaussian-noise run also did not record a deterministic corruption seed. RoboTwin
runs use separately recorded environment, policy, and corruption seeds.
