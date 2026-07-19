# Benchmark results

Curated, episode-connected experiment outputs are versioned here because the
GPU instance and its local storage are ephemeral.

The hierarchy is:

```text
results/benchmarks/<simulator>/<task-or-suite>/<run-design>/
```

Each run should retain its WAM-ART JSON report, exact configuration, clean
reference latent archive, and compact rollout evidence needed to audit measured
success. Model checkpoints, simulator assets, caches, and temporary logs do not
belong here.

`libero/spatial_task0/fixed_seed42` is the original connected LIBERO batch. Its
three trials used different benchmark initial states but one policy seed; it is
preserved as historical evidence, not presented as an independent seed sweep.
