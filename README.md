# WAM-ART: Predictive Red Teaming for World Action Models

Inference-only approach to predicting WAM policy failures under visual perturbations,
adapted from the RoboART methodology.

## Quick Start

```bash
# 1. Install with uv (recommended)
uv pip install -e ".[dev]"

# 2. Run smoke test (no real model weights needed)
python scripts/smoke_test.py

# 3. Run full benchmark (dummy WAM, fast)
python scripts/run_benchmark.py --model dummy --n-samples 20

# 4. Run with real OpenVLA weights (slow on CPU, fast on GPU)
python scripts/run_benchmark.py --model openvla --n-samples 50 \
    --device cuda --output results/openvla_run1

# 5. Run with a simulator (real task success measurement)
python scripts/run_benchmark.py --model dummy --n-samples 20 \
    --simulator mock --n-sim-episodes 10

# 6. Run tests
pytest tests/

# 7. Type check
mypy wam_art/

# 8. Lint
ruff check wam_art/ scripts/ tests/
```

## Project Structure

```
wam-art/
├── wam_art/
│   ├── models/        # WAM adapters (dummy, openvla, fastwam)
│   │   ├── base.py    # BaseWAMAdapter abstract interface
│   │   ├── dummy.py   # DummyWAMAdapter (random projection)
│   │   ├── openvla.py # OpenVLAAdapter (HF transformers, 7B)
│   │   └── fastwam.py # FastWAMAdapter (lazy-import scaffold)
│   ├── editing/       # Image perturbation + VLM critic
│   │   ├── corruptions.py   # 10 OpenCV/Pillow transforms
│   │   └── critic.py        # BaseCritic, DummyCritic, HeuristicCritic
│   ├── latents/       # Latent extraction, distance metrics
│   ├── anomaly/       # Conformal prediction, thresholding
│   ├── eval/          # Metrics, benchmark harness, plotting
│   │   ├── harness.py       # BenchmarkHarness (end-to-end loop)
│   │   └── viz.py           # matplotlib plotting helpers
│   └── config/        # Config loading + schema
├── configs/           # Experiment configs (YAML)
├── scripts/           # Entry points
│   ├── smoke_test.py            # Phase 1/2 pipeline sanity check
│   ├── smoke_test_openvla.py    # Minimal OpenVLA latent demo
│   └── run_benchmark.py         # CLI benchmark runner
├── tests/             # pytest suite (61 tests)
├── notebooks/         # Exploratory analysis
└── data/              # Gitignored; structured by run_id
```

Built-in adapters
-----------------

``wam_art.models`` ships with a thin inheritance hierarchy so swapping
models is one line:

- ``DummyWAMAdapter`` — fixed random projection, zero actions.  Used for
  smoke-testing the pipeline without downloading weights.
- ``OpenVLAAdapter`` — wraps `OpenVLA <https://huggingface.co/openvla/openvla-7b>`_
  via ``transformers.AutoModelForVision2Seq``.  Vision latent is extracted
  from the DINOv2/SigLIP backbone (mean-pooled patch features, L2-normalised).
  Install: ``pip install -e ".[openvla]"``.
- ``FastWAMAdapter`` — scaffold for `FastWAM <https://github.com/yuantianyuan01/FastWAM>`_
  (arXiv:2603.16666).  Latent uses the Wan2.2 VAE encoder.  Requires cloning
  the FastWAM repo and adding it to ``PYTHONPATH`` (see
  ``wam_art/models/fastwam.py`` docstring for full setup).

All adapters expose the same three methods:

.. code-block:: python

    adapter.load(checkpoint_path)          # or model_name for HF Hub
    latent = adapter.extract_latent(img)   # → Tensor  (d,)
    action, _ = adapter.predict_action(img, state)
    adapter.reset()

### Simulators

``wam_art.eval.simulator`` provides a pluggable interface for real
robot benchmark environments:

- ``MockSimulator`` — deterministic, no rendering.  Heavily corrupted
  observations have lower baseline success.  Perfect for CI and fast
  iteration.
- ``LiberoSimulator`` — wraps the LIBERO MuJoCo manipulation benchmark.
  Requires an off-screen renderer (EGL, OSMesa, or Xvfb+GLFW).  Falls
  back to ``MockSimulator`` automatically when rendering is unavailable.

Usage::

    from wam_art.eval.simulator import LiberoSimulator, MockSimulator

    sim = LiberoSimulator("libero_spatial")
    print(sim.list_tasks())
    result = sim.run_episode(adapter, task_id=0, max_steps=100)
    print(result.success, result.steps)

**Roadmap**

- [x] Phase 1: Smoke test & WAM adapter scaffold
- [x] Phase 2: Core method (Approach A) with real editing + benchmark harness
- [ ] Phase 3: Approach B — temporal latent dynamics
- [ ] Phase 4: Targeted data collection & fine-tuning (Q2)
- [ ] Phase 5: Full evaluation suite + ablations (Q3)
- [ ] Phase 6: Paper draft & submission

## Citation

TBD — placeholder for the eventual arXiv/CoRL paper.
