# WAM-ART: Predictive Red Teaming for World Action Models

Inference-only approach to predicting WAM policy failures under visual perturbations,
adapted from the RoboART methodology.

## Quick Start

```bash
# 1. Install with uv (recommended)
uv pip install -e ".[dev]"

# 2. Run smoke test (no real model weights needed)
python scripts/smoke_test.py

# 3. Run tests
pytest tests/

# 4. Type check
mypy wam_art/

# 5. Lint
ruff check wam_art/ scripts/ tests/
```

## Project Structure

```
wam-art/
├── wam_art/
│   ├── models/        # WAM adapters (dummy, fastwam, dreamzero, ...)
│   ├── editing/       # Image perturbation + VLM critic
│   ├── latents/       # Latent extraction, distance metrics
│   ├── anomaly/       # Conformal prediction, thresholding
│   ├── eval/          # Metrics, sim harness, plotting
│   └── config/        # Config loading + schema
├── configs/           # Experiment configs (YAML)
├── scripts/           # Entry points
├── tests/             # pytest suite
├── notebooks/         # Exploratory analysis
└── data/              # Gitignored; structured by run_id
```

## Roadmap

- [ ] Phase 1: Smoke test & WAM adapter for FastWAM
- [ ] Phase 2: Core method (Approach A) with real editing
- [ ] Phase 3: Approach B — temporal latent dynamics
- [ ] Phase 4: Targeted data collection & fine-tuning (Q2)
- [ ] Phase 5: Full evaluation suite + ablations (Q3)
- [ ] Phase 6: Paper draft & submission

## Citation

TBD — placeholder for the eventual arXiv/CoRL paper.
