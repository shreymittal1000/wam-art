"""Minimal demo: OpenVLAAdapter swapped in for DummyWAM.

Downloads ~13 GB of weights from HuggingFace on first run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wam_art.editing import HeuristicCritic, RichPerturbationEditor
from wam_art.models.openvla import OpenVLAAdapter


def main() -> int:
    print("=" * 60)
    print("OpenVLAAdapter Smoke Test")
    print("=" * 60)
    print("Loading OpenVLA-7b (this will download ~13 GB on first run)...")

    adapter = OpenVLAAdapter(
        model_name="openvla/openvla-7b",
        device="cpu",
        default_instruction="pick up the object",
    )
    adapter.load()
    print(f"[1/5] Loaded {adapter.model_name}")

    # Generate one nominal image
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    print(f"[2/5] Created sample observation {img.shape}")

    # Extract latent
    print("[3/5] Extracting latent from nominal image (CPU, ~5-30s)...")
    latent_nom = adapter.extract_latent(img)
    print(f"      Nominal latent: {latent_nom.shape}, norm={latent_nom.norm().item():.4f}")

    # Apply a corruption
    editor = RichPerturbationEditor(
        factor_name="occlusion_heavy",
        corruption="occlusion",
        corruption_kwargs={"ratio": 0.35, "position": "center"},
        critic=HeuristicCritic(),
    )
    corrupted = editor.edit(img, instruction="occlusion")
    print("[4/5] Applied occlusion corruption")

    print("[5/5] Extracting latent from corrupted image (CPU, ~5-30s)...")
    latent_cor = adapter.extract_latent(corrupted)
    print(f"      Corrupted latent: {latent_cor.shape}, norm={latent_cor.norm().item():.4f}")

    # Quick distance
    cos_dist = 1 - torch.nn.functional.cosine_similarity(
        latent_nom.unsqueeze(0), latent_cor.unsqueeze(0)
    ).item()
    print(f"\nCosine distance between latents: {cos_dist:.4f}")

    # Optional: action prediction (very slow on CPU for 7B)
    # print("Predicting action on nominal image...")
    # action_nom, _ = adapter.predict_action(img)
    # print(f"Nominal action: {action_nom}")

    print("\nSmoke test completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
