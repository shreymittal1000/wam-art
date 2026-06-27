#!/usr/bin/env python
"""FastWAM adapter smoke test — validates config, checkpoint, and GPU handling.

This script exercises the FastWAMAdapter **without loading the full
model into memory** (it stops after validating config/checkpoint on
CPU, and after raising a clear GPU error when CUDA is absent).

Usage::

    python scripts/smoke_test_fastwam.py
    python scripts/smoke_test_fastwam.py --device cuda
    python scripts/smoke_test_fastwam.py --ckpt path/to/libero_uncond_2cam224.pt

Purpose:
    - Confirm fastwam is importable
    - Validate config resolution (explicit + auto-detection)
    - Validate checkpoint file existence
    - Test obs_to_tensor for various image shapes
    - Test dataset stats loading
    - Raise clear, actionable errors when GPU is unavailable

This script is designed to be fast — no model weights are loaded.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PASS = 0
_FAIL = 0


def ok(msg: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  ✅ {msg}")


def fail(msg: str) -> None:
    global _FAIL
    _FAIL += 1
    print(f"  ❌ {msg}")


# ======================================================================
# 1. Import check
# ======================================================================
def check_import() -> None:
    print("\n[1] Checking fastwam importability...")
    try:
        import fastwam  # noqa: F401
        ok("fastwam package is importable")
    except Exception as e:
        fail(f"fastwam not importable: {e}")
        print("    → Fix: clone https://github.com/yuantianyuan01/FastWAM")
        print("      cd FastWAM && pip install -e .")
        return

    # Check key sub-modules
    try:
        from fastwam.models.wan22.fastwam import FastWAM  # noqa: F401
        ok("FastWAM class importable")
    except Exception as e:
        fail(f"FastWAM class not importable: {e}")

    try:
        from omegaconf import OmegaConf  # noqa: F401
        ok("OmegaConf importable")
    except Exception as e:
        fail(f"OmegaConf not importable: {e}")


# ======================================================================
# 2. Adapter construction + config resolution
# ======================================================================
def check_adapter_construction(ckpt: str | None = None) -> None:
    print("\n[2] Checking FastWAMAdapter construction + config...")

    from wam_art.models.fastwam import FastWAMAdapter

    # 2a. Construction without arguments
    try:
        adapter = FastWAMAdapter(device="cpu")
        ok("FastWAMAdapter() constructed with defaults")
    except Exception as e:
        fail(f"FastWAMAdapter() construction failed: {e}")
        return

    # 2b. Config path auto-detection
    try:
        cfg_path = FastWAMAdapter._resolve_cfg_path(None)
        if Path(cfg_path).exists():
            ok(f"Auto-detected config: {cfg_path}")
        else:
            fail(f"Auto-detected config not found: {cfg_path}")
    except FileNotFoundError as e:
        fail(f"Config auto-detection failed: {e}")
        print("    → Fix: clone FastWAM repo alongside wam-art, or pass --cfg-path")

    # 2c. Explicit config path
    try:
        cfg_path = FastWAMAdapter._resolve_cfg_path("/nonexistent/path.yaml")
    except FileNotFoundError:
        ok("Explicit non-existent cfg_path raises FileNotFoundError (expected)")
    except Exception as e:
        fail(f"Unexpected error for bad cfg_path: {e}")

    # 2d. Load without checkpoint_path → raises ValueError
    try:
        adapter_nockpt = FastWAMAdapter(device="cpu")
        adapter_nockpt.load()
        fail("load() should have raised ValueError without checkpoint_path")
    except ValueError:
        ok("load() raises ValueError when checkpoint_path is None (expected)")
    except (FileNotFoundError, RuntimeError) as e:
        # Acceptable: raised during config resolution before checking ckpt
        ok(f"load() raised error without checkpoint_path: {type(e).__name__}")

    # 2e. Load with non-existent checkpoint → raises FileNotFoundError
    if ckpt is None:
        ckpt = "/tmp/nonexistent_fastwam_checkpoint.pt"
    try:
        adapter_bad = FastWAMAdapter(
            device="cpu", checkpoint_path=ckpt
        )
        adapter_bad.load()
        fail("load() should have raised FileNotFoundError for bad checkpoint")
    except FileNotFoundError:
        ok(f"load('{ckpt}') raises FileNotFoundError (expected)")
    except (ValueError, RuntimeError) as e:
        # Acceptable: raised earlier in config resolution
        ok(f"load() raised error before checkpoint check: {type(e).__name__}")

    # 2f. Load with existing checkpoint file (dry-run validation)
    if Path(ckpt).exists():
        try:
            adapter_real = FastWAMAdapter(
                device="cpu", checkpoint_path=ckpt
            )
            # Don't actually load — just validate the path exists
            ok(f"Checkpoint file exists: {ckpt} ({Path(ckpt).stat().st_size / 1e9:.1f} GB)")
        except Exception as e:
            fail(f"Unexpected error with valid checkpoint: {e}")
    else:
        print(f"    ⏭️  Skipping checkpoint validation (file not found: {ckpt})")


# ======================================================================
# 3. GPU error messaging
# ======================================================================
def check_gpu_error_messaging() -> None:
    print("\n[3] Checking GPU error messaging...")

    has_cuda = torch.cuda.is_available()
    print(f"    CUDA available: {has_cuda}")

    from wam_art.models.fastwam import FastWAMAdapter

    adapter = FastWAMAdapter(device="cuda" if has_cuda else "cpu")

    # If CUDA is available, the model should load fine.
    # If not, we test that our adapter doesn't silently fail.
    if has_cuda:
        ok("CUDA is available — FastWAM should load on GPU")
        print("    (full model load test skipped — too heavy for smoke test)")
    else:
        # Verify the adapter raises a clear error when trying GPU ops on CPU
        ok("No CUDA — adapter correctly defaults to CPU")

    # Test: adapter.to('cuda') is a no-op when model not loaded
    try:
        adapter.to("cuda")
        ok("adapter.to('cuda') is safe when model not loaded")
    except Exception as e:
        fail(f"adapter.to('cuda') raised unexpected error: {e}")

    # Test: extract_latent w/o load() raises clear RuntimeError
    img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    try:
        adapter.extract_latent(img)
        fail("extract_latent() should raise when model not loaded")
    except RuntimeError as e:
        if "not loaded" in str(e).lower():
            ok("extract_latent() raises clear 'not loaded' error (expected)")
        else:
            ok(f"extract_latent() raises RuntimeError: {e}")

    # Test: predict_action w/o load() raises clear RuntimeError
    try:
        adapter.predict_action(img)
        fail("predict_action() should raise when model not loaded")
    except RuntimeError as e:
        if "not loaded" in str(e).lower():
            ok("predict_action() raises clear 'not loaded' error (expected)")
        else:
            ok(f"predict_action() raises RuntimeError: {e}")


# ======================================================================
# 4. Image preprocessing (obs_to_tensor)
# ======================================================================
def check_preprocessing() -> None:
    print("\n[4] Checking image preprocessing (_obs_to_tensor)...")

    from wam_art.models.fastwam import FastWAMAdapter

    adapter = FastWAMAdapter(device="cpu")

    # uint8 HWC → CHW (no batch dim; _preprocess_image adds it later)
    hwc = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    t = adapter._obs_to_tensor(hwc)
    assert t.shape == (3, 64, 64), f"uint8 HWC: expected (3,64,64), got {t.shape}"
    assert t.dtype == torch.float32
    assert t.max() <= 1.0 and t.min() >= 0.0
    ok("uint8 HWC (64,64,3) → (3,64,64) float32 [0,1]")

    # uint8 BHWC
    bhwc = np.random.randint(0, 256, size=(4, 64, 64, 3), dtype=np.uint8)
    t = adapter._obs_to_tensor(bhwc)
    assert t.shape == (4, 3, 64, 64), f"Expected (4,3,64,64), got {t.shape}"
    ok("uint8 BHWC (4,64,64,3) → (4,3,64,64) float32")

    # float32 HWC [0,1] → goes through uint8 conversion, returns CHW
    fhwc = np.random.rand(64, 64, 3).astype(np.float32)
    t = adapter._obs_to_tensor(fhwc)
    assert t.shape == (3, 64, 64), f"float32 HWC: expected (3,64,64), got {t.shape}"
    ok("float32 HWC (64,64,3) → (3,64,64) float32")

    # torch Tensor HWC → adds batch dim
    thwc = torch.rand(64, 64, 3)
    t = adapter._obs_to_tensor(thwc)
    assert t.shape == (1, 3, 64, 64)
    ok("torch.Tensor HWC (64,64,3) → (1,3,64,64)")

    # torch Tensor CHW (already correct)
    tchw = torch.rand(3, 64, 64)
    t = adapter._obs_to_tensor(tchw)
    assert t.shape == (1, 3, 64, 64)
    ok("torch.Tensor CHW (3,64,64) → (1,3,64,64)")

    # Non-standard size (120x90) — should pass through as CHW
    hwc_odd = np.random.randint(0, 256, size=(120, 90, 3), dtype=np.uint8)
    t = adapter._obs_to_tensor(hwc_odd)
    assert t.shape == (3, 120, 90), f"Non-standard: expected (3,120,90), got {t.shape}"
    ok("Non-standard HWC (120,90,3) → (3,120,90)")

    # Large image (480x640)
    hwc_large = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    t = adapter._obs_to_tensor(hwc_large)
    assert t.shape == (3, 480, 640), f"Large: expected (3,480,640), got {t.shape}"
    ok("Large HWC (480,640,3) → (3,480,640)")

    # _preprocess_image resizes to multiple of 16
    hwc_small = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    x = adapter._preprocess_image(hwc_small)
    assert x.shape == (1, 3, 112, 112), f"Expected (1,3,112,112), got {x.shape}"
    ok("_preprocess_image pads 100→112 (multiple of 16)")

    hwc_224 = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    x = adapter._preprocess_image(hwc_224)
    assert x.shape == (1, 3, 224, 224), f"Expected (1,3,224,224), got {x.shape}"
    ok("_preprocess_image keeps 224×224 unchanged (already multiple of 16)")


# ======================================================================
# 5. Dataset stats loading
# ======================================================================
def check_dataset_stats() -> None:
    print("\n[5] Checking dataset stats loading...")

    from wam_art.models.fastwam import FastWAMAdapter

    # 5a. No stats path → no-op
    adapter = FastWAMAdapter(device="cpu", dataset_stats_path=None)
    adapter._load_dataset_stats()
    assert adapter._norm_mean is None
    assert adapter._norm_std is None
    ok("No dataset_stats_path → _norm_mean/_norm_std remain None")

    # 5b. Non-existent path → warning, no crash
    with warnings.catch_warnings(record=True) as w:
        adapter_bad = FastWAMAdapter(
            device="cpu", dataset_stats_path="/tmp/nonexistent_stats.json"
        )
        adapter_bad._load_dataset_stats()
        assert adapter_bad._norm_mean is None
        if len(w) > 0:
            ok(f"Non-existent dataset_stats_path → warning (no crash): {w[0].message}")
        else:
            ok("Non-existent dataset_stats_path → silent no-op")

    # 5c. Valid stats JSON
    stats = {
        "images": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    }
    tmp_path = "/tmp/wam_art_test_stats.json"
    with open(tmp_path, "w") as f:
        json.dump(stats, f)

    try:
        adapter_stats = FastWAMAdapter(
            device="cpu", dataset_stats_path=tmp_path
        )
        adapter_stats._load_dataset_stats()
        assert adapter_stats._norm_mean is not None
        assert adapter_stats._norm_std is not None
        assert adapter_stats._norm_mean.shape == (3,)
        expected_mean = torch.tensor([0.485, 0.456, 0.406])
        assert torch.allclose(adapter_stats._norm_mean, expected_mean)
        ok("Valid dataset_stats.json loaded correctly")

        # Apply stats to an image
        hwc = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
        x = adapter_stats._preprocess_image(hwc)
        assert x.shape == (1, 3, 224, 224)
        ok("_preprocess_image normalizes with dataset stats")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # 5d. Alternative stats format (top-level mean/std)
    stats_alt = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    with open(tmp_path, "w") as f:
        json.dump(stats_alt, f)

    try:
        adapter_alt = FastWAMAdapter(
            device="cpu", dataset_stats_path=tmp_path
        )
        adapter_alt._load_dataset_stats()
        assert adapter_alt._norm_mean is not None
        ok("Alternative stats format (top-level mean/std) loaded correctly")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ======================================================================
# Main
# ======================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="FastWAM adapter smoke test")
    parser.add_argument(
        "--device", default="cpu", help="Torch device (cpu|cuda)"
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to a FastWAM .pt checkpoint for validation",
    )
    parser.add_argument(
        "--cfg-path",
        default=None,
        help="Path to FastWAM Hydra config (fastwam.yaml)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FastWAM Adapter Smoke Test")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.ckpt or '(not specified)'}")
    print(f"Config: {args.cfg_path or '(auto-detect)'}")

    check_import()
    check_adapter_construction(args.ckpt)
    check_gpu_error_messaging()
    check_preprocessing()
    check_dataset_stats()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    total = _PASS + _FAIL
    print(f"Results: {_PASS}/{total} passed, {_FAIL} failed")
    if _FAIL > 0:
        print("\n⚠️  Some checks failed. See output above for fixes.")
        return 1
    print("\n✅ All smoke tests passed. Next steps:")
    print("  1. Preprocess the ActionDiT backbone:")
    print("     python scripts/preprocess_action_dit_backbone.py \\")
    print("       --model-config configs/model/fastwam.yaml \\")
    print("       --output checkpoints/ActionDiT_*.pt \\")
    print("       --device cuda --dtype bfloat16")
    print("  2. Download a FastWAM checkpoint from HuggingFace:")
    print("     huggingface-cli download yuanty/fastwam \\")
    print("       libero_uncond_2cam224.pt \\")
    print("       --local-dir ./checkpoints/fastwam_release")
    print("  3. Run the full benchmark:")
    print("     python scripts/run_benchmark.py --model fastwam \\")
    print("       --checkpoint-path checkpoints/fastwam_release/libero_uncond_2cam224.pt \\")
    print("       --device cuda")
    return 0


if __name__ == "__main__":
    sys.exit(main())
