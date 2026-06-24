"""Tests for real model adapters.

All heavy-dependency test cases are gated with ``skipif`` so the suite
stays green in a minimal environment (e.g. CIwithout GPUs / large
model weights).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wam_art.models import BaseWAMAdapter, DummyWAMAdapter

# ---------------------------------------------------------------------------
# OpenVLA adapter gated tests
# ---------------------------------------------------------------------------
_OPENVLA_AVAILABLE = False
try:
    from wam_art.models.openvla import OpenVLAAdapter

    _OPENVLA_AVAILABLE = True
except Exception:
    pass


@pytest.mark.skipif(not _OPENVLA_AVAILABLE, reason="openvla deps not installed")
def test_openvla_adapter_is_wam_adapter() -> None:
    assert issubclass(OpenVLAAdapter, BaseWAMAdapter)


@pytest.mark.skipif(not _OPENVLA_AVAILABLE, reason="openvla deps not installed")
def test_openvla_adapter_load_without_checkpoint_raises() -> None:
    adapter = OpenVLAAdapter(device="cpu")
    # Loading without a valid checkpoint path should raise because HF Hub
    # access is unavailable in most CI environments.
    with pytest.raises((OSError, ValueError, RuntimeError)):
        adapter.load()


@pytest.mark.skipif(not _OPENVLA_AVAILABLE, reason="openvla deps not installed")
def test_openvla_to_pil_uint8_numpy() -> None:
    adapter = OpenVLAAdapter(device="cpu")
    img_arr = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    pil = adapter._to_pil(img_arr)
    assert pil.size == (224, 224)
    assert pil.mode == "RGB"


@pytest.mark.skipif(not _OPENVLA_AVAILABLE, reason="openvla deps not installed")
def test_openvla_to_pil_float_tensor() -> None:
    adapter = OpenVLAAdapter(device="cpu")
    img = torch.rand(224, 224, 3)
    pil = adapter._to_pil(img)
    assert pil.size == (224, 224)


@pytest.mark.skipif(not _OPENVLA_AVAILABLE, reason="openvla deps not installed")
def test_openvla_resolve_instruction() -> None:
    adapter = OpenVLAAdapter(device="cpu", default_instruction="default")
    assert adapter._resolve_instruction(None) == "default"
    assert adapter._resolve_instruction("pick") == "pick"
    assert adapter._resolve_instruction({"instruction": "place"}) == "place"


# ---------------------------------------------------------------------------
# FastWAM adapter gated tests
# ---------------------------------------------------------------------------
_FASTWAM_AVAILABLE = False
try:
    import fastwam  # noqa: F401

    from wam_art.models.fastwam import FastWAMAdapter

    _FASTWAM_AVAILABLE = True
except Exception:
    pass


@pytest.mark.skipif(not _FASTWAM_AVAILABLE, reason="fastwam repo not cloned / not in PYTHONPATH")
def test_fastwam_adapter_is_wam_adapter() -> None:
    assert issubclass(FastWAMAdapter, BaseWAMAdapter)


@pytest.mark.skipif(not _FASTWAM_AVAILABLE, reason="fastwam repo not cloned / not in PYTHONPATH")
def test_fastwam_load_without_deps_raises() -> None:
    adapter = FastWAMAdapter(device="cpu")
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter.load()


@pytest.mark.skipif(not _FASTWAM_AVAILABLE, reason="fastwam repo not cloned / not in PYTHONPATH")
def test_fastwam_obs_to_tensor() -> None:
    adapter = FastWAMAdapter(device="cpu")
    hwc = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    tensor = adapter._obs_to_tensor(hwc)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == torch.float32

    bhwc = np.random.randint(0, 256, size=(2, 64, 64, 3), dtype=np.uint8)
    tensor2 = adapter._obs_to_tensor(bhwc)
    assert tensor2.shape == (2, 3, 64, 64)


@pytest.mark.skipif(not _FASTWAM_AVAILABLE, reason="fastwam repo not cloned / not in PYTHONPATH")
def test_fastwam_obs_to_tensor_from_float_tensor() -> None:
    adapter = FastWAMAdapter(device="cpu")
    img = torch.rand(64, 64, 3)
    tensor = adapter._obs_to_tensor(img)
    assert tensor.shape == (1, 3, 64, 64)


# ---------------------------------------------------------------------------
# Base class / Dummy (already covered elsewhere, minimal sanity check)
# ---------------------------------------------------------------------------
def test_dummy_is_base() -> None:
    assert issubclass(DummyWAMAdapter, BaseWAMAdapter)


def test_dummy_smoke() -> None:
    wam = DummyWAMAdapter(device="cpu", latent_dim=64)
    img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    latent = wam.extract_latent(img)
    assert latent.shape == (64,)
    assert latent.norm().item() == pytest.approx(1.0, abs=1e-5)
    action, _ = wam.predict_action(img)
    assert action.shape == (7,)
