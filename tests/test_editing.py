"""Tests for editing corruptions and critics."""

from __future__ import annotations

import numpy as np
import pytest

from wam_art.editing import (
    BaseEditor,
    RichPerturbationEditor,
    SimplePerturbationEditor,
)
from wam_art.editing.corruptions import (
    apply_corruption,
    brightness_shift,
    jpeg_compression,
    list_corruptions,
    motion_blur,
    occlusion,
    perspective_warp,
)
from wam_art.editing.critic import (
    DummyCritic,
    HeuristicCritic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sample_image(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Corruption shape / dtype tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,kw", [
    ("gaussian_noise", {"sigma": 0.05}),
    ("salt_and_pepper", {"amount": 0.01}),
    ("gaussian_blur", {"kernel_size": 5, "sigma": 1.0}),
    ("motion_blur", {"kernel_size": 5, "angle": 0.0}),
    ("brightness_shift", {"factor": 1.2}),
    ("contrast_shift", {"factor": 1.2}),
    ("saturation_shift", {"factor": 1.2}),
    ("occlusion", {"ratio": 0.2, "position": "center"}),
    ("perspective_warp", {"magnitude": 0.05}),
    ("jpeg_compression", {"quality": 80}),
])
def test_corruption_preserves_shape_dtype(name: str, kw: dict) -> None:
    img = _sample_image()
    out = apply_corruption(name, img, **kw)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_list_corruptions_returns_sorted_names() -> None:
    names = list_corruptions()
    assert len(names) > 0
    assert names == sorted(names)
    assert "gaussian_noise" in names
    assert "occlusion" in names


def test_apply_corruption_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown corruption"):
        apply_corruption("not_a_real_corruption", _sample_image())


# ---------------------------------------------------------------------------
# Specific corruption sanity checks
# ---------------------------------------------------------------------------
def test_brightness_darkens() -> None:
    img = _sample_image()
    dark = brightness_shift(img, factor=0.5)
    # On average pixel values should drop
    assert float(dark.mean()) < float(img.mean())


def test_brightness_brightens() -> None:
    img = _sample_image()
    bright = brightness_shift(img, factor=1.5)
    assert float(bright.mean()) > float(img.mean())


def test_occlusion_creates_black_pixels() -> None:
    img = _sample_image()
    occ = occlusion(img, ratio=0.2, position="center")
    assert (occ == 0).any()


def test_occlusion_random_position() -> None:
    img = _sample_image()
    occ1 = occlusion(img, ratio=0.1, position="random", seed=1)
    occ2 = occlusion(img, ratio=0.1, position="random", seed=2)
    assert not np.array_equal(occ1, occ2)


def test_jpeg_compression_changes_image() -> None:
    img = _sample_image()
    compressed = jpeg_compression(img, quality=50)
    # Not identical (lossy), still same shape/dtype
    assert not np.array_equal(compressed, img)
    assert compressed.shape == img.shape


def test_perspective_warp_shape() -> None:
    img = _sample_image()
    warped = perspective_warp(img, magnitude=0.05, seed=42)
    assert warped.shape == img.shape


def test_motion_blur_kernel_even_gets_rounded() -> None:
    img = _sample_image()
    # kernel_size=6 gets forced to 7 internally
    blurred = motion_blur(img, kernel_size=6, angle=45.0)
    assert blurred.shape == img.shape


# ---------------------------------------------------------------------------
# Critic tests
# ---------------------------------------------------------------------------
def test_dummy_critic_always_passes() -> None:
    img = _sample_image()
    critic = DummyCritic()
    result = critic.judge(img, "test")
    assert result.passes
    assert result.score == 1.0


def test_heuristic_critic_passes_normal_image() -> None:
    img = _sample_image()
    critic = HeuristicCritic()
    result = critic.judge(img, "test")
    assert result.passes
    assert result.is_realistic


def test_heuristic_critic_rejects_all_black() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    critic = HeuristicCritic()
    result = critic.judge(img, "all_black")
    assert not result.passes
    assert "degenerate" in result.reason.lower() or "uniform" in result.reason.lower()


def test_heuristic_critic_rejects_all_white() -> None:
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    critic = HeuristicCritic()
    result = critic.judge(img, "all_white")
    assert not result.passes


# ---------------------------------------------------------------------------
# Editor tests
# ---------------------------------------------------------------------------
def test_rich_editor_valid_corruption() -> None:
    img = _sample_image()
    editor = RichPerturbationEditor(
        factor_name="blur_test",
        corruption="gaussian_blur",
        corruption_kwargs={"kernel_size": 5, "sigma": 1.0},
    )
    out = editor.edit(img, instruction="blur_test")
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_rich_editor_unknown_corruption_raises() -> None:
    with pytest.raises(ValueError, match="Unknown corruption"):
        RichPerturbationEditor(
            factor_name="bad",
            corruption="not_real",
            corruption_kwargs={},
        )


def test_rich_editor_with_heuristic_critic_fallback() -> None:
    img = _sample_image()
    # Extreme occlusion will be rejected by heuristic critic -> fallback to original image
    editor = RichPerturbationEditor(
        factor_name="heavy_occlusion",
        corruption="occlusion",
        corruption_kwargs={"ratio": 0.95, "position": "center"},
        critic=HeuristicCritic(),
    )
    out = editor.edit(img, instruction="occlusion")
    # Should fall back to unmodified image because occlusion is too heavy
    assert np.array_equal(out, img)


def test_simple_editor_noise_changes_pixels() -> None:
    img = _sample_image()
    editor = SimplePerturbationEditor(
        factor_name="noise", perturbation_type="noise", magnitude=0.1
    )
    out = editor.edit(img, instruction="n/a")
    assert not np.array_equal(out, img)
    assert out.shape == img.shape


def test_base_editor_is_abstract() -> None:
    class BadEditor(BaseEditor):
        pass

    with pytest.raises(TypeError):
        BadEditor(factor_name="x")
