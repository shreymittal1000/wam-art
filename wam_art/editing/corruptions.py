"""Image corruption functions for robustness testing.

All functions operate on uint8 RGB numpy arrays (H, W, 3) and return
arrays of the same dtype/shape.  Designed for use with Pillow and
OpenCV — no extra dependencies beyond the base package requirements.
"""

from __future__ import annotations

import io
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def _to_pil(img: np.ndarray) -> Image.Image:
    """Convert uint8 RGB array to PIL Image."""
    return Image.fromarray(img)


def _to_array(pil: Image.Image) -> np.ndarray:
    """Convert PIL Image to uint8 RGB array."""
    return np.array(pil, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
def gaussian_noise(
    image: np.ndarray, *, sigma: float = 0.05, seed: int | None = None
) -> np.ndarray:
    """Additive Gaussian noise (relative to [0,1] range).

    Args:
        image: uint8 RGB array.
        sigma: Standard deviation in [0,1] image space.
        seed: Optional RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma * 255, size=image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def salt_and_pepper(
    image: np.ndarray, *, amount: float = 0.01, ratio: float = 0.5, seed: int | None = None
) -> np.ndarray:
    """Salt-and-pepper impulse noise.

    Args:
        image: uint8 RGB array.
        amount: Fraction of pixels to corrupt.
        ratio: Fraction of corrupted pixels that are salt (vs pepper).
        seed: Optional RNG seed.
    """
    rng = np.random.default_rng(seed)
    img = image.copy()
    num_pixels = int(np.ceil(amount * image.size))
    coords = [rng.integers(0, image.shape[i], size=num_pixels) for i in range(3)]
    # Salt
    num_salt = int(np.ceil(num_pixels * ratio))
    img[coords[0][:num_salt], coords[1][:num_salt], :] = 255
    # Pepper
    img[coords[0][num_salt:], coords[1][num_salt:], :] = 0
    return img


# ---------------------------------------------------------------------------
# Blur
# ---------------------------------------------------------------------------
def gaussian_blur(image: np.ndarray, *, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """OpenCV Gaussian blur.

    Args:
        image: uint8 RGB array.
        kernel_size: Must be odd and >= 1.
        sigma: Gaussian kernel sigma.
    """
    k = max(1, kernel_size // 2 * 2 + 1)  # force odd
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma)


def motion_blur(
    image: np.ndarray, *, kernel_size: int = 5, angle: float = 0.0
) -> np.ndarray:
    """Directional motion blur.

    Args:
        image: uint8 RGB array.
        kernel_size: Length of motion kernel (odd).
        angle: Direction in degrees.
    """
    k = max(1, kernel_size // 2 * 2 + 1)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = np.ones(k, dtype=np.float32)
    center = (k - 1) / 2
    rot = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    # Normalize and ensure single row sums to 1 after rotation
    kernel = kernel / (kernel.sum() + 1e-8)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


# ---------------------------------------------------------------------------
# Colour / lighting
# ---------------------------------------------------------------------------
def brightness_shift(image: np.ndarray, *, factor: float = 1.0) -> np.ndarray:
    """Multiply image brightness by factor.

    factor > 1 brightens, < 1 darkens.
    """
    pil = _to_pil(image)
    enhancer = ImageEnhance.Brightness(pil)
    return _to_array(enhancer.enhance(factor))


def contrast_shift(image: np.ndarray, *, factor: float = 1.0) -> np.ndarray:
    """Multiply image contrast by factor.

    factor > 1 increases contrast, < 1 decreases it.
    """
    pil = _to_pil(image)
    enhancer = ImageEnhance.Contrast(pil)
    return _to_array(enhancer.enhance(factor))


def saturation_shift(image: np.ndarray, *, factor: float = 1.0) -> np.ndarray:
    """Multiply image saturation by factor."""
    pil = Image.fromarray(image).convert("HSV")
    enhancer = ImageEnhance.Color(pil)
    return _to_array(enhancer.enhance(factor).convert("RGB"))


# ---------------------------------------------------------------------------
# Geometric / occlusion
# ---------------------------------------------------------------------------
def occlusion(
    image: np.ndarray,
    *,
    ratio: float = 0.2,
    position: Literal["center", "random"] = "center",
    seed: int | None = None,
) -> np.ndarray:
    """Place a black rectangular occlusion on the image.

    Args:
        image: uint8 RGB array.
        ratio: Fraction of image area to occlude.
        position: "center" or "random" placement.
        seed: RNG seed when position="random".
    """
    h, w = image.shape[:2]
    area = h * w
    occ_area = int(area * ratio)
    occ_h = int(np.sqrt(occ_area * h / w))
    occ_w = int(occ_area / occ_h) if occ_h > 0 else 1
    occ_h = min(occ_h, h)
    occ_w = min(occ_w, w)

    if position == "center":
        y1 = (h - occ_h) // 2
        x1 = (w - occ_w) // 2
    else:
        rng = np.random.default_rng(seed)
        y1 = rng.integers(0, max(1, h - occ_h + 1))
        x1 = rng.integers(0, max(1, w - occ_w + 1))

    y2 = min(h, y1 + occ_h)
    x2 = min(w, x1 + occ_w)
    img = image.copy()
    img[y1:y2, x1:x2] = 0
    return img


def perspective_warp(
    image: np.ndarray, *, magnitude: float = 0.1, seed: int | None = None
) -> np.ndarray:
    """Apply a small random perspective warp.

    Args:
        image: uint8 RGB array.
        magnitude: Fractional displacement of corners (0.0 = identity).
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    # Corners: top-left, top-right, bottom-left, bottom-right
    max_shift = min(h, w) * magnitude
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    # Perturb each corner by up to max_shift
    noise = rng.uniform(-max_shift, max_shift, size=corners.shape)
    dst = (corners + noise).astype(np.float32)
    # Order must match: top-left, top-right, bottom-left, bottom-right
    # cv2.getPerspectiveTransform expects 4 points
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------
def jpeg_compression(image: np.ndarray, *, quality: int = 80) -> np.ndarray:
    """JPEG-compress and decompress image.

    Args:
        image: uint8 RGB array.
        quality: JPEG quality (1-100, higher is better).
    """
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Facility to apply by name
# ---------------------------------------------------------------------------
_CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "salt_and_pepper": salt_and_pepper,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "brightness_shift": brightness_shift,
    "contrast_shift": contrast_shift,
    "saturation_shift": saturation_shift,
    "occlusion": occlusion,
    "perspective_warp": perspective_warp,
    "jpeg_compression": jpeg_compression,
}


def list_corruptions() -> list[str]:
    """Return names of supported corruptions."""
    return sorted(_CORRUPTIONS.keys())


def apply_corruption(name: str, image: np.ndarray, **kwargs: object) -> np.ndarray:
    """Apply a named corruption to an image.

    Args:
        name: Corruption name (see list_corruptions()).
        image: uint8 RGB array.
        **kwargs: Forwarded to the corruption function.

    Returns:
        Corrupted uint8 RGB array.

    Raises:
        ValueError: If corruption name is unknown.
    """
    fn = _CORRUPTIONS.get(name)
    if fn is None:
        raise ValueError(f"Unknown corruption: {name!r}. Supported: {list_corruptions()}")
    return fn(image, **kwargs)
