"""Physically motivated corruption pipeline for disaster-imagery tiles,
matching the corruption families named in the AeroMesh manuscript Methods
section (fog/smoke, low light, motion blur, tile loss / block corruption).
Operates on PIL images, severity 0 (clean) through 5 (most severe), in the
style of Hendrycks & Dietterich (2019) corruption-severity benchmarks.
"""
import numpy as np
from PIL import Image, ImageFilter

CORRUPTION_TYPES = ["clean", "fog", "smoke", "low_light", "motion_blur", "tile_loss"]


def _to_array(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_image(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def apply_fog(img, severity):
    if severity == 0:
        return img
    arr = _to_array(img)
    alpha = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75][severity]
    fog_color = np.array([0.85, 0.85, 0.85])
    arr = arr * (1 - alpha) + fog_color * alpha
    return _to_image(arr)


def apply_smoke(img, severity):
    if severity == 0:
        return img
    arr = _to_array(img)
    alpha = [0.0, 0.12, 0.25, 0.40, 0.55, 0.70][severity]
    smoke_color = np.array([0.45, 0.42, 0.40])
    h, w, _ = arr.shape
    rng = np.random.RandomState(severity * 97 + 3)
    gx, gy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    noise = 0.5 + 0.5 * np.sin(6 * gx + rng.uniform(0, 6)) * np.cos(6 * gy + rng.uniform(0, 6))
    mask = (alpha * noise)[:, :, None]
    arr = arr * (1 - mask) + smoke_color * mask
    return _to_image(arr)


def apply_low_light(img, severity):
    if severity == 0:
        return img
    arr = _to_array(img)
    scale = [1.0, 0.75, 0.55, 0.40, 0.28, 0.18][severity]
    rng = np.random.RandomState(severity * 53 + 7)
    noise = rng.normal(0, 0.02 * (6 - severity) / 5.0, arr.shape)
    arr = arr * scale + noise
    return _to_image(arr)


def apply_motion_blur(img, severity):
    if severity == 0:
        return img
    radius = [0, 1, 2, 3, 5, 7][severity]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_tile_loss(img, severity):
    if severity == 0:
        return img
    arr = _to_array(img).copy()
    h, w, _ = arr.shape
    n_blocks = [0, 1, 2, 4, 7, 11][severity]
    block_size = max(4, min(h, w) // 6)
    rng = np.random.RandomState(severity * 31 + 11)
    for _ in range(n_blocks):
        by = rng.randint(0, max(1, h - block_size))
        bx = rng.randint(0, max(1, w - block_size))
        arr[by:by + block_size, bx:bx + block_size, :] = 0.0
    return _to_image(arr)


CORRUPTION_FNS = {
    "clean": lambda img, s: img,
    "fog": apply_fog,
    "smoke": apply_smoke,
    "low_light": apply_low_light,
    "motion_blur": apply_motion_blur,
    "tile_loss": apply_tile_loss,
}


def corrupt(img, corruption_type, severity):
    return CORRUPTION_FNS[corruption_type](img, severity)
