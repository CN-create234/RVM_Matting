"""Depth-guided refinement for an existing RGB human-matting alpha.

The module deliberately keeps the RGB matting model unchanged.  It builds a
robust per-pixel background-depth model from actor-free frames and uses the
depth difference only where the depth evidence is reliable.  Invalid depth and
depth boundaries fall back to the original alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEPTH_EXTENSIONS = (".png", ".tif", ".tiff", ".exr", ".npy", ".npz")


@dataclass(frozen=True)
class DepthFusionConfig:
    """Thresholds use the same unit as the input depth (normally millimetres)."""

    background_tolerance: float = 30.0
    foreground_margin: float = 100.0
    noise_multiplier: float = 3.0
    boundary_width: int = 5
    foreground_strength: float = 1.0
    background_strength: float = 1.0

    def __post_init__(self) -> None:
        if self.background_tolerance <= 0:
            raise ValueError("background_tolerance must be positive")
        if self.foreground_margin <= self.background_tolerance:
            raise ValueError("foreground_margin must exceed background_tolerance")
        if self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative")
        if self.boundary_width < 0:
            raise ValueError("boundary_width must be non-negative")
        if not 0 <= self.foreground_strength <= 1:
            raise ValueError("foreground_strength must be in [0, 1]")
        if not 0 <= self.background_strength <= 1:
            raise ValueError("background_strength must be in [0, 1]")


@dataclass(frozen=True)
class BackgroundDepthModel:
    median: np.ndarray
    robust_sigma: np.ndarray
    valid: np.ndarray

    @classmethod
    def from_frames(
        cls,
        frames: Iterable[np.ndarray],
        *,
        minimum_valid_samples: int = 3,
    ) -> "BackgroundDepthModel":
        prepared = []
        expected_shape = None
        for frame in frames:
            depth = prepare_depth(frame)
            if expected_shape is None:
                expected_shape = depth.shape
            elif depth.shape != expected_shape:
                raise ValueError(
                    f"Background depth shape mismatch: {depth.shape} != {expected_shape}"
                )
            prepared.append(np.where(depth > 0, depth, np.nan))

        if not prepared:
            raise ValueError("At least one background depth frame is required")

        stack = np.stack(prepared, axis=0).astype(np.float32, copy=False)
        counts = np.sum(np.isfinite(stack), axis=0)
        with np.errstate(all="ignore"):
            median = np.nanmedian(stack, axis=0)
            mad = np.nanmedian(np.abs(stack - median[None, ...]), axis=0)

        valid = counts >= minimum_valid_samples
        median = np.where(valid, median, 0.0).astype(np.float32)
        robust_sigma = np.where(valid, 1.4826 * mad, 0.0).astype(np.float32)
        return cls(median=median, robust_sigma=robust_sigma, valid=valid)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        maximum_frames: int = 30,
        depth_scale: float = 1.0,
        minimum_valid_samples: int = 3,
    ) -> "BackgroundDepthModel":
        paths = list_depth_files(directory)
        if maximum_frames > 0:
            paths = paths[:maximum_frames]
        return cls.from_frames(
            (load_depth(path, depth_scale=depth_scale) for path in paths),
            minimum_valid_samples=minimum_valid_samples,
        )


def prepare_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth must be single-channel, got shape {depth.shape}")
    depth = depth.astype(np.float32, copy=False)
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)


def load_depth(path: str | Path, *, depth_scale: float = 1.0) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        if not archive.files:
            raise ValueError(f"No arrays in depth archive: {path}")
        depth = archive[archive.files[0]]
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Unable to read depth image: {path}")
    return prepare_depth(depth) * float(depth_scale)


def list_depth_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Depth directory does not exist: {directory}")
    paths = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in DEPTH_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported depth files found in: {directory}")
    return paths


def find_depth_frame(directory: str | Path, stem: str) -> Path:
    directory = Path(directory)
    for extension in DEPTH_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No depth frame for '{stem}' in {directory}")


def _interior_weight(mask: np.ndarray, boundary_width: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if boundary_width <= 0:
        return mask_u8.astype(np.float32)
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    return np.clip(distance / float(boundary_width), 0.0, 1.0)


def fuse_alpha_with_depth(
    alpha: np.ndarray,
    depth: np.ndarray,
    background: BackgroundDepthModel,
    config: DepthFusionConfig = DepthFusionConfig(),
    *,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fuse an RGB matte with registered depth while preserving uncertain edges.

    Smaller depth means closer to the camera.  A positive background-current
    difference is therefore foreground evidence.  Depth values must already be
    registered to the RGB/alpha coordinate system.
    """

    alpha_array = np.asarray(alpha)
    original_dtype = alpha_array.dtype
    alpha_float = alpha_array.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        alpha_float /= float(np.iinfo(original_dtype).max)
    alpha_float = np.clip(alpha_float, 0.0, 1.0)

    current = prepare_depth(depth)
    if alpha_float.shape != current.shape or current.shape != background.median.shape:
        raise ValueError(
            "Alpha, current depth, and background depth must have identical shapes: "
            f"{alpha_float.shape}, {current.shape}, {background.median.shape}"
        )

    valid = (current > 0) & background.valid & (background.median > 0)
    delta = background.median - current
    adaptive_noise = config.noise_multiplier * background.robust_sigma
    background_limit = config.background_tolerance + adaptive_noise
    foreground_limit = config.foreground_margin + adaptive_noise
    transition = np.maximum(foreground_limit - background_limit, 1e-6)

    foreground_score = np.clip((delta - background_limit) / transition, 0.0, 1.0)
    # Keep a full-confidence plateau inside the user-specified tolerance, then
    # taper through the measured background noise band.
    background_score = np.clip(
        (background_limit - np.abs(delta)) / np.maximum(adaptive_noise, 1e-6),
        0.0,
        1.0,
    )
    foreground_score *= valid
    background_score *= valid

    foreground_score *= _interior_weight(delta > background_limit, config.boundary_width)
    background_score *= _interior_weight(
        valid & (np.abs(delta) < background_limit), config.boundary_width
    )

    # High-confidence background vetoes RGB leakage; high-confidence foreground
    # rescues missing body regions.  Uncertain/invalid depth leaves alpha intact.
    fused = alpha_float * (1.0 - config.background_strength * background_score)
    fused += config.foreground_strength * foreground_score * (1.0 - fused)
    fused = np.clip(fused, 0.0, 1.0)

    if np.issubdtype(original_dtype, np.integer):
        max_value = np.iinfo(original_dtype).max
        result = np.round(fused * max_value).astype(original_dtype)
    else:
        result = fused.astype(original_dtype, copy=False)

    if not return_diagnostics:
        return result
    return result, {
        "valid": valid.astype(np.float32),
        "delta": delta.astype(np.float32),
        "foreground_score": foreground_score.astype(np.float32),
        "background_score": background_score.astype(np.float32),
    }

