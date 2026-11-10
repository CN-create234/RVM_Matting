"""Conservative 2D-pose guidance for human alpha mattes.

The module consumes Sapiens2 308-keypoint JSON predictions and only raises
alpha inside high-confidence forearm/hand regions. Existing alpha values are
never reduced, so disabling pose guidance preserves the original RVM path.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LEFT_ELBOW = 7
RIGHT_ELBOW = 8
RIGHT_WRIST = 41
LEFT_WRIST = 62
RIGHT_HAND = tuple(range(21, 41))
LEFT_HAND = tuple(range(42, 62))


@dataclass(frozen=True)
class PoseFusionConfig:
    keypoint_threshold: float = 0.25
    min_hand_keypoints: int = 5
    forearm_radius_ratio: float = 0.11
    min_radius: int = 4
    max_radius: int = 32
    hand_padding_ratio: float = 0.40
    feather_radius: int = 3
    target_alpha: float = 0.98

    def __post_init__(self) -> None:
        if not 0 <= self.keypoint_threshold <= 1:
            raise ValueError("keypoint_threshold must be in [0, 1]")
        if self.min_hand_keypoints < 1:
            raise ValueError("min_hand_keypoints must be positive")
        if self.forearm_radius_ratio <= 0:
            raise ValueError("forearm_radius_ratio must be positive")
        if self.min_radius < 1 or self.max_radius < self.min_radius:
            raise ValueError("invalid radius range")
        if self.hand_padding_ratio < 0:
            raise ValueError("hand_padding_ratio must be non-negative")
        if self.feather_radius < 0:
            raise ValueError("feather_radius must be non-negative")
        if not 0 <= self.target_alpha <= 1:
            raise ValueError("target_alpha must be in [0, 1]")


class PoseSequence:
    """Frame-indexed Sapiens2 pose predictions."""

    def __init__(self, payload: dict[str, Any]):
        frames = payload.get("frames")
        if not isinstance(frames, list):
            raise ValueError("pose JSON must contain a frames list")
        self.image_size = payload.get("image_size")
        self.num_keypoints = payload.get("num_keypoints")
        self._by_name: dict[str, dict[str, Any]] = {}
        self._by_stem: dict[str, dict[str, Any]] = {}
        for frame in frames:
            name = frame.get("image_name")
            if not isinstance(name, str):
                continue
            self._by_name[name] = frame
            self._by_stem[Path(name).stem] = frame

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "PoseSequence":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def frame(self, image_name: str) -> dict[str, Any] | None:
        return self._by_name.get(image_name) or self._by_stem.get(Path(image_name).stem)

    def validate_alpha_shape(self, shape: tuple[int, int]) -> None:
        """Reject pose/alpha coordinate mismatches instead of silently shifting priors."""

        if self.image_size is None:
            return
        values = np.asarray(self.image_size).reshape(-1)
        if len(values) < 2:
            raise ValueError("pose JSON image_size must contain [height, width]")
        pose_shape = (int(values[0]), int(values[1]))
        alpha_shape = (int(shape[0]), int(shape[1]))
        if pose_shape != alpha_shape:
            raise ValueError(
                "pose JSON and alpha resolution differ: "
                f"pose={pose_shape}, alpha={alpha_shape}. "
                "Run pose estimation on the same-resolution source frames."
            )


def _as_pose(instance: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    keypoints = np.asarray(instance.get("keypoints", []), dtype=np.float32)
    scores = np.asarray(instance.get("keypoint_scores", []), dtype=np.float32).reshape(-1)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        return None
    if len(scores) != len(keypoints) or len(keypoints) <= LEFT_WRIST:
        return None
    return keypoints, scores


def select_instance(
    frame: dict[str, Any] | None,
    alpha: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Select the person most consistent with the current RVM foreground."""

    if frame is None:
        return None
    instances = [item for item in frame.get("instances", []) if _as_pose(item) is not None]
    if not instances:
        return None
    if len(instances) == 1 or alpha is None:
        return max(instances, key=lambda item: _bbox_area(item.get("bbox", [])))

    height, width = alpha.shape[:2]
    alpha_float = alpha.astype(np.float32) / 255.0

    def score(item: dict[str, Any]) -> float:
        bbox = np.asarray(item.get("bbox", []), dtype=np.float32).reshape(-1)
        if len(bbox) < 4:
            return -1.0
        x1, y1, x2, y2 = bbox[:4]
        ix1 = int(np.clip(np.floor(x1), 0, width - 1))
        iy1 = int(np.clip(np.floor(y1), 0, height - 1))
        ix2 = int(np.clip(np.ceil(x2), ix1 + 1, width))
        iy2 = int(np.clip(np.ceil(y2), iy1 + 1, height))
        crop = alpha_float[iy1:iy2, ix1:ix2]
        return float(crop.sum()) / max(float(crop.size), 1.0)

    return max(instances, key=score)


def _bbox_area(bbox: Any) -> float:
    values = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if len(values) < 4:
        return 0.0
    return max(float(values[2] - values[0]), 0.0) * max(
        float(values[3] - values[1]), 0.0
    )


def _point(keypoints: np.ndarray, index: int) -> tuple[int, int]:
    return tuple(np.round(keypoints[index]).astype(np.int32).tolist())


def _draw_soft_region(
    prior: np.ndarray,
    mask: np.ndarray,
    confidence: float,
) -> None:
    if confidence <= 0:
        return
    np.maximum(prior, mask.astype(np.float32) * float(confidence), out=prior)


def build_pose_prior(
    shape: tuple[int, int],
    instance: dict[str, Any] | None,
    config: PoseFusionConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a soft alpha prior for hands, wrists, and forearms."""

    config = config or PoseFusionConfig()
    height, width = shape
    prior = np.zeros((height, width), dtype=np.float32)
    details: dict[str, Any] = {"used_sides": [], "radii": {}}
    pose = _as_pose(instance) if instance is not None else None
    if pose is None:
        return prior, details
    keypoints, scores = pose

    sides = (
        ("left", LEFT_ELBOW, LEFT_WRIST, LEFT_HAND),
        ("right", RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HAND),
    )
    for side, elbow_index, wrist_index, hand_indices in sides:
        elbow_score = float(scores[elbow_index])
        wrist_score = float(scores[wrist_index])
        if min(elbow_score, wrist_score) < config.keypoint_threshold:
            continue

        elbow = np.asarray(keypoints[elbow_index], dtype=np.float32)
        wrist = np.asarray(keypoints[wrist_index], dtype=np.float32)
        forearm_length = float(np.linalg.norm(wrist - elbow))
        radius = int(
            np.clip(
                round(forearm_length * config.forearm_radius_ratio),
                config.min_radius,
                config.max_radius,
            )
        )

        forearm_mask = np.zeros_like(prior, dtype=np.uint8)
        cv2.line(
            forearm_mask,
            _point(keypoints, elbow_index),
            _point(keypoints, wrist_index),
            1,
            thickness=radius * 2,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            forearm_mask,
            _point(keypoints, wrist_index),
            max(radius, 1),
            1,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        _draw_soft_region(prior, forearm_mask, min(elbow_score, wrist_score))

        valid_hand_indices = [
            index
            for index in hand_indices
            if float(scores[index]) >= config.keypoint_threshold
        ]
        if len(valid_hand_indices) >= config.min_hand_keypoints:
            hand_points = np.vstack(
                [keypoints[valid_hand_indices], keypoints[wrist_index][None]]
            )
            hand_points[:, 0] = np.clip(hand_points[:, 0], 0, width - 1)
            hand_points[:, 1] = np.clip(hand_points[:, 1], 0, height - 1)
            hull = cv2.convexHull(np.round(hand_points).astype(np.int32))
            hand_mask = np.zeros_like(prior, dtype=np.uint8)
            cv2.fillConvexPoly(hand_mask, hull, 1, lineType=cv2.LINE_AA)
            padding = max(int(round(radius * config.hand_padding_ratio)), 1)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1)
            )
            hand_mask = cv2.dilate(hand_mask, kernel)
            hand_confidence = float(
                np.mean(scores[np.asarray(valid_hand_indices, dtype=np.int32)])
            )
            _draw_soft_region(
                prior,
                hand_mask,
                min(wrist_score, hand_confidence),
            )

        details["used_sides"].append(side)
        details["radii"][side] = radius

    if config.feather_radius > 0 and np.any(prior):
        kernel_size = config.feather_radius * 2 + 1
        prior = cv2.GaussianBlur(
            prior,
            (kernel_size, kernel_size),
            sigmaX=max(config.feather_radius / 2.0, 0.5),
        )
    prior = np.clip(prior * config.target_alpha, 0.0, 1.0)
    return prior, details


def fuse_alpha_with_pose(
    alpha: np.ndarray,
    frame: dict[str, Any] | None,
    config: PoseFusionConfig | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    if alpha.ndim != 2:
        raise ValueError("alpha must be a single-channel image")
    config = config or PoseFusionConfig()
    instance = select_instance(frame, alpha)
    prior, details = build_pose_prior(alpha.shape, instance, config)
    alpha_float = alpha.astype(np.float32) / 255.0
    fused = np.maximum(alpha_float, prior)
    fused_u8 = np.round(np.clip(fused, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not return_diagnostics:
        return fused_u8
    diagnostics = {
        "pose_prior": prior,
        "alpha_added": np.clip(fused - alpha_float, 0.0, 1.0),
        "selected_bbox": instance.get("bbox") if instance is not None else None,
        **details,
    }
    return fused_u8, diagnostics


def _alpha_stem(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_alpha")] if stem.endswith("_alpha") else stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse RVM alpha with Sapiens2 pose JSON")
    parser.add_argument("--alpha-dir", required=True)
    parser.add_argument("--pose-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--target-alpha", type=float, default=0.98)
    parser.add_argument("--save-diagnostics", action="store_true")
    args = parser.parse_args()

    config = PoseFusionConfig(
        keypoint_threshold=args.keypoint_threshold,
        target_alpha=args.target_alpha,
    )
    sequence = PoseSequence.from_json(args.pose_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output_dir / "pose_diagnostics"
    if args.save_diagnostics:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    all_png_paths = sorted(Path(args.alpha_dir).glob("*.png"))
    named_alpha_paths = [path for path in all_png_paths if path.stem.endswith("_alpha")]
    alpha_paths = named_alpha_paths or all_png_paths
    processed = 0
    matched = 0
    shape_validated = False
    for alpha_path in alpha_paths:
        alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            continue
        if not shape_validated:
            sequence.validate_alpha_shape(alpha.shape)
            shape_validated = True
        stem = _alpha_stem(alpha_path)
        frame = sequence.frame(stem)
        matched += int(frame is not None)
        if args.save_diagnostics:
            fused, diagnostics = fuse_alpha_with_pose(
                alpha, frame, config, return_diagnostics=True
            )
            cv2.imwrite(
                str(diagnostic_dir / f"{stem}_pose_prior.png"),
                np.round(diagnostics["pose_prior"] * 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(diagnostic_dir / f"{stem}_alpha_added.png"),
                np.round(diagnostics["alpha_added"] * 255).astype(np.uint8),
            )
        else:
            fused = fuse_alpha_with_pose(alpha, frame, config)
        cv2.imwrite(str(output_dir / f"{stem}_alpha.png"), fused)
        processed += 1
    print(
        f"[PoseFusion] processed {processed} alpha frames; "
        f"pose matched {matched} frames -> {output_dir}"
    )


if __name__ == "__main__":
    main()
