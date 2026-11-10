"""Evaluate baseline and pose-guided alpha mattes against binary ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from pose_guided_matting import PoseFusionConfig, PoseSequence, build_pose_prior, select_instance


SUFFIXES = ("_alpha", "_mask", "_gt", "_ground_truth")


def canonical_stem(path: Path) -> str:
    stem = path.stem
    lowered = stem.lower()
    for suffix in SUFFIXES:
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def index_images(directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    paths: list[Path] = []
    for pattern in ("*.png", "*.tif", "*.tiff", "*.exr"):
        paths.extend(directory.glob(pattern))
    result: dict[str, Path] = {}
    for path in sorted(paths):
        result.setdefault(canonical_stem(path), path)
    return result


def load_alpha(path: Path, *, binary: bool = False, threshold: float = 0.5) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    if image.ndim == 3:
        if image.shape[2] == 4 and float(np.std(image[:, :, 3])) > 1e-8:
            image = image[:, :, 3]
        else:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    original_dtype = image.dtype
    image = image.astype(np.float32)
    max_value = float(np.nanmax(image)) if image.size else 0.0
    if np.issubdtype(original_dtype, np.integer):
        dtype_max = float(np.iinfo(original_dtype).max)
        image /= max(dtype_max, 1.0)
    elif max_value > 1.0:
        image /= max_value
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0)
    return (image >= threshold).astype(np.float32) if binary else image


@dataclass
class MetricAccumulator:
    abs_sum: float = 0.0
    squared_sum: float = 0.0
    pixels: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        selected = mask.astype(bool)
        if not np.any(selected):
            return
        pred = prediction[selected]
        gt = target[selected]
        difference = pred - gt
        self.abs_sum += float(np.abs(difference).sum())
        self.squared_sum += float(np.square(difference).sum())
        self.pixels += int(pred.size)
        pred_binary = pred >= 0.5
        gt_binary = gt >= 0.5
        self.tp += int(np.logical_and(pred_binary, gt_binary).sum())
        self.fp += int(np.logical_and(pred_binary, ~gt_binary).sum())
        self.fn += int(np.logical_and(~pred_binary, gt_binary).sum())
        self.tn += int(np.logical_and(~pred_binary, ~gt_binary).sum())

    def result(self) -> dict[str, float | int | None]:
        if self.pixels == 0:
            return {"pixels": 0, "sad": None, "mae": None, "mse": None,
                    "iou": None, "precision": None, "recall": None,
                    "false_positive_pixels": 0, "false_negative_pixels": 0}
        union = self.tp + self.fp + self.fn
        return {
            "pixels": self.pixels,
            "sad": self.abs_sum / 1000.0,
            "mae": self.abs_sum / self.pixels,
            "mse": self.squared_sum / self.pixels,
            "iou": self.tp / union if union else 1.0,
            "precision": self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0,
            "recall": self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0,
            "false_positive_pixels": self.fp,
            "false_negative_pixels": self.fn,
        }


def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: prediction={prediction.shape}, target={target.shape}")
    if mask is None:
        mask = np.ones_like(target, dtype=bool)
    accumulator = MetricAccumulator()
    accumulator.update(prediction, target, mask)
    return accumulator.result()


def make_hand_roi(
    shape: tuple[int, int],
    frame: dict | None,
    baseline: np.ndarray,
    padding: int,
) -> np.ndarray:
    instance = select_instance(frame, np.round(baseline * 255).astype(np.uint8))
    prior, _ = build_pose_prior(shape, instance, PoseFusionConfig())
    roi = (prior > 0.01).astype(np.uint8)
    if padding > 0 and np.any(roi):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1,) * 2)
        roi = cv2.dilate(roi, kernel)
    return roi.astype(bool)


def _metric_delta(baseline: dict, fused: dict) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for key in ("sad", "mae", "mse", "iou", "precision", "recall",
                "false_positive_pixels", "false_negative_pixels"):
        a, b = baseline.get(key), fused.get(key)
        result[key] = None if a is None or b is None else b - a
    return result


def _to_panel(image: np.ndarray, label: str, color_map: int | None = None) -> np.ndarray:
    u8 = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    panel = cv2.applyColorMap(u8, color_map) if color_map is not None else cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(panel, (0, 0), (360, 42), (0, 0, 0), -1)
    cv2.putText(panel, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _save_montage(rows: list[np.ndarray], path: Path) -> None:
    if rows:
        preview = cv2.vconcat(rows)
        max_width = 2400
        if preview.shape[1] > max_width:
            scale = max_width / preview.shape[1]
            preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), preview)


def evaluate(
    baseline_dir: str | Path,
    fused_dir: str | Path,
    ground_truth_dir: str | Path,
    pose_json: str | Path,
    output_dir: str | Path,
    roi_padding: int = 12,
    preview_frames: int = 8,
) -> dict:
    baseline_index = index_images(baseline_dir)
    fused_index = index_images(fused_dir)
    gt_index = index_images(ground_truth_dir)
    stems = sorted(set(baseline_index) & set(fused_index) & set(gt_index))
    if not stems:
        raise ValueError("no matching baseline, fused, and ground-truth frame stems")

    sequence = PoseSequence.from_json(pose_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulators = {
        method: {region: MetricAccumulator() for region in ("full_frame", "hand_roi")}
        for method in ("baseline", "pose_fused")
    }
    frame_rows: list[dict] = []
    montage_rows: list[np.ndarray] = []

    for stem in stems:
        baseline = load_alpha(baseline_index[stem])
        fused = load_alpha(fused_index[stem])
        target = load_alpha(gt_index[stem], binary=True)
        if baseline.shape != fused.shape or baseline.shape != target.shape:
            raise ValueError(f"frame {stem} has inconsistent shapes")
        sequence.validate_alpha_shape(baseline.shape)
        roi = make_hand_roi(baseline.shape, sequence.frame(stem), baseline, roi_padding)
        regions = {"full_frame": np.ones_like(target, dtype=bool), "hand_roi": roi}
        for method, prediction in (("baseline", baseline), ("pose_fused", fused)):
            for region, mask in regions.items():
                accumulators[method][region].update(prediction, target, mask)
                metrics = calculate_metrics(prediction, target, mask)
                frame_rows.append({"frame": stem, "method": method, "region": region, **metrics})

        if len(montage_rows) < preview_frames:
            error_gain = np.abs(baseline - target) - np.abs(fused - target)
            gain_display = np.clip(error_gain * 0.5 + 0.5, 0, 1)
            panels = [
                _to_panel(target, f"{stem} GT"),
                _to_panel(baseline, "RVM baseline"),
                _to_panel(fused, "Pose fused"),
                _to_panel(roi.astype(np.float32), "Hand ROI"),
                _to_panel(gain_display, "Error gain: red=better", cv2.COLORMAP_TURBO),
            ]
            montage_rows.append(cv2.hconcat(panels))

    aggregate = {
        method: {region: acc.result() for region, acc in regions.items()}
        for method, regions in accumulators.items()
    }
    improvements = {
        region: _metric_delta(aggregate["baseline"][region], aggregate["pose_fused"][region])
        for region in ("full_frame", "hand_roi")
    }
    payload = {
        "schema_version": 1,
        "ground_truth_type": "binary_actor_mask",
        "frames": stems,
        "frame_count": len(stems),
        "roi_definition": f"Sapiens2 hand/forearm prior dilated by {roi_padding} pixels",
        "delta_definition": "pose_fused minus baseline; negative is better for SAD/MAE/MSE/FP/FN, positive is better for IoU/precision/recall",
        "aggregate": aggregate,
        "improvement": improvements,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fieldnames = list(frame_rows[0].keys())
    with open(output_dir / "per_frame_metrics.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)
    _save_montage(montage_rows, output_dir / "comparison.png")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pose-prior alpha against UE actor masks")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--fused-dir", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--pose-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi-padding", type=int, default=12)
    parser.add_argument("--preview-frames", type=int, default=8)
    args = parser.parse_args()
    payload = evaluate(
        args.baseline_dir, args.fused_dir, args.ground_truth_dir,
        args.pose_json, args.output_dir, args.roi_padding, args.preview_frames,
    )
    print(json.dumps(payload["improvement"], indent=2))


if __name__ == "__main__":
    main()
