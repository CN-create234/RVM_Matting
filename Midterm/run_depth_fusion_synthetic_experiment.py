"""Controlled smoke test for the depth-guided alpha fusion.

This does not claim real-camera quality.  It verifies that registered depth can
remove background leakage and recover missing foreground while invalid depth
falls back to the RGB matte.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depth_guided_matting import (  # noqa: E402
    BackgroundDepthModel,
    DepthFusionConfig,
    fuse_alpha_with_depth,
)


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction.astype(np.float32) - target.astype(np.float32))))


def make_scene(seed: int = 7):
    rng = np.random.default_rng(seed)
    height, width = 360, 640
    ground_truth = np.zeros((height, width), np.float32)
    cv2.ellipse(ground_truth, (320, 200), (82, 130), 0, 0, 360, 1, -1)
    cv2.circle(ground_truth, (320, 74), 45, 1, -1)
    cv2.line(ground_truth, (245, 160), (160, 255), 1, 28)
    cv2.line(ground_truth, (395, 160), (480, 250), 1, 28)
    for x in range(286, 357, 10):
        cv2.line(ground_truth, (x, 44), (x + (x % 3 - 1) * 8, 15), 1, 2)

    # Simulated RGB-only matte: a halo, two missing body regions, and leakage.
    rgb_alpha = cv2.GaussianBlur(ground_truth, (0, 0), 7)
    halo = cv2.GaussianBlur(ground_truth, (0, 0), 20)
    rgb_alpha = np.maximum(rgb_alpha * 0.9, halo * 0.35)
    cv2.circle(rgb_alpha, (278, 185), 25, 0.15, -1)
    cv2.rectangle(rgb_alpha, (500, 80), (585, 155), 0.55, -1)
    rgb_alpha = np.clip(rgb_alpha, 0, 1)

    background_frames = [
        4000.0 + rng.normal(0, 6, (height, width)).astype(np.float32)
        for _ in range(8)
    ]
    depth = 4000.0 + rng.normal(0, 8, (height, width)).astype(np.float32)
    depth[ground_truth > 0.5] = (
        2250.0 + rng.normal(0, 10, int(np.sum(ground_truth > 0.5)))
    )
    boundary = cv2.morphologyEx(
        (ground_truth > 0.5).astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((7, 7), np.uint8),
    )
    missing = (boundary > 0) & (rng.random((height, width)) < 0.55)
    depth[missing] = 0
    return ground_truth, rgb_alpha, depth, background_frames


def main() -> None:
    ground_truth, rgb_alpha, depth, background_frames = make_scene()
    model = BackgroundDepthModel.from_frames(background_frames)
    fused, diagnostics = fuse_alpha_with_depth(
        rgb_alpha,
        depth,
        model,
        DepthFusionConfig(
            background_tolerance=30,
            foreground_margin=100,
            boundary_width=6,
        ),
        return_diagnostics=True,
    )

    baseline_mae = mae(rgb_alpha, ground_truth)
    fused_mae = mae(fused, ground_truth)
    if fused_mae >= baseline_mae:
        raise RuntimeError(
            f"Depth fusion did not improve synthetic MAE: {baseline_mae} -> {fused_mae}"
        )

    output_dir = ROOT / "Midterm" / "DepthFusionSyntheticExperiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [ground_truth, rgb_alpha, diagnostics["foreground_score"], fused]
    labels = ["Ground truth", "RGB matte", "Depth evidence", "Fused matte"]
    rendered = []
    for panel, label in zip(panels, labels):
        image = np.round(np.clip(panel, 0, 1) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        rendered.append(image)
    comparison = np.hstack(rendered)
    cv2.imwrite(str(output_dir / "comparison.png"), comparison)

    metrics = {
        "scope": "controlled synthetic validation; not a real-camera quality result",
        "baseline_mae": baseline_mae,
        "depth_fused_mae": fused_mae,
        "relative_mae_reduction": (baseline_mae - fused_mae) / baseline_mae,
        "invalid_depth_fallback_max_error": float(
            np.max(np.abs(fused[depth <= 0] - rgb_alpha[depth <= 0]))
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
