"""Build visual and numeric RVM vs. MatAnyone 2 comparisons for the white case."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
CASE = ROOT / "SOTA_Test_white_sequence_0160_0167"
RVM_ROOT = ROOT.parent / "Results" / "Test1228_1_lab" / "RVMResults"
MA2_ROOT = CASE / "matanyone2_output" / "input" / "pha"


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_alpha(path: Path) -> np.ndarray:
    alpha = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if alpha is None:
        raise FileNotFoundError(path)
    return alpha


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (16, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def alpha_bgr(alpha: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)


def composite(image: np.ndarray, alpha: np.ndarray, background: tuple[int, int, int]) -> np.ndarray:
    a = alpha.astype(np.float32)[..., None] / 255.0
    bg = np.empty_like(image, dtype=np.float32)
    bg[:] = background
    return np.clip(image.astype(np.float32) * a + bg * (1.0 - a), 0, 255).astype(np.uint8)


def crop_subject(image: np.ndarray, alpha_a: np.ndarray, alpha_b: np.ndarray) -> np.ndarray:
    mask = (alpha_a > 3) | (alpha_b > 3)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image
    margin = 80
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(image.shape[1], int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(image.shape[0], int(ys.max()) + margin + 1)
    return image[y0:y1, x0:x1]


def alpha_stats(alpha: np.ndarray) -> dict[str, float | int]:
    total = alpha.size
    return {
        "foreground_pixels_alpha_ge_128": int(np.count_nonzero(alpha >= 128)),
        "foreground_ratio_alpha_ge_128": float(np.count_nonzero(alpha >= 128) / total),
        "fractional_pixels_1_to_254": int(np.count_nonzero((alpha > 0) & (alpha < 255))),
        "fractional_ratio_1_to_254": float(np.count_nonzero((alpha > 0) & (alpha < 255)) / total),
        "soft_edge_pixels_16_to_239": int(np.count_nonzero((alpha >= 16) & (alpha <= 239))),
    }


def main() -> None:
    frame = read_color(CASE / "input" / "0166.png")
    rvm = read_alpha(CASE / "rvm_0166_alpha.png")
    ma2 = read_alpha(MA2_ROOT / "0006.png")

    diff = cv2.absdiff(rvm, ma2)
    diff_heat = cv2.applyColorMap(diff, cv2.COLORMAP_TURBO)
    dark = (32, 32, 32)

    panels = [
        label(frame, "Input frame 0166"),
        label(alpha_bgr(rvm), "RVM alpha"),
        label(alpha_bgr(ma2), "MatAnyone 2 alpha"),
        label(composite(frame, rvm, dark), "RVM on dark gray"),
        label(composite(frame, ma2, dark), "MatAnyone 2 on dark gray"),
        label(diff_heat, "Absolute alpha difference"),
    ]
    full = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
    cv2.imwrite(str(CASE / "comparison_full.png"), full)

    cropped = [crop_subject(panel, rvm, ma2) for panel in panels]
    target_h = max(item.shape[0] for item in cropped)
    normalized = []
    for item in cropped:
        scale = target_h / item.shape[0]
        normalized.append(cv2.resize(item, (round(item.shape[1] * scale), target_h)))
    row1 = np.hstack(normalized[:3])
    row2 = np.hstack(normalized[3:])
    width = min(row1.shape[1], row2.shape[1])
    crop_panel = np.vstack((row1[:, :width], row2[:, :width]))
    cv2.imwrite(str(CASE / "comparison_subject_crop.png"), crop_panel)

    rvm_temporal = []
    ma2_temporal = []
    for index in range(7):
        rvm_a = read_alpha(RVM_ROOT / f"{160 + index:04d}_alpha.png")
        rvm_b = read_alpha(RVM_ROOT / f"{161 + index:04d}_alpha.png")
        ma2_a = read_alpha(MA2_ROOT / f"{index:04d}.png")
        ma2_b = read_alpha(MA2_ROOT / f"{index + 1:04d}.png")
        rvm_temporal.append(float(cv2.absdiff(rvm_a, rvm_b).mean()))
        ma2_temporal.append(float(cv2.absdiff(ma2_a, ma2_b).mean()))

    metrics = {
        "case": "Test1228_1_lab frames 0160-0167; reported frame 0166",
        "note": "No ground-truth alpha is available. These are descriptive statistics, not accuracy metrics.",
        "rvm_alpha": alpha_stats(rvm),
        "matanyone2_alpha": alpha_stats(ma2),
        "mean_absolute_alpha_difference_rvm_vs_matanyone2": float(diff.mean()),
        "temporal_change_proxy_mean_abs_delta_8bit_unaligned": {
            "rvm_mean": float(np.mean(rvm_temporal)),
            "matanyone2_mean": float(np.mean(ma2_temporal)),
            "rvm_per_transition": rvm_temporal,
            "matanyone2_per_transition": ma2_temporal,
        },
    }
    (CASE / "comparison_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
