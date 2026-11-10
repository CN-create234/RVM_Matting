import numpy as np
import pytest

from evaluate_pose_prior import MetricAccumulator, calculate_metrics


def test_perfect_prediction_metrics():
    target = np.array([[0, 1], [1, 0]], dtype=np.float32)
    metrics = calculate_metrics(target, target)
    assert metrics["sad"] == 0
    assert metrics["mse"] == 0
    assert metrics["iou"] == 1
    assert metrics["recall"] == 1


def test_recovered_hand_improves_recall_and_error():
    target = np.zeros((10, 10), dtype=np.float32)
    target[2:8, 4:7] = 1
    baseline = target.copy()
    baseline[6:8, 4:7] = 0
    fused = target.copy()
    roi = np.zeros_like(target, dtype=bool)
    roi[5:9, 3:8] = True
    baseline_metrics = calculate_metrics(baseline, target, roi)
    fused_metrics = calculate_metrics(fused, target, roi)
    assert fused_metrics["recall"] > baseline_metrics["recall"]
    assert fused_metrics["mae"] < baseline_metrics["mae"]


def test_accumulator_matches_single_pass():
    target = np.array([[0, 1], [1, 0]], dtype=np.float32)
    prediction = np.array([[0.2, 0.8], [0.4, 0.1]], dtype=np.float32)
    mask = np.ones_like(target, dtype=bool)
    accumulator = MetricAccumulator()
    accumulator.update(prediction[:1], target[:1], mask[:1])
    accumulator.update(prediction[1:], target[1:], mask[1:])
    accumulated = accumulator.result()
    single_pass = calculate_metrics(prediction, target, mask)
    for key in accumulated:
        if isinstance(accumulated[key], float):
            assert accumulated[key] == pytest.approx(single_pass[key])
        else:
            assert accumulated[key] == single_pass[key]
