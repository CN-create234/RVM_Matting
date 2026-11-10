import json

import cv2
import numpy as np
import pytest

from pose_guided_matting import (
    LEFT_ELBOW,
    LEFT_HAND,
    LEFT_WRIST,
    PoseFusionConfig,
    PoseSequence,
    fuse_alpha_with_pose,
)


def _instance(score=0.95):
    keypoints = np.zeros((308, 2), dtype=np.float32)
    scores = np.zeros(308, dtype=np.float32)
    keypoints[LEFT_ELBOW] = (35, 30)
    keypoints[LEFT_WRIST] = (35, 65)
    scores[LEFT_ELBOW] = score
    scores[LEFT_WRIST] = score
    for offset, index in enumerate(LEFT_HAND):
        keypoints[index] = (30 + offset % 5 * 2, 68 + offset // 5 * 3)
        scores[index] = score
    return {
        "bbox": [20, 10, 55, 90],
        "keypoints": keypoints.tolist(),
        "keypoint_scores": scores.tolist(),
    }


def test_pose_fusion_never_reduces_existing_alpha():
    alpha = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(alpha, (35, 30), 8, 255, -1)
    frame = {"instances": [_instance()]}
    fused = fuse_alpha_with_pose(alpha, frame)
    assert np.all(fused >= alpha)


def test_pose_fusion_recovers_forearm_and_hand_region():
    alpha = np.zeros((100, 100), dtype=np.uint8)
    frame = {"instances": [_instance()]}
    fused, diagnostics = fuse_alpha_with_pose(
        alpha,
        frame,
        PoseFusionConfig(feather_radius=0),
        return_diagnostics=True,
    )
    assert fused[50, 35] > 200
    assert fused[72, 35] > 200
    assert diagnostics["used_sides"] == ["left"]


def test_low_confidence_pose_is_ignored():
    alpha = np.full((100, 100), 17, dtype=np.uint8)
    frame = {"instances": [_instance(score=0.1)]}
    fused = fuse_alpha_with_pose(alpha, frame)
    assert np.array_equal(fused, alpha)


def test_pose_sequence_matches_filename_and_stem(tmp_path):
    payload = {
        "frames": [{"image_name": "0166.png", "instances": [_instance()]}]
    }
    path = tmp_path / "poses.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    sequence = PoseSequence.from_json(path)
    assert sequence.frame("0166.png") is not None
    assert sequence.frame("0166") is not None


def test_pose_sequence_rejects_resolution_mismatch():
    sequence = PoseSequence(
        {"image_size": [1080, 1920], "frames": []}
    )
    sequence.validate_alpha_shape((1080, 1920))
    with pytest.raises(ValueError, match="resolution differ"):
        sequence.validate_alpha_shape((720, 1280))
