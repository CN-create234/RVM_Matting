import unittest

import numpy as np

from depth_guided_matting import (
    BackgroundDepthModel,
    DepthFusionConfig,
    fuse_alpha_with_depth,
)


class DepthGuidedMattingTests(unittest.TestCase):
    def setUp(self):
        background_frames = [np.full((32, 48), 4000 + offset, np.float32) for offset in (-2, 0, 2)]
        self.background = BackgroundDepthModel.from_frames(background_frames)

    def test_invalid_depth_falls_back_to_original_alpha(self):
        alpha = np.linspace(0, 1, 32 * 48, dtype=np.float32).reshape(32, 48)
        depth = np.zeros((32, 48), np.float32)
        fused = fuse_alpha_with_depth(alpha, depth, self.background)
        np.testing.assert_array_equal(fused, alpha)

    def test_foreground_is_rescued_and_background_is_suppressed(self):
        alpha = np.full((32, 48), 0.4, np.float32)
        depth = np.full((32, 48), 4000, np.float32)
        depth[8:24, 16:32] = 2500
        fused = fuse_alpha_with_depth(
            alpha,
            depth,
            self.background,
            DepthFusionConfig(boundary_width=2),
        )
        self.assertGreater(float(fused[16, 24]), 0.95)
        self.assertLess(float(fused[2, 2]), 0.05)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            fuse_alpha_with_depth(
                np.zeros((16, 16), np.float32),
                np.zeros((16, 16), np.float32),
                self.background,
            )


if __name__ == "__main__":
    unittest.main()
