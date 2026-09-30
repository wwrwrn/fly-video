"""关键实验规则测试：六管标定、背景终点与管内白环区分、固定身份及首次过线计时。"""

import unittest
from pathlib import Path

import cv2
import numpy as np

from single_fly_geometry import auto_calibrate, propose_geometry, TranslationRegistration
from single_fly_tracker import SingleFlyTracker


def box(cx, cy, confidence=0.7):
    return {"x1": cx - 5, "y1": cy - 6, "x2": cx + 5, "y2": cy + 6, "confidence": confidence}


class SingleFlyTests(unittest.TestCase):
    def test_current_video_auto_geometry_and_registration(self):
        for stem in ["20260921_193819", "20260921_193846"]:
            reference, geometry = auto_calibrate(Path("video") / f"{stem}.mp4")
            self.assertEqual(len(geometry["tubes"]), 6)
            self.assertLess(geometry["finish_line"]["fit_rmse_pixels"], 6)
            centers = [item["axis_x_reference"] for item in geometry["tubes"].values()]
            self.assertTrue(all(a < b for a, b in zip(centers, centers[1:])))
            self.assertTrue(all(490 < item["finish_y_at_axis"] < 520 for item in geometry["tubes"].values()))
            self.assertTrue(all(850 < item["bottom_y_reference"] < 960 for item in geometry["tubes"].values()))
            registration = TranslationRegistration(reference)
            dx, dy, score, valid = registration.update(reference)
            self.assertTrue(valid)
            self.assertAlmostEqual(dx, 0, delta=1)
            self.assertAlmostEqual(dy, 0, delta=1)

    def test_reject_missing_six_tube_layout(self):
        with self.assertRaises(ValueError):
            propose_geometry(np.full((1080, 1920, 3), 180, dtype=np.uint8))

    def test_fixed_id_and_first_crossing_time(self):
        tracker = SingleFlyTracker(1, bottom_y=500, finish_line=lambda x: 100)
        positions = [495 - 8 * i for i in range(55)]
        events = []
        for index, y in enumerate(positions):
            status = tracker.update(index, index / 30, [box(20, y)], (0, 0), (0, 0), True)
            if status["event"]:
                events.append(status["event"])
        self.assertEqual(tracker.track_id, "T1-F001")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["index"], 50)
        self.assertEqual(tracker.result("ok")["candidate_speed_cm_s"], 5.5 / (50 / 30))
        self.assertEqual(tracker.update(55, 55 / 30, [box(20, 20)], (0, 0), (0, 0), True)["status"], "finished")

    def test_no_interpolated_observation_during_gap(self):
        tracker = SingleFlyTracker(3, 500, lambda x: 100)
        tracker.update(0, 0, [box(10, 490)], (0, 0), (0, 0), True)
        tracker.update(1, 1 / 30, [box(12, 470)], (0, 0), (0, 0), True)
        tracker.update(2, 2 / 30, [], (0, 0), (0, 0), True)
        tracker.update(3, 3 / 30, [], (0, 0), (0, 0), True)
        self.assertEqual(len(tracker.points), 2)
        self.assertEqual(tracker.missing_frames, 2)
        self.assertIsNone(tracker.result("ok")["candidate_speed_cm_s"])

    def test_crossing_through_small_margin_band(self):
        tracker = SingleFlyTracker(4, 500, lambda x: 100)
        events = []
        for frame, y in enumerate([110, 105, 102, 99, 96, 94, 90]):
            result = tracker.update(frame, frame / 30, [box(20, y)], (0, 0), (0, 0), True)
            if result["event"]:
                events.append(result["event"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["index"], 4)

    def test_nonzero_roi_origin_keeps_same_target(self):
        tracker = SingleFlyTracker(1, 900, lambda x: 500)
        first = tracker.update(0, 0, [box(106, 480)], (250, 411), (0, 0), True)
        second = tracker.update(1, 1 / 30, [box(108, 478)], (250, 411), (0, 0), True)
        self.assertEqual(first["status"], "observed")
        self.assertEqual(second["status"], "observed")
        self.assertAlmostEqual(second["selected"]["cx"], 358)
        self.assertAlmostEqual(second["selected"]["cy"], 889)

    def test_ambiguous_cannot_force_identity(self):
        tracker = SingleFlyTracker(2, 500, lambda x: 100)
        state = tracker.update(0, 0, [box(20, 490, 0.7), box(80, 490, 0.68)], (0, 0), (0, 0), True)
        self.assertEqual(state["status"], "not_initialized")
        self.assertEqual(tracker.ambiguous_frames, 1)
        self.assertEqual(tracker.track_id, "T2-F001")
        self.assertEqual(len(tracker.points), 0)

    def test_above_bottom_is_reviewed(self):
        tracker = SingleFlyTracker(6, 500, lambda x: 100)
        tracker.update(0, 0, [box(20, 420)], (0, 0), (0, 0), True)
        for frame, y in [(1, 105), (2, 90), (3, 80)]:
            tracker.update(frame, frame / 30, [box(20, y)], (0, 0), (0, 0), True)
        report = tracker.result("ok")
        self.assertEqual(report["speed_quality"], "review")
        self.assertIn("initial_position_above_bottom", report["quality_flags"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
