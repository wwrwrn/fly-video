"""升级新增规则的测试：缩放适配、净爬升指标、过线后停止、标定纠偏、人工点选和批量汇总。"""

import unittest

import numpy as np

from app_server import REPEAT_SUFFIX, result_dir
from batch_single_fly import parse_group, summarize
from calibrate_manual import click_problem
from single_fly_geometry import correct_columns, geometry_from_clicks
from single_fly_tracker import SingleFlyTracker


def box(cx, cy, confidence=0.7):
    return {"x1": cx - 5, "y1": cy - 6, "x2": cx + 5, "y2": cy + 6, "confidence": confidence}


def feed(tracker, positions, fps=30):
    for index, y in enumerate(positions):
        rows = [] if y is None else [box(100, y)]
        tracker.update(index, index / fps, rows, (0, 0), (0, 0), True)


class TrackerTests(unittest.TestCase):
    def test_net_speed_excludes_waiting_before_onset(self):
        tracker = SingleFlyTracker(1, bottom_y=505, finish_line=lambda x: 100)
        feed(tracker, [500] * 10 + [500 - 8 * k for k in range(1, 55)])
        result = tracker.result("ok")
        self.assertEqual(result["first_crossing_frame"], 60)
        self.assertAlmostEqual(result["candidate_speed_cm_s"], 5.5 / 2.0)
        self.assertAlmostEqual(result["onset_time_sec"], 12 / 30)
        self.assertAlmostEqual(result["net_distance_cm"], 400 / 76)
        self.assertAlmostEqual(result["net_climb_speed_cm_s"], (400 / 76) / 1.6)

    def test_no_net_speed_without_crossing(self):
        tracker = SingleFlyTracker(1, bottom_y=505, finish_line=lambda x: 100)
        feed(tracker, [500 - 2 * k for k in range(60)])
        result = tracker.result("ok")
        self.assertEqual(result["outcome"], "not_reached")
        self.assertIsNone(result["net_climb_speed_cm_s"])

    def test_tracking_stops_after_crossing(self):
        tracker = SingleFlyTracker(1, bottom_y=505, finish_line=lambda x: 100)
        feed(tracker, [500 - 8 * k for k in range(60)])
        points = len(tracker.points)
        self.assertIsNotNone(tracker.crossing)
        update = tracker.update(60, 2.0, [box(100, 500)], (0, 0), (0, 0), True)
        self.assertEqual(update["status"], "finished")
        self.assertEqual(len(tracker.points), points)

    def test_gap_restarts_pending_crossing_from_next_frame(self):
        tracker = SingleFlyTracker(1, bottom_y=140, finish_line=lambda x: 100)
        feed(tracker, [130, 120, 110, 96, 94, None, 92, 90, 88])
        self.assertEqual(tracker.crossing["index"], 6)
        self.assertIn("crossing_bracket_has_missing_frames", tracker.flags)

    def test_association_gate_scales_with_magnification(self):
        for pixels_per_cm, expected in ((76, "selected"), (38, "missing")):
            tracker = SingleFlyTracker(1, bottom_y=305, finish_line=lambda x: 100, pixels_per_cm=pixels_per_cm)
            tracker.update(0, 0, [box(100, 300)], (0, 0), (0, 0), True)
            selected, reason = tracker.choose([box(100, 280)], 1, (0, 0), (0, 0))
            self.assertEqual(reason, expected)
            self.assertEqual(selected is not None, expected == "selected")


    def test_lost_fly_is_relocked_after_long_gap(self):
        tracker = SingleFlyTracker(1, bottom_y=505, finish_line=lambda x: 100)
        feed(tracker, [500, 498] + [None] * 40 + [200, 192, 184])
        self.assertEqual(len(tracker.points), 5)
        self.assertIn("reacquired_after_gap", tracker.flags)
        self.assertEqual(tracker.result("ok")["speed_quality"], "review")

    def test_weak_far_detection_is_not_relocked(self):
        tracker = SingleFlyTracker(1, bottom_y=505, finish_line=lambda x: 100)
        feed(tracker, [500, 498] + [None] * 40)
        update = tracker.update(42, 1.4, [box(100, 200, 0.4)], (0, 0), (0, 0), True)
        self.assertIsNone(update["selected"])

    def test_long_gap_makes_not_reached_reviewable(self):
        tracker = SingleFlyTracker(1, bottom_y=505, finish_line=lambda x: 100)
        feed(tracker, [500, 498] + [None] * 40)
        result = tracker.result("ok")
        self.assertEqual(result["outcome"], "not_reached")
        self.assertEqual(result["speed_quality"], "review")
        self.assertIn("long_tracking_gap", result["quality_flags"])


class GeometryTests(unittest.TestCase):
    def board(self, tube_centers):
        gray = np.full((1080, 1920), 200, np.uint8)
        for center in tube_centers:
            gray[:, center - 80:center + 80:6] = 120
        return gray, np.ones(gray.shape, bool)

    def test_shifted_comb_moves_back_onto_tubes(self):
        tubes = np.arange(6) * 200 + 400
        gray, board = self.board(tubes)
        moved = correct_columns(gray, board, np.array([0.0, 300.0]), tubes + 200.0, 200.0, 84.0)
        np.testing.assert_allclose(moved, tubes)

    def test_comb_without_six_textured_columns_is_rejected(self):
        tubes = np.arange(5) * 200 + 400
        gray, board = self.board(tubes)
        centers = np.arange(6) * 200.0 + 400
        self.assertIsNone(correct_columns(gray, board, np.array([0.0, 300.0]), centers, 200.0, 84.0))

    def test_manual_clicks_match_automatic_structure(self):
        bottoms = [(300 + 200 * i, 700) for i in range(6)]
        geometry = geometry_from_clicks([(100, 200), (1800, 220)], bottoms, (1920, 1080), 30)
        self.assertEqual(len(geometry["tubes"]), 6)
        self.assertEqual(geometry["tube_pitch_pixels"], 200)
        self.assertAlmostEqual(geometry["finish_line"]["slope"], 20 / 1700)
        self.assertAlmostEqual(geometry["pixels_per_cm"], (700 - (200 + 20 / 1700 * 700)) / 5.5, delta=1)

    def test_manual_clicks_reject_bad_input(self):
        ordered = [(300 + 200 * i, 700) for i in range(6)]
        with self.assertRaises(ValueError):
            geometry_from_clicks([(100, 200), (1800, 220)], ordered[::-1], (1920, 1080), 30)
        with self.assertRaises(ValueError):
            geometry_from_clicks([(100, 800), (1800, 820)], ordered, (1920, 1080), 30)


    def test_clicks_on_flies_are_refused_immediately(self):
        line = [(150, 435), (1530, 395)]
        size = (1920, 1080)
        self.assertIsNone(click_problem(line[:1], line[1], size))
        self.assertIsNotNone(click_problem([(690, 640)], (1540, 290), size))
        self.assertIsNotNone(click_problem(line, (900, 300), size))
        self.assertIsNotNone(click_problem(line + [(400, 780)], (300, 780), size))
        self.assertIsNotNone(click_problem(line + [(400, 780)], (620, 500), size))
        self.assertIsNone(click_problem(line + [(400, 780)], (620, 770), size))

    def test_line_clicked_beside_tube_bottoms_is_refused(self):
        size = (1920, 1080)
        self.assertIsNotNone(click_problem([(430, 750)], (550, 743), size))
        self.assertIsNotNone(click_problem([(100, 720), (1600, 680)], (500, 750), size))
        with self.assertRaises(ValueError):
            geometry_from_clicks([(430, 750), (1500, 690)], [(500 + 180 * i, 760) for i in range(6)], size, 30)


class AppTests(unittest.TestCase):
    def test_result_paths_stay_inside_outputs(self):
        for run in ("../config", "../../Windows", "no_such_run"):
            with self.assertRaises(ValueError):
                result_dir(run)

    def test_repeat_runs_share_one_video_key(self):
        self.assertEqual(REPEAT_SUFFIX.sub("", "20261009_190337_retry2"), "20261009_190337")
        self.assertEqual(REPEAT_SUFFIX.sub("", "20261009_190337_run1"), "20261009_190337")
        self.assertEqual(REPEAT_SUFFIX.sub("", "20261009_190337"), "20261009_190337")


class BatchTests(unittest.TestCase):
    def test_parse_group(self):
        self.assertEqual(parse_group("14541 雄"), ("14541", "雄"))
        self.assertEqual(parse_group("九月旧视频"), ("九月旧视频", ""))

    def test_summary_excludes_review_from_speed(self):
        row = {"genotype": "W118", "sex": "雌", "outcome": "first_crossing_recorded", "net_climb_speed_cm_s": 2.0}
        rows = [{**row, "candidate_speed_cm_s": 1.0, "speed_quality": "auto_candidate"},
                {**row, "candidate_speed_cm_s": 3.0, "speed_quality": "auto_candidate"},
                {**row, "candidate_speed_cm_s": 9.0, "speed_quality": "review"}]
        entry = summarize(rows)[0]
        self.assertEqual((entry["tubes"], entry["crossed"], entry["review"]), (3, 3, 1))
        self.assertEqual(entry["candidate_speed_cm_s_n"], 2)
        self.assertEqual(entry["candidate_speed_cm_s_median"], 2.0)


if __name__ == "__main__":
    unittest.main()
