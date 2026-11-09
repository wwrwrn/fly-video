"""升级新增规则的测试：缩放适配、净爬升指标、过线后停止、标定纠偏、人工点选和批量汇总。"""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

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

    def make_run(self, root, group, name, speeds):
        folder = root / "app_runs" / group / name
        folder.mkdir(parents=True)
        results = [{"tube_id": f"tube_{i}", "outcome": "first_crossing_recorded" if speed else "not_reached",
                    "speed_quality": "auto_candidate", "first_crossing_time_sec": 1.0 if speed else None,
                    "candidate_speed_cm_s": speed, "net_climb_speed_cm_s": speed and speed * 2}
                   for i, speed in enumerate(speeds, 1)]
        (folder / "single_fly_report.json").write_text(json.dumps({"results": results}), encoding="utf-8")
        return folder

    def with_outputs(self):
        import app_server
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        outputs = root / "outputs"
        for name, value in {"OUTPUTS": outputs, "RUNS": outputs / "app_runs", "RESULT_ROOTS": (outputs / "batch_runs", outputs / "app_runs")}.items():
            patcher = patch.object(app_server, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(app_server, "send_to_recycle_bin", shutil.rmtree)
        patcher.start()
        self.addCleanup(patcher.stop)
        return app_server, outputs

    def test_video_average_skips_tubes_that_never_crossed(self):
        app_server, outputs = self.with_outputs()
        self.make_run(outputs, "A", "v1", [1.0, 2.0, None, 3.0, None, None])
        video = app_server.overview()["videos"][0]
        self.assertEqual(video["crossed"], 3)
        self.assertAlmostEqual(video["mean_speed"], 2.0)
        self.assertAlmostEqual(video["mean_net_speed"], 4.0)
        self.assertTrue(video["editable"])
        self.assertIn("2.000", app_server.summary_csv())
        self.make_run(outputs, "A", "v2", [None] * 6)
        self.assertIsNone(app_server.overview()["videos"][1]["mean_speed"])

    def test_move_rename_and_delete_keep_repeat_runs_together(self):
        app_server, outputs = self.with_outputs()
        self.make_run(outputs, "未分组", "v1", [1.0] * 6)
        self.make_run(outputs, "未分组", "v1_run1", [2.0] * 6)
        self.make_run(outputs, "B", "v2", [1.0] * 6)
        self.assertEqual(app_server.move_videos(["app_runs/未分组/v1"], "B"), ["app_runs/B/v1"])
        self.assertEqual(sorted(p.name for p in (outputs / "app_runs" / "B").iterdir()), ["v1", "v1_run1", "v2"])
        self.assertFalse((outputs / "app_runs" / "未分组").exists())
        self.make_run(outputs, "C", "v1", [1.0] * 6)
        with self.assertRaises(ValueError):
            app_server.move_videos(["app_runs/C/v1"], "B")
        app_server.move_videos(app_server.group_videos("B"), "W118 雄")
        self.assertEqual(len(list((outputs / "app_runs" / "W118 雄").iterdir())), 3)
        app_server.delete_video("app_runs/W118 雄/v1")
        self.assertEqual([p.name for p in (outputs / "app_runs" / "W118 雄").iterdir()], ["v2"])
        app_server.delete_group("W118 雄")
        self.assertFalse((outputs / "app_runs" / "W118 雄").exists())

    def test_batch_results_and_bad_group_names_are_refused(self):
        app_server, outputs = self.with_outputs()
        batch = outputs / "batch_runs" / "A" / "v1"
        batch.mkdir(parents=True)
        (batch / "single_fly_report.json").write_text('{"results": []}', encoding="utf-8")
        with self.assertRaises(ValueError):
            app_server.delete_video("batch_runs/A/v1")
        for name in ("a/b", "x:y", "a" * 61):
            with self.assertRaises(ValueError):
                app_server.clean_group(name)
        self.assertEqual(app_server.clean_group("  "), "未分组")

    def test_dropped_folder_groups_by_video_folder(self):
        import app_server
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        for relative in ("W118 雄/a.mp4", "W118 雄/b.MP4", "14541 雌/c.mp4", "14541 雌/notes.txt", "d.mp4",
                         "W118 雄/a/annotated_single_fly.mp4"):
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_bytes(b"")
        found = app_server.videos_in([str(root / "W118 雄"), str(root / "14541 雌"), str(root / "d.mp4")], "手动组")
        self.assertEqual([(path.name, group) for path, group in found],
                         [("a.mp4", "W118 雄"), ("b.MP4", "W118 雄"), ("c.mp4", "14541 雌"), ("d.mp4", "手动组")])
        self.assertEqual(app_server.videos_in(["", ".", "W118 雄"], ""), [])

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
