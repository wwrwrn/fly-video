"""第3.5阶段检查：局部/全局坐标、逐帧序号、连续区间及实际输出的一致性。"""

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from analyze_detection_stability import true_runs
from detect_full_video import detect_rois


class FakeData:
    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [[1.0, 2.0, 5.0, 6.0, 0.8, 0.0]]


class FullVideoTests(unittest.TestCase):
    def test_roi_coordinate_mapping_and_frame_local_index(self):
        class FakeModel:
            def predict(self, source, **kwargs):
                assert source.shape == (8, 8, 3)
                return [SimpleNamespace(boxes=SimpleNamespace(data=FakeData()))]

        rois = {f"tube_{i}": [i * 10, 20, i * 10 + 8, 28] for i in range(1, 7)}
        result = detect_rois(FakeModel(), np.zeros((50, 80, 3), np.uint8), rois, "cpu", 0.25, 640)
        for i in range(1, 7):
            row = result[f"tube_{i}"][0]
            self.assertEqual(row["detection_index"], 1)
            self.assertEqual((row["center_x"], row["center_y"]), (3, 4))
            self.assertEqual((row["global_center_x"], row["global_center_y"]), (i * 10 + 3, 24))
            self.assertEqual((row["width"], row["height"]), (4, 4))

    def test_runs_do_not_cross_missing_frames(self):
        indices = np.array([0, 1, 2, 3, 5, 6, 8])
        runs = true_runs([True, True, True, False, True, True, True], indices, indices / 30, 30)
        self.assertEqual(len(runs), 1)
        self.assertEqual((runs[0]["start_frame"], runs[0]["end_frame"], runs[0]["frames"]), (0, 2, 3))
        self.assertAlmostEqual(runs[0]["duration_sec"], 0.1)
        self.assertEqual(true_runs([False] * 7, indices, indices / 30, 30), [])

    def test_actual_csv_and_output_video(self):
        root = Path("outputs/full_video_detection")
        performance = json.loads((root / "performance.json").read_text(encoding="utf-8"))
        stability = json.loads((root / "stability_report.json").read_text(encoding="utf-8"))
        with (root / "detection_counts.csv").open(encoding="utf-8-sig", newline="") as stream:
            counts = list(csv.DictReader(stream))
        self.assertEqual(len(counts), performance["processed_frames"])
        self.assertEqual([int(row["frame_idx"]) for row in counts], list(range(425)))
        self.assertEqual(stability["output_video_verification"]["decoded_output_frames"], 425)
        self.assertEqual((stability["output_video_verification"]["width"], stability["output_video_verification"]["height"]), (1920, 1080))
        self.assertAlmostEqual(stability["output_video_verification"]["reported_fps"], performance["source_metadata"]["fps"], places=5)
        observed = {}
        total = 0
        with (root / "full_video_detections.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                frame, tube = int(row["frame_idx"]), row["tube_id"]
                key = (frame, tube)
                observed[key] = observed.get(key, 0) + 1
                self.assertEqual(int(row["detection_index"]), observed[key])
                x1, y1, x2, y2 = [float(row[key]) for key in ["x1", "y1", "x2", "y2"]]
                ox, oy, ex, ey = performance["roi_snapshot"][tube]
                self.assertTrue(0 <= x1 < x2 <= ex - ox and 0 <= y1 < y2 <= ey - oy)
                self.assertAlmostEqual(float(row["width"]), x2 - x1, places=6)
                self.assertAlmostEqual(float(row["height"]), y2 - y1, places=6)
                self.assertAlmostEqual(float(row["global_center_x"]), (x1 + x2) / 2 + ox, places=6)
                self.assertAlmostEqual(float(row["global_center_y"]), (y1 + y2) / 2 + oy, places=6)
                self.assertAlmostEqual(float(row["time_sec"]), frame / performance["source_metadata"]["fps"], places=7)
                self.assertTrue(0.25 <= float(row["confidence"]) <= 1)
                total += 1
        self.assertEqual(total, performance["detection_rows"])
        for row in counts:
            expected_total = 0
            for tube in performance["roi_snapshot"]:
                value = observed.get((int(row["frame_idx"]), tube), 0)
                self.assertEqual(value, int(row[tube]))
                expected_total += value
            self.assertEqual(expected_total, int(row["total"]))
        self.assertLessEqual(len(list((root / "anomaly_frames").glob("*.png"))), 30)
        self.assertEqual(len(list((root / "plots").glob("*.png"))), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
