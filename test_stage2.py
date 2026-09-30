"""第 2 阶段验证：坐标往返、非法标签、鼠标操作分支及候选图像素一致性。"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from annotate_flies import annotate_image
from build_dataset_sample import build_dataset
from roi_utils import representative_frame
from video_utils import save_frame
from yolo_labels import read_image, read_labels, write_labels, xyxy_to_yolo, yolo_to_xyxy


class Stage2Tests(unittest.TestCase):
    def test_conversion_and_saved_roundtrip(self):
        # 使用独立计算的期望值，防止两个转换函数同时写错仍互相抵消。
        self.assertEqual(xyxy_to_yolo([10, 20, 30, 60], 100, 200), [0.2, 0.2, 0.2, 0.2])
        np.testing.assert_allclose(yolo_to_xyxy([0.2, 0.2, 0.2, 0.2], 100, 200), [10, 20, 30, 60])
        boxes = [[0, 0, 219, 597], [218, 596, 219, 597], [63, 511, 77, 528]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.txt"
            write_labels(path, boxes, 219, 597)
            restored = read_labels(path, 219, 597)
            np.testing.assert_allclose(restored, boxes, atol=1e-6)
            self.assertEqual([[round(v) for v in box] for box in restored], boxes)

    def test_invalid_labels(self):
        bad_lines = ["1 .5 .5 .1 .1", "0 .5 .5 0 .1", "0 nan .5 .1 .1",
                     "0 .99 .5 .2 .1", "0 .5 .5 -.1 .1", "0 .5 .5 .1", "0 .5 .5 inf .1"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.txt"
            for line in bad_lines:
                path.write_text(line, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_labels(path, 219, 597)
            path.write_text("", encoding="utf-8")
            self.assertEqual(read_labels(path, 219, 597), [])
        for box in [[0, 0, 0, 1], [-1, 0, 4, 5], [0, 0, 220, 600]]:
            with self.assertRaises(ValueError):
                xyxy_to_yolo(box, 219, 597)

    def drive_ui(self, events):
        callback = None
        event_iter = iter(events)

        def set_callback(window, function):
            nonlocal callback
            callback = function

        def wait(delay):
            mouse_events, key = next(event_iter)
            for event, x, y in mouse_events:
                callback(event, x, y, 0, None)
            return key

        with patch.multiple(cv2, namedWindow=lambda *a: None, setMouseCallback=set_callback,
                            imshow=lambda *a: None, waitKeyEx=wait,
                            getWindowProperty=lambda *a: 1, destroyAllWindows=lambda: None):
            return annotate_image(np.zeros((100, 100, 3), np.uint8), [], "test.png")

    def test_mouse_multiple_undo_reverse_and_scaling(self):
        # 100x100 图放大 3 倍，画布左边距 230、上边距 85。
        down, up = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP
        events = [([(down, 260, 115), (up, 320, 175)], -1),
                  ([(down, 410, 265), (up, 350, 205)], -1),
                  ([], ord("u")),
                  ([(down, 410, 265), (up, 350, 205)], 13)]
        action, boxes = self.drive_ui(events)
        self.assertEqual(action, "complete")
        self.assertEqual(boxes, [[10, 10, 30, 30], [40, 40, 60, 60]])

    def test_empty_draft_and_cancel(self):
        self.assertEqual(self.drive_ui([([], 13), ([], ord("e")), ([], 13)]), ("complete", []))
        self.assertEqual(self.drive_ui([([], ord("d"))]), ("draft", []))
        self.assertEqual(self.drive_ui([([], 27)]), ("quit", []))

    def test_independent_preview_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.png"
            label = root / "sample.txt"
            preview = root / "preview.png"
            save_frame(np.full((100, 100, 3), 100, np.uint8), image)
            label.write_text("0 0.2 0.3 0.2 0.2\n", encoding="utf-8")
            result = subprocess.run([sys.executable, "verify_yolo_labels.py", str(image), str(label),
                                     "--output", str(preview)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = read_image(preview)
            # 固定标签应得到 [10,20,30,40]；矩形底边不得跑到其他位置。
            np.testing.assert_array_equal(rendered[39, 15], [0, 255, 0])
            np.testing.assert_array_equal(rendered[41, 15], [100, 100, 100])

    def test_unconfirmed_roi_stops_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "rois.json"
            config.write_text(Path("config/tube_rois.json").read_text(encoding="utf-8"), encoding="utf-8")
            config.with_suffix(".meta.json").write_text('{"activity_region_confirmation": {"roi_sha256": "outdated"}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                build_dataset("20260817_202618.mp4", config, root / "output")
            self.assertFalse((root / "output").exists())

    def test_all_30_candidates(self):
        manifest = json.loads(Path("dataset_sample/manifest.json").read_text(encoding="utf-8"))
        images = list(Path("dataset_sample/images").glob("*.png"))
        self.assertEqual(len(images), 30)
        self.assertEqual(manifest["frames"], [0, 80, 160, 240, 320])
        frames = {index: representative_frame(manifest["source_video"], index) for index in manifest["frames"]}
        for entry in manifest["images"]:
            x1, y1, x2, y2 = entry["roi_xyxy"]
            crop = read_image(Path("dataset_sample") / entry["image"])
            np.testing.assert_array_equal(crop, frames[entry["frame_index"]][y1:y2, x1:x2])
            self.assertEqual([crop.shape[1], crop.shape[0]], entry["image_size"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
