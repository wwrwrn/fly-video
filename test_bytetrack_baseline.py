"""ByteTrack接入检查：六管隔离、低分补救、丢失恢复、失败ID保留及与原算法一致性。"""

import os
from pathlib import Path
import unittest
from types import SimpleNamespace

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from tube_bytetrack import TubeByteTrack

PARAMETERS = dict(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1,
                  new_track_thresh=0.25, track_buffer=30, match_thresh=0.8, fuse_score=True)


def boxes(values=()):
    return Boxes(np.asarray(values, dtype=np.float32).reshape(-1, 6), orig_shape=(100, 100))


class ByteTrackTests(unittest.TestCase):
    def test_six_independent_namespaces_and_state(self):
        owners = [TubeByteTrack(i, PARAMETERS) for i in range(1, 7)]
        for i, owner in enumerate(owners, 1):
            output, states = owner.update(boxes([[10, 10, 20, 20, 0.9, 0]]), 0)
            self.assertEqual(owner.name(output[0, 4]), f"T{i}-F001")
        self.assertEqual(len({id(owner.tracker.tracked_stracks) for owner in owners}), 6)
        self.assertEqual(len({id(owner.tracker.kalman_filter) for owner in owners}), 6)
        self.assertEqual(len({id(owner.records[1]["object"].mean) for owner in owners}), 6)
        owners[0].update(boxes(), 1)
        self.assertEqual(owners[0].records[1]["last_state"], "lost")
        self.assertEqual(owners[1].records[1]["last_state"], "tracked")
        # 后来创建另一个tracker，不能重置先前管的编号或轨迹。
        TubeByteTrack(7, PARAMETERS)
        owners[0].update(boxes([[70, 70, 80, 80, 0.9, 0]]), 2)
        self.assertEqual(owners[0].next_number, 2)
        self.assertEqual(owners[1].next_number, 1)

    def test_low_score_rescues_existing_id(self):
        owner = TubeByteTrack(1, PARAMETERS)
        owner.update(boxes([[10, 10, 20, 20, 0.9, 0]]), 0)
        output, _ = owner.update(boxes([[10, 10, 20, 20, 0.2, 0]]), 1)
        self.assertEqual(int(output[0, 4]), 1)
        self.assertAlmostEqual(output[0, 5], 0.2)
        self.assertEqual(owner.records[1]["last_state"], "tracked")
        other = TubeByteTrack(2, PARAMETERS)
        other.update(boxes([[10, 10, 20, 20, 0.2, 0]]), 0)
        self.assertEqual(other.next_number, 0)

    def test_lost_state_and_reactivation_gap(self):
        owner = TubeByteTrack(1, PARAMETERS)
        owner.update(boxes([[10, 10, 20, 20, 0.9, 0]]), 0)
        for frame in [1, 2]:
            output, states = owner.update(boxes(), frame)
            self.assertEqual(len(output), 0)
            self.assertEqual(states[0]["state"], "lost")
            self.assertEqual(states[0]["has_detection"], 0)
        output, _ = owner.update(boxes([[10, 10, 20, 20, 0.9, 0]]), 3)
        self.assertEqual(int(output[0, 4]), 1)
        summary = owner.summaries(30, 3)[0]
        self.assertEqual(summary["duration_frames"], 4)
        self.assertEqual(summary["number_of_detected_frames"], 2)
        self.assertEqual(summary["max_missing_gap_frames"], 2)
        self.assertTrue(any(event["event"] == "reactivated" for event in owner.events))

    def test_unconfirmed_failure_is_preserved(self):
        owner = TubeByteTrack(1, PARAMETERS)
        owner.update(boxes(), 0)
        output, _ = owner.update(boxes([[10, 10, 20, 20, 0.9, 0]]), 1)
        self.assertEqual(len(output), 0)
        owner.update(boxes(), 2)
        summary = owner.summaries(30, 2)[0]
        self.assertEqual(owner.next_number, 1)
        self.assertEqual(summary["duration_frames"], 1)
        self.assertEqual(summary["number_of_output_frames"], 0)
        self.assertEqual(summary["final_state"], "removed")

    def test_buffer_expiry_does_not_restore_removed_id(self):
        owner = TubeByteTrack(1, PARAMETERS)
        owner.update(boxes([[10, 10, 20, 20, 0.9, 0]]), 0)
        for frame in range(1, 31):
            owner.update(boxes(), frame)
            self.assertEqual(owner.records[1]["last_state"], "lost")
        owner.update(boxes(), 31)
        self.assertEqual(owner.records[1]["last_state"], "removed")
        # 再推进一帧，让原生实现清理暂留在lost池中的历史removed引用。
        owner.update(boxes(), 32)
        owner.update(boxes([[10, 10, 20, 20, 0.9, 0]]), 33)
        output, _ = owner.update(boxes([[10, 10, 20, 20, 0.9, 0]]), 34)
        self.assertEqual(int(output[0, 4]), 2)
        self.assertEqual(len(owner.summaries(30, 34)), 2)

    def test_native_expiry_boundary_audit_uses_current_live_state(self):
        owner = TubeByteTrack(1, PARAMETERS)
        owner.update(boxes([[10, 10, 20, 20, 0.9, 0]]), 0)
        for frame in range(1, 32):
            owner.update(boxes(), frame)
        self.assertEqual(owner.records[1]["last_state"], "removed")
        output, states = owner.update(boxes([[10, 10, 20, 20, 0.9, 0]]), 32)
        # 忠实保留当前原生实现的边界行为，不自行修补关联算法。
        self.assertEqual(int(output[0, 4]), 1)
        self.assertEqual(states[0]["state"], "tracked")
        self.assertEqual(owner.summaries(30, 32)[0]["terminal_frame"], -1)

    def test_native_algorithm_output_is_unchanged(self):
        owner = TubeByteTrack(1, PARAMETERS)
        native = BYTETracker(SimpleNamespace(**PARAMETERS))
        sequence = [[[10, 10, 20, 20, 0.9, 0]], [[11, 10, 21, 20, 0.8, 0]],
                    [[12, 10, 22, 20, 0.2, 0]], [], [[13, 10, 23, 20, 0.9, 0]]]
        for frame, values in enumerate(sequence):
            actual, _ = owner.update(boxes(values), frame)
            expected = np.asarray(native.update(boxes(values)), dtype=np.float32).reshape(-1, 8)
            np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
