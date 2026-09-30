"""每根试管独立的ByteTrack适配：只隔离ID计数并审计生命周期，不改变官方关联算法。"""

import os
from pathlib import Path
from types import SimpleNamespace

_settings_dir = Path(__file__).resolve().parent / ".ultralytics"
_settings_dir.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_settings_dir))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker, STrack


class TubeByteTrack:
    """六管分别创建本类；tracker池、均值/协方差、ID计数和审计记录互不共享。"""

    def __init__(self, tube_number, parameters):
        self.tube_number = tube_number
        self.tube_id = f"tube_{tube_number}"
        self.next_number = 0
        self.source_frame = -1
        self.records = {}
        self.events = []
        owner = self

        class LocalSTrack(STrack):
            @staticmethod
            def next_id():
                owner.next_number += 1
                return owner.next_number

            def activate(self, kalman_filter, frame_id):
                super().activate(kalman_filter, frame_id)
                owner.observe(self, self.xyxy, "created")

            def update(self, new_track, frame_id):
                super().update(new_track, frame_id)
                owner.observe(self, new_track.xyxy, "matched")

            def re_activate(self, new_track, frame_id, new_id=False):
                super().re_activate(new_track, frame_id, new_id=new_id)
                owner.observe(self, new_track.xyxy, "reactivated")

        class LocalBYTETracker(BYTETracker):
            track_class = LocalSTrack

            def reset_id(self):
                # 官方默认计数器是全局变量；只替换分配方式，不改匹配、预测、确认或删除规则。
                owner.next_number = 0

        self.tracker = LocalBYTETracker(SimpleNamespace(**dict(parameters)))

    def name(self, native_id):
        return f"T{self.tube_number}-F{int(native_id):03d}"

    def observe(self, track, detection_box, action):
        """激活时即记录ID，连从未确认的失败轨迹也保留；低分匹配属于真实检测观测。"""
        native_id = int(track.track_id)
        if native_id not in self.records:
            self.records[native_id] = {"track_id": self.name(native_id), "object": track,
                                       "observations": [], "output_frames": [], "states": {},
                                       "last_state": None, "terminal_frame": None}
        record = self.records[native_id]
        observations = record["observations"]
        previous = observations[-1]["frame_idx"] if observations else None
        if previous is not None and previous >= self.source_frame:
            raise RuntimeError("同一轨迹在同一源帧被重复更新。")
        observation = {"frame_idx": self.source_frame, "confidence": float(track.score),
                       "detection_index": int(track.idx) + 1,
                       "detection_xyxy": [float(value) for value in detection_box], "action": action}
        observations.append(observation)
        if action in ("created", "reactivated"):
            self.events.append({"frame_idx": self.source_frame, "tube_id": self.tube_id,
                                "track_id": record["track_id"], "event": action,
                                "previous_state": record["last_state"] or "none", "current_state": "matched",
                                "missing_frames_before": self.source_frame - previous - 1 if previous is not None else 0})

    def update(self, detections, source_frame):
        """每个源帧都更新，空检测也必须传入；返回可见确认轨迹与状态审计。"""
        if source_frame != self.source_frame + 1:
            raise ValueError("baseline要求连续源帧，不能跳帧而不推进tracker时钟。")
        self.source_frame = source_frame
        output = self.tracker.update(detections)
        output = np.asarray(output, dtype=np.float32).reshape(-1, 8)
        emitted = {int(row[4]) for row in output}
        active = {int(track.track_id) for track in self.tracker.tracked_stracks}
        lost = {int(track.track_id) for track in self.tracker.lost_stracks}
        removed = {int(track.track_id) for track in self.tracker.removed_stracks}
        states = []
        for native_id, record in self.records.items():
            previous = record["last_state"]
            if previous in ("removed", "pruned_native") and native_id not in active | lost:
                continue
            track = record["object"]
            # removed_stracks是历史列表，原生算法在过期边界可能重激活其中的对象。
            # 当前活跃池和对象实际状态优先，不能把历史移除记录当作当前状态。
            if native_id in active:
                state = "tracked" if track.is_activated else "tentative"
            elif native_id in lost and track.state != 3:
                state = "lost"
            elif native_id in removed or track.state == 3:
                state = "removed"
            else:
                # 官方算法会在tracked/lost池之间剔除重叠轨迹；此处只记录，不自行删失败记录。
                state = "pruned_native"
            latest = record["observations"][-1]
            if state != previous:
                self.events.append({"frame_idx": source_frame, "tube_id": self.tube_id,
                                    "track_id": record["track_id"], "event": "state_change",
                                    "previous_state": previous or "none", "current_state": state,
                                    "missing_frames_before": source_frame - latest["frame_idx"]})
            record["last_state"] = state
            record["states"][source_frame] = state
            if state in ("removed", "pruned_native"):
                record["terminal_frame"] = source_frame
            else:
                record["terminal_frame"] = None
            if native_id in emitted:
                if latest["frame_idx"] != source_frame:
                    raise RuntimeError("tracker输出了无本帧检测支持的轨迹。")
                record["output_frames"].append(source_frame)
            xyxy = [float(value) for value in track.xyxy]
            states.append({"frame_idx": source_frame, "tube_id": self.tube_id, "track_id": record["track_id"],
                           "state": state, "emitted": int(native_id in emitted),
                           "has_detection": int(latest["frame_idx"] == source_frame),
                           "last_detection_frame": latest["frame_idx"],
                           "missing_since_last_detection": source_frame - latest["frame_idx"],
                           "last_detection_confidence": latest["confidence"],
                           "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]})
        return output, states

    def summaries(self, fps, final_frame):
        """生命周期按首次到末次检测计算，空缺另记，不含末尾lost缓冲等待时间。"""
        result = []
        for record in self.records.values():
            observed = [row["frame_idx"] for row in record["observations"]]
            gaps = [{"start_frame": a + 1, "end_frame": b - 1, "frames": b - a - 1}
                    for a, b in zip(observed, observed[1:]) if b > a + 1]
            start, end = observed[0], observed[-1]
            duration = end - start + 1
            result.append({"tube_id": self.tube_id, "track_id": record["track_id"], "start_frame": start,
                           "end_frame": end, "duration_frames": duration, "duration_sec": duration / fps,
                           "number_of_detected_frames": len(observed), "number_of_output_frames": len(record["output_frames"]),
                           "first_confirmed_output_frame": min(record["output_frames"]) if record["output_frames"] else -1,
                           "missing_frames": sum(gap["frames"] for gap in gaps), "missing_gap_count": len(gaps),
                           "max_missing_gap_frames": max((gap["frames"] for gap in gaps), default=0),
                           "missing_gaps": gaps, "final_state": record["last_state"],
                           "terminal_frame": record["terminal_frame"] if record["terminal_frame"] is not None else -1,
                           "censored_at_video_end": int(record["terminal_frame"] is None),
                           "unconfirmed_only": int(not record["output_frames"]), "short_track_le_5_frames": int(duration <= 5)})
        return result
