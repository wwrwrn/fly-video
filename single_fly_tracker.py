"""每管固定一个身份的单目标关联与首次过线状态机；不把缺失帧伪造成轨迹。"""

import math


class SingleFlyTracker:
    def __init__(self, tube_number, bottom_y, finish_line):
        self.tube_id = f"tube_{tube_number}"
        self.track_id = f"T{tube_number}-F001"
        self.bottom_y = float(bottom_y)
        self.finish_line = finish_line
        self.last = None
        self.points = []
        self.status = "not_initialized"
        self.initial_above_bottom = None
        self.ambiguous_frames = 0
        self.missing_frames = 0
        self.reacquisition_count = 0
        self.crossing = None
        self.pending_crossing = None
        self.seen_below_line = False
        self.flags = set()

    @staticmethod
    def xyxy(row):
        return [float(row[key]) for key in ("x1", "y1", "x2", "y2")]

    def choose(self, rows, index, roi_origin, motion):
        candidates = []
        for row in rows:
            x1, y1, x2, y2 = self.xyxy(row)
            width, height = x2 - x1, y2 - y1
            if not (3 <= width <= 70 and 3 <= height <= 70 and width * height <= 2500):
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            score = float(row["confidence"])
            if not 0.1 <= score <= 1:
                continue
            item = {**row, "cx": cx, "cy": cy, "confidence": score}
            if self.last is None:
                if score >= 0.3:
                    candidates.append((score, item))
            else:
                gap = index - self.last["frame_idx"] - 1
                gate = min(110, max(28, 16 * (gap + 1)))
                reference_x = cx + roi_origin[0] - motion[0]
                reference_y = cy + roi_origin[1] - motion[1]
                distance = math.hypot(reference_x - self.last["cx"], reference_y - self.last["cy"])
                if distance <= gate:
                    candidates.append((score + 0.45 * (1 - distance / gate), item))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        if not candidates:
            return None, "missing"
        selected = candidates[0][1]
        if len(candidates) > 1:
            runner_up = candidates[1][1]
            separation = math.hypot(selected["cx"] - runner_up["cx"], selected["cy"] - runner_up["cy"])
            if candidates[0][0] - candidates[1][0] < 0.12 and separation > 14:
                return None, "ambiguous"
        return selected, "selected"

    def update(self, index, time_sec, rows, roi_origin, motion, geometry_valid):
        """返回本帧观测与事件。只有实际检测支持的观测才能写入轨迹。"""
        if self.crossing is not None:
            return {"status": "finished", "selected": None, "event": None, "candidate_count": len(rows)}
        selected, reason = self.choose(rows, index, roi_origin, motion)
        if selected is None:
            if reason == "ambiguous":
                self.ambiguous_frames += 1
                self.flags.add("ambiguous_detection")
            self.missing_frames += 1
            if self.last is not None:
                self.status = "missing" if reason == "missing" else "ambiguous"
            self.pending_crossing = None
            return {"status": self.status, "selected": None, "event": None, "candidate_count": len(rows)}
        dx, dy = motion
        ox, oy = roi_origin
        current = {"frame_idx": index, "time_sec": time_sec,
                   "cx": selected["cx"] + ox - dx, "cy": selected["cy"] + oy - dy,
                   "global_x": selected["cx"] + ox, "global_y": selected["cy"] + oy,
                   "confidence": selected["confidence"], "box": self.xyxy(selected),
                   "geometry_valid": bool(geometry_valid)}
        if self.last is not None and index - self.last["frame_idx"] > 1:
            self.reacquisition_count += 1
            self.flags.add("reacquired_after_gap")
        line = float(self.finish_line(current["cx"]))
        current["signed_line_distance"] = current["cy"] - line
        if current["signed_line_distance"] >= 3:
            self.seen_below_line = True
        if self.last is None:
            # 管底是图像估计位置；误差较大时不给出可靠的完整5.5cm速率。
            self.initial_above_bottom = self.bottom_y - current["cy"] > max(35, 0.12 * (self.bottom_y - line))
            if self.initial_above_bottom:
                self.flags.add("initial_position_above_bottom")
            if index > 0:
                self.flags.add("first_observation_after_frame0")
            if current["signed_line_distance"] <= -3:
                self.flags.add("already_above_finish_at_first_observation")
        previous = self.last
        self.points.append(current)
        self.last = current
        self.status = "observed"
        event = None
        if "already_above_finish_at_first_observation" not in self.flags:
            if self.pending_crossing is None:
                if previous and self.seen_below_line and current["signed_line_distance"] <= -3:
                    self.pending_crossing = {"index": index, "time_sec": time_sec,
                                             "gap": index - previous["frame_idx"] - 1,
                                             "checks": 1, "registered": bool(geometry_valid)}
            elif current["signed_line_distance"] <= -3 and previous and index == previous["frame_idx"] + 1:
                self.pending_crossing["checks"] += 1
                self.pending_crossing["registered"] &= bool(geometry_valid)
                if self.pending_crossing["checks"] >= 3:
                    self.crossing = self.pending_crossing.copy()
                    self.status = "finished"
                    event = self.crossing.copy()
                    if event["gap"]:
                        self.flags.add("crossing_bracket_has_missing_frames")
                    if not event["registered"]:
                        self.flags.add("crossing_geometry_uncertain")
            elif current["signed_line_distance"] > -3 or not previous or index > previous["frame_idx"] + 1:
                self.pending_crossing = None
        return {"status": self.status, "selected": current, "event": event, "candidate_count": len(rows)}

    def result(self, reference_quality):
        if self.crossing is None:
            outcome = "already_above_line" if "already_above_finish_at_first_observation" in self.flags else "not_reached"
            crossing_frame = crossing_time = speed = None
        else:
            crossing_frame, crossing_time = self.crossing["index"], self.crossing["time_sec"]
            speed = 5.5 / crossing_time if crossing_time > 0 else None
            outcome = "first_crossing_recorded"
        quality = "review" if self.flags or reference_quality != "ok" else "auto_candidate"
        if reference_quality != "ok":
            self.flags.add("finish_line_low_contrast_review")
        if self.reacquisition_count:
            self.flags.add("identity_reacquisition_review")
            quality = "review"
        return {"tube_id": self.tube_id, "track_id": self.track_id, "outcome": outcome,
                "first_crossing_frame": crossing_frame, "first_crossing_time_sec": crossing_time,
                "candidate_speed_cm_s": speed, "speed_quality": quality,
                "distance_cm": 5.5, "start_time_sec": 0.0,
                "initial_observation_frame": self.points[0]["frame_idx"] if self.points else None,
                "initial_y_reference": self.points[0]["cy"] if self.points else None,
                "bottom_y_reference": self.bottom_y,
                "observed_frames_before_finish": len(self.points), "missing_frames": self.missing_frames,
                "ambiguous_frames": self.ambiguous_frames, "reacquisition_count": self.reacquisition_count,
                "quality_flags": sorted(self.flags)}
