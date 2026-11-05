"""每管固定一个身份的单目标关联与首次过线状态机；不把缺失帧伪造成轨迹。"""

import math

# 阈值按九月视频约76像素/厘米整定，其余缩放下等比换算。
REFERENCE_PIXELS_PER_CM = 76.0
# 超过这个缺口才算身份风险；一两帧漏检在连续位置上不改变身份判断。
IDENTITY_GAP_FRAMES = 5
ONSET_RISE_CM = 0.3
# 每管只有一只：丢失约一秒后位置已不可信，高置信检测直接接回，结果标记复核。
LOST_GAP_FRAMES = 30
RELOCK_CONFIDENCE = 0.6


class SingleFlyTracker:
    def __init__(self, tube_number, bottom_y, finish_line, pixels_per_cm=REFERENCE_PIXELS_PER_CM, distance_cm=5.5):
        self.tube_id = f"tube_{tube_number}"
        self.track_id = f"T{tube_number}-F001"
        self.bottom_y = float(bottom_y)
        self.finish_line = finish_line
        self.pixels_per_cm = float(pixels_per_cm)
        self.distance_cm = float(distance_cm)
        self.scale = self.pixels_per_cm / REFERENCE_PIXELS_PER_CM
        self.margin = max(2.0, 3 * self.scale)
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
            limit = 70 * self.scale
            if not (3 <= width <= limit and 3 <= height <= limit and width * height <= 2500 * self.scale ** 2):
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
                gate = min(110, max(28, 16 * (gap + 1))) * max(self.scale, 0.6)
                reference_x = cx + roi_origin[0] - motion[0]
                reference_y = cy + roi_origin[1] - motion[1]
                distance = math.hypot(reference_x - self.last["cx"], reference_y - self.last["cy"])
                if distance <= gate:
                    candidates.append((score + 0.45 * (1 - distance / gate), item))
                elif gap >= LOST_GAP_FRAMES and score >= RELOCK_CONFIDENCE:
                    candidates.append((score, item))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        if not candidates:
            return None, "missing"
        selected = candidates[0][1]
        if len(candidates) > 1:
            runner_up = candidates[1][1]
            separation = math.hypot(selected["cx"] - runner_up["cx"], selected["cy"] - runner_up["cy"])
            if candidates[0][0] - candidates[1][0] < 0.12 and separation > 14 * self.scale:
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
        if self.last is not None and index - self.last["frame_idx"] > IDENTITY_GAP_FRAMES:
            self.reacquisition_count += 1
            self.flags.add("reacquired_after_gap")
        line = float(self.finish_line(current["cx"]))
        current["signed_line_distance"] = current["cy"] - line
        above_line = current["signed_line_distance"] <= -self.margin
        if current["signed_line_distance"] >= self.margin:
            self.seen_below_line = True
        if self.last is None:
            # 管底是图像估计位置；误差较大时不给出可靠的完整5.5cm速率。
            self.initial_above_bottom = self.bottom_y - current["cy"] > max(35 * self.scale, 0.12 * (self.bottom_y - line))
            if self.initial_above_bottom:
                self.flags.add("initial_position_above_bottom")
            if index > 0:
                self.flags.add("first_observation_after_frame0")
            if above_line:
                self.flags.add("already_above_finish_at_first_observation")
        previous = self.last
        self.points.append(current)
        self.last = current
        self.status = "observed"
        event = None
        if "already_above_finish_at_first_observation" not in self.flags:
            consecutive = previous is not None and index == previous["frame_idx"] + 1
            if self.pending_crossing is not None and not (above_line and consecutive):
                self.pending_crossing = None
            if self.pending_crossing is None:
                # 被缺口打断时本帧重新起算，不丢掉这一帧。
                if previous and self.seen_below_line and above_line:
                    self.pending_crossing = {"index": index, "time_sec": time_sec,
                                             "gap": index - previous["frame_idx"] - 1,
                                             "checks": 1, "registered": bool(geometry_valid)}
            else:
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
        return {"status": self.status, "selected": current, "event": event, "candidate_count": len(rows)}

    def climb_metrics(self):
        """净爬升：从实际起步高度到终点线，计时从首次上升超过0.3 cm开始，不含起步前的停留。"""
        empty = {"onset_time_sec": None, "net_distance_cm": None, "net_climb_speed_cm_s": None}
        if self.crossing is None or not self.points:
            return empty
        start = self.points[0]
        rise = ONSET_RISE_CM * self.pixels_per_cm
        onset = next((p for p in self.points if start["cy"] - p["cy"] >= rise), None)
        if onset is None:
            return empty
        distance = (start["cy"] - self.finish_line(start["cx"])) / self.pixels_per_cm
        duration = self.crossing["time_sec"] - onset["time_sec"]
        speed = distance / duration if duration > 0 and distance > 0 else None
        return {"onset_time_sec": onset["time_sec"], "net_distance_cm": distance, "net_climb_speed_cm_s": speed}

    def result(self, reference_quality):
        if self.crossing is None:
            outcome = "already_above_line" if "already_above_finish_at_first_observation" in self.flags else "not_reached"
            crossing_frame = crossing_time = speed = None
        else:
            crossing_frame, crossing_time = self.crossing["index"], self.crossing["time_sec"]
            speed = self.distance_cm / crossing_time if crossing_time > 0 else None
            outcome = "first_crossing_recorded"
        if self.crossing is None and self.missing_frames >= LOST_GAP_FRAMES:
            # 大段缺失时“未到达”不可靠。
            self.flags.add("long_tracking_gap")
        quality = "review" if self.flags or reference_quality != "ok" else "auto_candidate"
        if reference_quality != "ok":
            self.flags.add("finish_line_low_contrast_review")
        if self.reacquisition_count:
            self.flags.add("identity_reacquisition_review")
            quality = "review"
        return {"tube_id": self.tube_id, "track_id": self.track_id, "outcome": outcome,
                "first_crossing_frame": crossing_frame, "first_crossing_time_sec": crossing_time,
                "candidate_speed_cm_s": speed, "speed_quality": quality,
                "distance_cm": self.distance_cm, "start_time_sec": 0.0, **self.climb_metrics(),
                "initial_observation_frame": self.points[0]["frame_idx"] if self.points else None,
                "initial_y_reference": self.points[0]["cy"] if self.points else None,
                "bottom_y_reference": self.bottom_y,
                "observed_frames_before_finish": len(self.points), "missing_frames": self.missing_frames,
                "ambiguous_frames": self.ambiguous_frames, "reacquisition_count": self.reacquisition_count,
                "quality_flags": sorted(self.flags)}
