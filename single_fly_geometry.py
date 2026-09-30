"""固定六管拍摄布局的自动标定：白色塞子、背景深色终点线、管底参考及轻微平移补偿。"""

from pathlib import Path

import cv2
import numpy as np

from video_utils import open_video, read_frame, video_info


def odd(value):
    return max(3, int(round(value)) | 1)


def propose_geometry(frame):
    """仅适用于六根直立试管、上部白塞、浅背景深横线的当前拍摄协议。失败时不硬猜。"""
    h, w = frame.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (odd(w / 215), odd(h / 216)), 0)
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, odd(h * 0.112))))
    mask = (top > 10).astype(np.uint8) * 255
    mask[:round(h * 0.1)] = 0
    mask[round(h * 0.31):] = 0
    mask[:, :round(w * 0.05)] = 0
    mask[:, round(w * 0.9):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((odd(h / 120), odd(w / 91)), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 9), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    caps = sorted([list(map(int, s[:4])) for s in stats[1:]
                   if s[4] > w * h * 0.00024 and w * 0.057 < s[2] < w * 0.146 and s[3] > h * 0.018], key=lambda s: s[0])
    if len(caps) != 6:
        raise ValueError(f"白色塞子候选数为{len(caps)}，不是6，自动标定不通过。")
    centers = np.array([x + width / 2 for x, y, width, height in caps])
    pitches = np.diff(centers)
    pitch = float(np.median(pitches))
    if np.any(pitches < pitch * 0.7) or np.any(pitches > pitch * 1.3):
        raise ValueError("试管间距不符合固定六管布局，请人工复核。")

    black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, odd(h * 0.029))))
    profile = black[:, round(w * 0.06):round(w * 0.88)].mean(axis=1)
    lower, upper = round(h * 0.33), round(h * 0.65)
    coarse = int(np.argmax(profile[lower:upper]) + lower)
    if profile[coarse] < 2:
        raise ValueError("背景终点横线对比度不足，不能自动确认。")
    points = []
    for x in np.linspace(w * 0.08, w * 0.86, 24):
        x = int(x)
        local = black[:, x - 25:x + 25].mean(axis=1)
        lo, hi = max(0, coarse - 20), min(h, coarse + 21)
        y = int(np.argmax(local[lo:hi]) + lo)
        if local[y] >= 2.5:
            points.append((x, y))
    if len(points) < 8:
        raise ValueError("深色横线缺少足够的横向支持点。")
    points = np.asarray(points, dtype=float)
    coefficients = np.polyfit(points[:, 0], points[:, 1], 1)
    for _ in range(3):
        residual = points[:, 1] - np.polyval(coefficients, points[:, 0])
        keep = np.abs(residual - np.median(residual)) <= max(4, 2.5 * float(np.median(np.abs(residual))))
        if keep.sum() < 6:
            raise ValueError("终点横线拟合不稳定。")
        points = points[keep]
        coefficients = np.polyfit(points[:, 0], points[:, 1], 1)
    line_error = float(np.sqrt(np.mean((points[:, 1] - np.polyval(coefficients, points[:, 0])) ** 2)))
    if line_error > 6 or abs(coefficients[0]) > 0.06:
        raise ValueError("标线倾斜或拟合误差超出本地原型的适用范围。")

    gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    tubes = {}
    for i, center in enumerate(centers):
        left = int((centers[i - 1] + center) / 2 + 3) if i else int(center - pitch / 2)
        right = int((center + centers[i + 1]) / 2 - 3) if i < 5 else int(center + pitch / 2)
        roi_profile = gradient[:, int(center - pitch * 0.35):int(center + pitch * 0.35)].mean(axis=1)
        roi_profile = cv2.GaussianBlur(roi_profile[:, None], (1, 9), 0).ravel()
        lo, hi = round(h * 0.68), round(h * 0.94)
        strongest = float(np.max(roi_profile[lo:hi]))
        peaks = [y for y in range(lo + 8, hi - 8)
                 if roi_profile[y] == np.max(roi_profile[y - 8:y + 9]) and roi_profile[y] >= max(5, strongest * 0.5)]
        if not peaks:
            raise ValueError(f"tube_{i+1}管底边缘不清楚，自动标定不通过。")
        bottom = max(peaks)
        finish = float(np.polyval(coefficients, center))
        if bottom - finish < h * 0.22:
            raise ValueError("管底和终点的相对位置不合理。")
        tubes[f"tube_{i+1}"] = {"roi": [max(0, left), max(0, round(finish - h * 0.09)), min(w, right), min(h, bottom + round(h * 0.025))],
                                  "bottom_y_reference": bottom, "axis_x_reference": float(center),
                                  "finish_y_at_axis": finish, "bottom_edge_strength": strongest}
    return {"image_size": [w, h], "caps": caps, "tubes": tubes,
            "finish_line": {"slope": float(coefficients[0]), "intercept": float(coefficients[1]),
                            "fit_rmse_pixels": line_error, "support_bins": len(points),
                            "support_span_fraction": float(np.ptp(points[:, 0]) / w),
                            "timing_quality": "ok" if len(points) >= 12 and np.ptp(points[:, 0]) > w * 0.6 else "review",
                            "definition": "背景深色横线中心，不是白色环线。"},
            "calibration_method": "规则图像处理；适用于固定六管布局，不是通用标线识别模型。"}


def auto_calibrate(video):
    """在前两秒寻找曝光合适且六管完整可见的参考帧，保留失败原因供复核。"""
    failures = []
    chosen = None
    with open_video(video) as capture:
        info = video_info(capture)
        for index in sorted({min(info["frame_count"] - 1, round(info["fps"] * t)) for t in [0, 0.5, 1, 2]}):
            frame = read_frame(capture, index, info["frame_count"])
            try:
                geometry = propose_geometry(frame)
                geometry["reference_frame"] = index
                chosen = frame, geometry
            except ValueError as exc:
                failures.append({"frame": index, "reason": str(exc)})
    if chosen is None:
        raise ValueError("无法可靠自动标定，应人工设置ROI/标线：" + str(failures))
    frame, geometry = chosen
    geometry["calibration_attempt_failures"] = failures
    geometry["video"] = str(Path(video).resolve())
    return frame, geometry


class TranslationRegistration:
    """用静态背景估计小幅画面平移，目标选择在参考坐标中进行，避免把相机抖动当作果蝇跳动。"""

    def __init__(self, reference):
        self.factor = 0.25
        self.reference = self.prepare(reference)
        self.shift = (0.0, 0.0)
        h, w = self.reference.shape
        self.mask = np.zeros((h, w), np.uint8)
        self.mask[round(h * 0.1):round(h * 0.94), round(w * 0.06):round(w * 0.89)] = 255
        self.points = cv2.goodFeaturesToTrack(self.reference, 200, 0.01, 6, mask=self.mask)
        if self.points is None or len(self.points) < 12:
            raise ValueError("静态背景特征不足，无法检查拍摄位移。")

    def prepare(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, None, fx=self.factor, fy=self.factor, interpolation=cv2.INTER_AREA)

    def update(self, frame):
        try:
            current = self.prepare(frame)
            forward, status, _ = cv2.calcOpticalFlowPyrLK(self.reference, current, self.points, None, winSize=(21, 21), maxLevel=3)
            backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, self.reference, forward, None, winSize=(21, 21), maxLevel=3)
            good = (status[:, 0] > 0) & (back_status[:, 0] > 0) & (np.linalg.norm(backward[:, 0] - self.points[:, 0], axis=1) < 0.8)
            if good.sum() < 12:
                return *self.shift, float(good.mean()), False
            changes = forward[:, 0] - self.points[:, 0]
            median = np.median(changes[good], axis=0)
            inliers = good & (np.linalg.norm(changes - median, axis=1) < 1.5)
            if inliers.sum() < 12:
                return *self.shift, float(inliers.mean()), False
            dx, dy = map(float, np.median(changes[inliers], axis=0) / self.factor)
            score = float(inliers.mean())
            valid = score >= 0.15 and abs(dx) < frame.shape[1] * 0.06 and abs(dy) < frame.shape[0] * 0.06
            if valid:
                self.shift = (dx, dy)
            return *self.shift, score, bool(valid)
        except cv2.error:
            return *self.shift, 0.0, False


def draw_geometry(image, geometry, dx=0, dy=0):
    result = image.copy()
    a, b = geometry["finish_line"]["slope"], geometry["finish_line"]["intercept"]
    for tube, item in geometry["tubes"].items():
        x1, y1, x2, y2 = item["roi"]
        cv2.rectangle(result, (round(x1 + dx), round(y1 + dy)), (round(x2 + dx), round(y2 + dy)), (0, 190, 0), 1)
        cv2.putText(result, tube, (round(x1 + dx + 3), round(y1 + dy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
        cv2.line(result, (round(x1 + dx), round(a * x1 + b + dy)), (round(x2 + dx), round(a * x2 + b + dy)), (0, 0, 255), 2)
        bottom = round(item["bottom_y_reference"] + dy)
        cv2.line(result, (round(x1 + dx), bottom), (round(x2 + dx), bottom), (255, 180, 0), 2)
    return result
