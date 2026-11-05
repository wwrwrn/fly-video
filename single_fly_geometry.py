"""固定六管拍摄布局的自动标定：白色塞子、背景深色终点线、管底参考及轻微平移补偿。"""

from itertools import combinations
import json
from pathlib import Path

import cv2
import numpy as np

from video_utils import open_video, read_frame, video_info


def odd(value):
    return max(3, int(round(value)) | 1)


def propose_geometry_caps(frame):
    """九月旧机位：白塞在画面上部且与背景可分。条件严格，通过时优先使用。"""
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


def board_mask(gray):
    """最大亮连通域即实验白板；横线只在白板内部搜索，避开桌沿、书脊和备用试管。"""
    h, w = gray.shape
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    threshold, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (small >= max(threshold, 130)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise ValueError("画面中找不到浅色实验白板。")
    board = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255
    return cv2.resize(board, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def line_candidates(gray, board, limit=4):
    """按支持点数量返回互不重叠的近水平深色线假设，真假由管距/管长比例裁决。"""
    h, w = gray.shape
    black = cv2.morphologyEx(cv2.GaussianBlur(gray, (9, 5), 0), cv2.MORPH_BLACKHAT,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, odd(h * 0.06)))).astype(np.float32)
    inner = cv2.erode(board.astype(np.uint8), np.ones((odd(h * 0.03), 1), np.uint8)) > 0
    black[~inner] = 0
    rows = np.flatnonzero(board.any(axis=1))
    # 终点线位于白板上半部分；管底必须还在其下方。
    bottom = int(rows[0] + (rows[-1] - rows[0]) * 0.7)
    remaining = black.copy()
    results = []
    columns = np.arange(int(w * 0.02), int(w * 0.98), 12)
    for _ in range(limit):
        xs, ys = [], []
        for x in columns:
            local = remaining[:bottom, x - 6:x + 6].mean(axis=1)
            y = int(np.argmax(local))
            if local[y] >= 2.5:
                xs.append(x)
                ys.append(y)
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if len(xs) < 8:
            break
        best = None
        for i in range(len(xs)):
            for slope in np.arange(-0.07, 0.0701, 0.01):
                inliers = np.abs(ys - (ys[i] + slope * (xs - xs[i]))) < 4
                if best is None or inliers.sum() > best.sum():
                    best = inliers
        coefficients = np.polyfit(xs[best], ys[best], 1)
        for _ in range(3):
            best = np.abs(ys - np.polyval(coefficients, xs)) < 4
            if best.sum() < 6:
                break
            coefficients = np.polyfit(xs[best], ys[best], 1)
        if best.sum() >= 8:
            results.append((coefficients, xs[best], ys[best]))
        center = np.polyval(coefficients, np.arange(w)).round().astype(int)
        for x, y in enumerate(center):
            remaining[max(0, y - 25):max(0, y + 26), x] = 0
    return results


def tube_gaps(frame, gray, coefficients, board):
    """横线在两管之间直接露出、颜色更深更黄；提取五个窄缺口即得六管位置，假线凑不出规则缺口。"""
    h, w = gray.shape
    xs = np.arange(w)
    ys = np.polyval(coefficients, xs).round().astype(int)
    valid = (ys >= 3) & (ys < h - 3)
    valid[valid] = board[ys[valid], xs[valid]]
    if valid.sum() < w * 0.25:
        return None
    x0, x1 = int(np.flatnonzero(valid)[0]), int(np.flatnonzero(valid)[-1]) + 1
    yellow = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 2]
    rows = ys[x0:x1, None] + np.arange(-2, 3)[None, :]
    columns = xs[x0:x1, None]
    light = gray[rows, columns].mean(axis=1).astype(np.float32)
    tint = yellow[rows, columns].mean(axis=1).astype(np.float32)
    # 横线下方的白板亮度：线的末端外侧深度为零，真缺口两侧仍是横线。
    offset = round(h * 0.03)
    depth = gray[np.clip(ys[x0:x1] + offset, 0, h - 1), columns[:, 0]].astype(np.float32) - light
    depth = cv2.GaussianBlur(depth[:, None], (1, 9), 0).ravel()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, odd(w * 0.08)))
    dark = cv2.morphologyEx(light[:, None], cv2.MORPH_BLACKHAT, kernel).ravel()
    warm = cv2.morphologyEx(tint[:, None], cv2.MORPH_TOPHAT, kernel).ravel()
    evidence = dark / max(3.0, float(np.percentile(dark, 99))) + warm / max(1.5, float(np.percentile(warm, 99)))
    evidence = cv2.GaussianBlur(evidence[:, None], (1, 5), 0).ravel()
    def flanked(gap):
        side = max(6, round(w * 0.012))
        a, b = gap[0] - x0, gap[1] - x0
        if a - side < 0 or b + side > len(depth):
            return False
        inner = float(np.median(depth[a:b]))
        return (float(np.median(depth[a - side:a])) >= 0.2 * inner
                and float(np.median(depth[b:b + side])) >= 0.2 * inner)

    def cores_lighter(gap_centers, pitch, chosen):
        """六根管的横线段都要明显亮于缺口；把线的深色末端算成一根管时这里不成立。"""
        tube_centers = [gap_centers[0] - pitch / 2, *((gap_centers[:-1] + gap_centers[1:]) / 2), gap_centers[-1] + pitch / 2]
        half = pitch * 0.2
        levels = []
        for center in tube_centers:
            a, b = int(center - half) - x0, int(center + half) - x0
            if a < 0 or b > len(light):
                return False
            levels.append(float(np.median(light[a:b])))
        gap_level = float(np.median([np.median(light[a - x0:b - x0]) for a, b, _ in chosen]))
        contrast = float(np.median(levels)) - gap_level
        return contrast >= 2.5 and min(levels) >= gap_level + 0.55 * contrast

    best = None
    for threshold in (0.9, 0.7, 0.5, 0.35):
        above = evidence > threshold
        edges = np.flatnonzero(np.diff(np.r_[0, above.astype(np.int8), 0]))
        gaps = []
        for start, stop in zip(edges[::2], edges[1::2]):
            if gaps and x0 + start - gaps[-1][1] < w * 0.008:
                gaps[-1] = (gaps[-1][0], x0 + stop, max(gaps[-1][2], float(evidence[start:stop].max())))
            else:
                gaps.append((x0 + start, x0 + stop, float(evidence[start:stop].max())))
        gaps = [gap for gap in gaps if w * 0.008 <= gap[1] - gap[0] <= w * 0.075 and flanked(gap)]
        if len(gaps) > 12:
            continue
        for chosen in combinations(gaps, 5):
            centers = np.array([(a + b) / 2 for a, b, _ in chosen])
            steps = np.diff(centers)
            pitch = float(np.median(steps))
            widths = np.array([b - a for a, b, _ in chosen], float)
            if pitch < w * 0.05 or np.max(np.abs(steps - pitch)) > pitch * 0.2:
                continue
            if np.max(np.abs(widths - np.median(widths))) > max(0.45 * np.median(widths), w * 0.008):
                continue
            if not cores_lighter(centers, pitch, chosen):
                continue
            score = sum(item[2] for item in chosen) - 8 * float(np.ptp(steps)) / pitch
            if best is None or score > best[0]:
                best = (score, centers, pitch, float(np.median([b - a for a, b, _ in chosen])))
    if best is None:
        return None
    _, gap_centers, pitch, gap_width = best
    centers = [gap_centers[0] - pitch / 2, *((gap_centers[:-1] + gap_centers[1:]) / 2), gap_centers[-1] + pitch / 2]
    return np.asarray(centers), pitch, (pitch - gap_width) / 2


def find_bottom(gradient, center, half_width, finish, pitch, h):
    """管底圆环按管距比例限定搜索范围；缩放变化时不依赖画面绝对高度。"""
    lo = max(0, round(finish + pitch * 1.35))
    hi = min(h, round(finish + pitch * 2.2))
    if hi - lo < 24:
        return None
    x1, x2 = max(0, int(center - half_width * 0.75)), min(gradient.shape[1], int(center + half_width * 0.75))
    if x2 - x1 < 8:
        return None
    profile = gradient[:, x1:x2].mean(axis=1)
    profile = cv2.GaussianBlur(profile[:, None], (1, 9), 0).ravel()
    strongest = float(np.max(profile[lo:hi]))
    peaks = [y for y in range(lo + 8, hi - 8)
             if profile[y] == np.max(profile[y - 8:y + 9]) and profile[y] >= max(3, strongest * 0.5)]
    return (max(peaks), strongest) if peaks else None


def build_tubes(gradient, coefficients, centers, pitch, half_width, shape):
    """由管中心和终点线生成六管ROI；单管底边不清时用同视频其余管的管长估计。"""
    h, w = shape
    bottoms = {}
    for i, center in enumerate(centers):
        found = find_bottom(gradient, center, half_width, float(np.polyval(coefficients, center)), pitch, h)
        if found:
            bottoms[i] = found
    if len(bottoms) < 4:
        raise ValueError("管底不足")
    spans = {i: bottoms[i][0] - float(np.polyval(coefficients, centers[i])) for i in bottoms}
    typical = float(np.median(list(spans.values())))
    if not 1.4 <= typical / pitch <= 2.1:
        raise ValueError(f"管长/管距={typical / pitch:.2f}")
    estimated = []
    tubes = {}
    margin = round(typical * 0.07)
    for i, center in enumerate(centers):
        finish = float(np.polyval(coefficients, center))
        if i in bottoms and abs(spans[i] - typical) <= typical * 0.1:
            bottom, strength = bottoms[i]
        else:
            bottom, strength = round(finish + typical), 0.0
            estimated.append(f"tube_{i + 1}")
        left = (centers[i - 1] + center) / 2 + 3 if i else center - pitch / 2
        right = (center + centers[i + 1]) / 2 - 3 if i < 5 else center + pitch / 2
        tubes[f"tube_{i + 1}"] = {
            "roi": [max(0, int(left)), max(0, round(finish - typical * 0.25)), min(w, int(right)), min(h, bottom + margin)],
            "bottom_y_reference": bottom, "axis_x_reference": float(center),
            "finish_y_at_axis": finish, "bottom_edge_strength": strength}
    if len(estimated) > 2:
        raise ValueError("多根管底不一致")
    return tubes, typical, estimated


def gradient_of(gray):
    return np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_32F, 0, 1, ksize=3))


def assemble(gray, coefficients, xs, ys, centers, pitch, half_width, method):
    h, w = gray.shape
    tubes, typical, estimated = build_tubes(gradient_of(gray), coefficients, centers, pitch, half_width, (h, w))
    line_error = float(np.sqrt(np.mean((ys - np.polyval(coefficients, xs)) ** 2)))
    return {"image_size": [w, h], "tubes": tubes, "tube_pitch_pixels": pitch, "tube_half_width": half_width,
            "pixels_per_cm": typical / 5.5, "estimated_bottoms": estimated,
            "finish_line": {"slope": float(coefficients[0]), "intercept": float(coefficients[1]),
                            "fit_rmse_pixels": line_error, "support_bins": len(xs),
                            "support_span_fraction": float(np.ptp(xs) / w),
                            "timing_quality": "ok" if line_error <= 3 and np.ptp(xs) > pitch * 4 else "review",
                            "definition": "背景深色横线中心，不是白色环线。"},
            "calibration_method": method}


def edge_profile(gray, board, coefficients, pitch_hint):
    """横线下方各列的竖直边缘强度：管身有管壁和反光条纹，空白板几乎为零。"""
    w = gray.shape[1]
    line_y = float(np.polyval(coefficients, w / 2))
    top, bottom = round(line_y + pitch_hint * 0.1), min(gray.shape[0], round(line_y + pitch_hint * 1.3))
    edges = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, 1, 0, ksize=3))
    usable = board[top:bottom].mean(axis=0) >= 0.95
    return edges[top:bottom].mean(axis=0), usable


def column_textures(profile, usable, centers, half_width):
    values = []
    for center in centers:
        a, b = int(center - half_width), int(center + half_width)
        if a < 0 or b > len(profile) or not usable[a:b].all():
            values.append(-1.0)
        else:
            values.append(float(profile[a:b].mean()))
    return np.asarray(values)


def correct_columns(gray, board, coefficients, centers, pitch, half_width):
    """六列都要有管身纹理。某列落在空白板上说明整体错位，按整数管距找到全部有纹理的位置；找不到就放弃。"""
    profile, usable = edge_profile(gray, board, coefficients, pitch)

    def valid(candidate):
        values = column_textures(profile, usable, candidate, half_width)
        return values.min() >= 0.62 * float(np.median(values))

    for shift in (0, -1, 1, -2, 2):
        moved = centers + shift * pitch
        if valid(moved):
            return moved
    return None


def tube_columns(gray, board, coefficients):
    """管间缺口看不见时，直接按管身纹理切出六列。"""
    w = gray.shape[1]
    profile, usable = edge_profile(gray, board, coefficients, w * 0.11)
    profile = np.where(usable, profile, 0).astype(np.float32)
    smooth = cv2.GaussianBlur(profile[:, None], (1, odd(w * 0.012)), 0).ravel()
    for level in (0.55, 0.45, 0.65, 0.35):
        high = float(np.percentile(smooth[smooth > 0], 80)) if (smooth > 0).any() else 0.0
        above = smooth > high * level
        edges = np.flatnonzero(np.diff(np.r_[0, above.astype(np.int8), 0]))
        runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= w * 0.035]
        if len(runs) != 6:
            continue
        centers = np.array([(a + b) / 2 for a, b in runs])
        widths = np.array([b - a for a, b in runs], float)
        steps = np.diff(centers)
        pitch = float(np.median(steps))
        if pitch < w * 0.05 or np.max(np.abs(steps - pitch)) > pitch * 0.2:
            continue
        if widths.min() < 0.5 * pitch or widths.max() > 1.05 * pitch:
            continue
        return centers, pitch, float(np.median(widths)) / 2
    return None


def propose_geometry(frame):
    """仅适用于六根直立试管贴在浅色白板、背景有深横线的拍摄协议；缩放和位置可变，失败时不硬猜。"""
    try:
        geometry = propose_geometry_caps(frame)
        spans = [item["bottom_y_reference"] - item["finish_y_at_axis"] for item in geometry["tubes"].values()]
        pitch = float(np.median(np.diff([item["axis_x_reference"] for item in geometry["tubes"].values()])))
        geometry.update(pixels_per_cm=float(np.median(spans)) / 5.5, estimated_bottoms=[],
                        tube_pitch_pixels=pitch, tube_half_width=pitch * 0.42)
        return geometry
    except ValueError:
        pass
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    board = board_mask(gray)
    hypotheses = line_candidates(gray, board)
    if not hypotheses:
        raise ValueError("白板内找不到深色横线。")
    rejected = []
    for coefficients, xs, ys in hypotheses:
        if abs(coefficients[0]) > 0.06:
            rejected.append("倾斜过大")
            continue
        comb = tube_gaps(frame, gray, coefficients, board)
        if comb is not None:
            centers = correct_columns(gray, board, coefficients, *comb)
            comb = None if centers is None else (centers, *comb[1:])
        if comb is None:
            comb = tube_columns(gray, board, coefficients)
        if comb is None:
            rejected.append("找不到六列管身")
            continue
        centers, pitch, half_width = comb
        # 真终点线贯穿六管并向两侧延伸；塞子边、架子边只覆盖局部。
        if np.ptp(xs) < (centers[-1] - centers[0]) * 0.85:
            rejected.append("横线未贯穿六管")
            continue
        try:
            return assemble(gray, coefficients, xs, ys, centers, pitch, half_width,
                            "规则图像处理：白板定位、深色横线、管间缺口或管身纹理、管长比例校验；不是通用场景识别模型。")
        except ValueError as exc:
            rejected.append(str(exc))
    raise ValueError(f"没有满足六管布局的横线假设：{rejected}")


def shifted(image, dx, dy):
    """把当前帧平移回参考坐标。"""
    matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def dark_detail(frame, size=31):
    """黑帽只响应比结构元小的暗斑，不受手机自动曝光造成的整帧亮度变化影响。"""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))


def static_detail(details):
    """低分位去掉移动果蝇、保留管壁等静态结构；果蝇只要四分之一时间不在某处就会被去掉。"""
    return np.percentile(np.stack(details), 25, axis=0).astype(np.float32)


def auto_calibrate(video):
    """前三秒多帧独立标定，取横线位置和管距都一致的最大一组，单帧误判不决定全局几何。"""
    failures, proposals = [], []
    frames = []
    with open_video(video) as capture:
        info = video_info(capture)
        for index in sorted({min(info["frame_count"] - 1, round(info["fps"] * t)) for t in [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3]}):
            frame = read_frame(capture, index, info["frame_count"])
            frames.append((index, frame))
            try:
                geometry = propose_geometry(frame)
                geometry["reference_frame"] = index
                proposals.append((frame, geometry))
            except ValueError as exc:
                failures.append({"frame": index, "reason": str(exc)})

    def agrees(one, other):
        return (abs(one["finish_line"]["intercept"] - other["finish_line"]["intercept"]) <= 15
                and abs(one["tube_pitch_pixels"] - other["tube_pitch_pixels"]) <= 0.1 * other["tube_pitch_pixels"])

    groups = [[item for item in proposals if agrees(item[1], seed[1])] for seed in proposals]
    largest = max(groups, key=len) if groups else []
    # 少数帧才通过的结果常是错位，宁可转人工点选，也不输出看似正常的错误几何。
    if len(largest) < 3:
        raise ValueError(f"无法可靠自动标定（{len(largest)}/{len(frames)}帧一致）：{failures}")
    largest.sort(key=lambda item: item[1]["finish_line"]["intercept"])
    frame, geometry = largest[len(largest) // 2]
    geometry["calibration_attempt_failures"] = failures
    geometry["calibration_votes"] = [len(largest), len(frames)]
    geometry["video"] = str(Path(video).resolve())
    return frame, geometry


# 与分析时的 ROI 下限一致，太小的标定当场拒绝。
MIN_ROI_WIDTH = 40
MIN_ROI_HEIGHT = 120


def geometry_from_clicks(line_points, bottom_points, image_size, reference_frame, distance_cm=5.5):
    """人工点选的终点线两端和六个管底中心转成与自动标定相同的几何结构。"""
    (ax, ay), (bx, by) = line_points
    if ax == bx or len(bottom_points) != 6:
        raise ValueError("需要终点线上左右两点和六个管底点。")
    w, h = image_size
    slope = (by - ay) / (bx - ax)
    intercept = ay - slope * ax
    centers = [float(x) for x, _ in bottom_points]
    if any(a >= b for a, b in zip(centers, centers[1:])):
        raise ValueError("六个管底必须从左到右依次点选。")
    pitch = float(np.median(np.diff(centers)))
    spans = [y - (slope * x + intercept) for x, y in bottom_points]
    typical = float(np.median(spans))
    if typical <= 0:
        raise ValueError("管底必须在终点线下方。")
    tubes = {}
    for i, (x, y) in enumerate(bottom_points):
        finish = slope * x + intercept
        left = (centers[i - 1] + x) / 2 + 3 if i else x - pitch / 2
        right = (x + centers[i + 1]) / 2 - 3 if i < 5 else x + pitch / 2
        tubes[f"tube_{i + 1}"] = {
            "roi": [max(0, int(left)), max(0, round(finish - typical * 0.25)), min(w, int(right)), min(h, round(y + typical * 0.07))],
            "bottom_y_reference": int(y), "axis_x_reference": float(x), "finish_y_at_axis": float(finish), "bottom_edge_strength": 0.0}
    for item in tubes.values():
        x1, y1, x2, y2 = item["roi"]
        if x2 - x1 < MIN_ROI_WIDTH or y2 - y1 < MIN_ROI_HEIGHT:
            raise ValueError("管子区域太小，请确认点的是横贯六管的深色横线和管子最底部。")
    return {"image_size": [w, h], "tubes": tubes, "tube_pitch_pixels": pitch,
            "pixels_per_cm": typical / distance_cm, "estimated_bottoms": [], "reference_frame": int(reference_frame),
            "finish_line": {"slope": float(slope), "intercept": float(intercept), "fit_rmse_pixels": 0.0,
                            "support_bins": 2, "support_span_fraction": abs(bx - ax) / w, "timing_quality": "ok",
                            "definition": "人工点选的背景深色横线。"},
            "calibration_method": "人工点选"}


def load_manual_geometry(video, path):
    geometry = json.loads(Path(path).read_text(encoding="utf-8"))
    with open_video(video) as capture:
        info = video_info(capture)
        if geometry["image_size"] != [info["width"], info["height"]]:
            raise ValueError("手动标定的画面尺寸与视频不一致。")
        frame = read_frame(capture, geometry["reference_frame"], info["frame_count"])
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
