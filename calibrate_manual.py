"""自动标定失败时手动点选：先点终点线左右两端，再从左到右点六个管底中心。"""

import argparse
import json
from pathlib import Path
import sys

import cv2

from app_paths import DATA
from single_fly_geometry import draw_geometry, geometry_from_clicks
from video_utils import open_video, read_frame, video_info

MANUAL_DIR = DATA / "config" / "manual_geometry"
PROMPTS = ["终点线左端", "终点线右端"] + [f"tube_{i} 管底中心" for i in range(1, 7)]
# OpenCV 窗口画不了中文，窗口内提示用英文。
HINTS = ["LEFT end of the dark horizontal line", "RIGHT end of the dark horizontal line"] + [
    f"BOTTOM center of tube {i} (the tube, not the fly)" for i in range(1, 7)]
WINDOW = "manual calibration: click points, Backspace undo, Esc cancel"
MAX_LINE_SLOPE = 0.15
MAX_BOTTOM_SPREAD = 0.25
# 按画面比例判断：横线要横贯六管，管底到横线要有足够高度。
MIN_LINE_WIDTH = 0.4
MIN_TUBE_HEIGHT = 0.15


def click_problem(points, point, frame_size):
    """这一下点得不合理时返回原因，当场拒绝，不等到八个点点完才报错。"""
    x, y = point
    width, height = frame_size
    if len(points) == 1:
        (ax, ay) = points[0]
        if x <= ax or abs(y - ay) > MAX_LINE_SLOPE * (x - ax):
            return "line must go left to right and stay nearly level"
        if x - ax < MIN_LINE_WIDTH * width:
            return "line must span all six tubes"
    if len(points) < 2:
        return None
    (ax, ay), (bx, by) = points[:2]
    span = y - (ay + (by - ay) / (bx - ax) * (x - ax))
    if span <= 0:
        return "tube bottom must be below the line"
    if span < MIN_TUBE_HEIGHT * height:
        return "tube bottom is too close to the line"
    bottoms = points[2:]
    if bottoms:
        if x <= bottoms[-1][0]:
            return "click tubes from left to right"
        if abs(y - bottoms[0][1]) > MAX_BOTTOM_SPREAD * span:
            return "tube bottoms must be at about the same height"
    return None


def caption(view, text, colour):
    cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(view, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame", type=int, default=30, help="用于点选的帧号")
    parser.add_argument("--distance-cm", type=float, default=5.5)
    args = parser.parse_args()
    with open_video(args.video) as capture:
        info = video_info(capture)
        frame = read_frame(capture, args.frame, info["frame_count"])
    size = (info["width"], info["height"])
    scale = min(1.0, 1400 / info["width"], 800 / info["height"])
    points = []
    message = [None]

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(PROMPTS):
            point = (round(x / scale), round(y / scale))
            message[0] = click_problem(points, point, size)
            if message[0] is None:
                points.append(point)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    geometry = None
    while True:
        if len(points) == len(PROMPTS):
            # 先看预览：线和六个框都对才保存。
            geometry = geometry or geometry_from_clicks(points[:2], points[2:], size, args.frame, args.distance_cm)
            view = cv2.resize(draw_geometry(frame, geometry), None, fx=scale, fy=scale)
            caption(view, "Check the red line and six boxes. Enter = save, Backspace = undo", (0, 255, 0))
        else:
            geometry = None
            view = cv2.resize(frame, None, fx=scale, fy=scale)
            for x, y in points:
                cv2.circle(view, (round(x * scale), round(y * scale)), 4, (0, 0, 255), -1)
            if len(points) >= 2:
                (ax, ay), (bx, by) = points[:2]
                cv2.line(view, (round(ax * scale), round(ay * scale)), (round(bx * scale), round(by * scale)), (0, 0, 255), 1)
            text = f"{len(points) + 1}/{len(PROMPTS)}: click {HINTS[len(points)]}"
            caption(view, f"Not accepted: {message[0]}" if message[0] else text, (0, 0, 255) if message[0] else (255, 255, 255))
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            print("已取消，没有保存。")
            return 2
        if key == 8 and points:
            points.pop()
            message[0] = None
        if key == 13 and geometry is not None:
            break
    cv2.destroyAllWindows()
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    output = MANUAL_DIR / f"{args.video.stem}.json"
    output.write_text(json.dumps(geometry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview = output.with_suffix(".png")
    ok, encoded = cv2.imencode(".png", draw_geometry(frame, geometry))
    if ok:
        preview.write_bytes(encoded.tobytes())
    print(f"已保存：{output}\n预览：{preview}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("依次点击：" + "、".join(PROMPTS))
    raise SystemExit(main())
