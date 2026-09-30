"""OpenCV 临时鼠标标注窗口：从左到右选择六根试管，保存可复用的 ROI。"""

import argparse
import math
from pathlib import Path
import sys

import cv2

from roi_utils import draw_rois, export_rois, metadata_path, representative_frame, save_rois


def select_rois(frame, max_width=1280, max_height=800):
    """拖动鼠标画框，Enter 确认，Backspace 撤销，Esc 或关闭窗口取消。"""
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    display_width, display_height = max(1, round(width * scale)), max(1, round(height * scale))
    scale_x, scale_y = width / display_width, height / display_height
    state = {"start": None, "end": None, "dragging": False}
    rois = {}
    window = "Tube ROI - drag / Enter confirm / Backspace undo / Esc cancel"

    def mouse(event, x, y, flags, userdata):
        # 将窗口中的坐标映射回原始分辨率；保存文件不使用缩略图坐标。
        point = (min(max(x, 0), display_width), min(max(y, 0), display_height))
        if event == cv2.EVENT_LBUTTONDOWN and len(rois) < 6:
            state.update(start=point, end=point, dragging=True)
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["end"] = point
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state.update(end=point, dragging=False)

    def current_box():
        if state["start"] is None or state["end"] is None:
            return None
        (ax, ay), (bx, by) = state["start"], state["end"]
        if ax == bx or ay == by:
            return None
        return [math.floor(min(ax, bx) * scale_x), math.floor(min(ay, by) * scale_y),
                min(width, math.ceil(max(ax, bx) * scale_x)), min(height, math.ceil(max(ay, by) * scale_y))]

    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, mouse)
        while True:
            preview = draw_rois(frame, rois)
            box = current_box()
            if box:
                cv2.rectangle(preview, tuple(box[:2]), (box[2] - 1, box[3] - 1), (0, 170, 255), 3)
            display = cv2.resize(preview, (display_width, display_height))
            message = f"Select tube_{len(rois) + 1}/6 LEFT TO RIGHT" if len(rois) < 6 else "6/6 selected - Enter SAVE, Backspace undo"
            cv2.rectangle(display, (0, 0), (display_width, 40), (20, 20, 20), -1)
            cv2.putText(display, message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            cv2.imshow(window, display)
            key = cv2.waitKeyEx(30)
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (8, 127):
                if state["start"] is None and rois:
                    rois.popitem()
                state.update(start=None, end=None, dragging=False)
            elif key in (10, 13, 32) and not state["dragging"]:
                # waitKey 会处理鼠标事件，确认时重新获取最新矩形。
                box = current_box()
                if len(rois) == 6:
                    return rois
                if box and box not in rois.values():
                    name = f"tube_{len(rois) + 1}"
                    rois[name] = box
                    print(f"已确认 {name}：{box}", flush=True)
                    state.update(start=None, end=None, dragging=False)
    finally:
        cv2.destroyAllWindows()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame", type=int, default=100, help="代表帧号，默认 100，从 0 开始")
    parser.add_argument("--config", type=Path, default=Path("config/tube_rois.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/roi_selected"))
    parser.add_argument("--overwrite", action="store_true", help="允许替换已有 ROI 配置；不会覆盖裁剪图片")
    args = parser.parse_args()
    try:
        if args.config.suffix.lower() != ".json" or args.config.name.endswith(".meta.json"):
            raise ValueError("--config 应为普通 .json 文件，不能使用 .meta.json 后缀。")
        if not args.overwrite and (args.config.exists() or metadata_path(args.config).exists()):
            raise FileExistsError("ROI 配置已存在，请使用新文件名，或通过 --overwrite 重新选择。")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("输出目录非空，请通过 --output-dir 指定新目录。")
        frame = representative_frame(args.video, args.frame)
        print("请从左到右框选完整试管。拖动鼠标后按 Enter 确认，每次确认一根。", flush=True)
        print("框满 6 根后再次按 Enter 保存；Backspace 撤销；Esc 或关闭窗口取消，不改动配置。", flush=True)
        rois = select_rois(frame)
        if rois is None:
            print("已取消，未保存 ROI 配置。")
            return
        save_rois(args.config, rois, frame, args.video, args.frame, overwrite=args.overwrite)
        print(f"ROI 已保存：{args.config.resolve()}")
        export_rois(args.video, args.frame, args.config, args.output_dir)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
