"""YOLO Detection 标签工具：像素框与归一化框转换、严格校验、读写和反绘。"""

import math
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    """使用字节解码以兼容 Windows 中文路径。"""
    data = np.fromfile(Path(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"图片文件为空：{path}")
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"无法读取图片：{path}")
    return frame


def xyxy_to_yolo(box, width, height):
    """原图半开边界 [x1,y1,x2,y2] 转为归一化中心、宽、高。"""
    if width <= 0 or height <= 0 or len(box) != 4:
        raise ValueError("图片尺寸或矩形格式无效。")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in box):
        raise ValueError("像素坐标必须为有限数字。")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"矩形为空或超出图片范围：{box}")
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def yolo_to_xyxy(values, width, height):
    """反算像素坐标；检查整个框的边界，不能只检查中心是否在 0～1 内。"""
    if len(values) != 4 or width <= 0 or height <= 0:
        raise ValueError("归一化坐标或图片尺寸无效。")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("归一化坐标不能包含 NaN 或无穷大。")
    cx, cy, bw, bh = values
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
        raise ValueError("归一化中心必须在 0～1 内，宽和高必须大于 0 且不超过 1。")
    edges = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
    # 仅容忍小数文本舍入误差，明显越界的框直接拒绝。
    if any(v < -1e-8 or v > 1 + 1e-8 for v in edges):
        raise ValueError("归一化矩形超出图像边界。")
    return [min(1, max(0, v)) * size for v, size in zip(edges, [width, height, width, height])]


def read_labels(path, width, height):
    """每行恰好五列，仅接受类别 0；空文件表示人工确认没有目标。"""
    boxes = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError("每行必须为 0 x_center y_center width height。")
            box = yolo_to_xyxy([float(v) for v in fields[1:]], width, height)
            boxes.append(box)
        except ValueError as exc:
            raise ValueError(f"标签第 {line_number} 行错误：{exc}") from exc
    return boxes


def write_labels(path, boxes, width, height):
    """固定类别为 0，保留九位小数；用户主动保存时原子替换旧标签。"""
    lines = ["0 " + " ".join(f"{v:.9f}" for v in xyxy_to_yolo(box, width, height)) for box in boxes]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(path)
    return lines


def draw_boxes(image, boxes):
    """在原尺寸图像副本上反绘 fly 框；不修改待标注原图。"""
    result = image.copy()
    height, width = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = [round(v) for v in box]
        x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
        x2, y2 = max(x1 + 1, min(width, x2)), max(y1 + 1, min(height, y2))
        cv2.rectangle(result, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 1)
        cv2.putText(result, "fly", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(result, "fly", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return result
