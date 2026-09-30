"""独立验证一张图片和同名 YOLO txt，将标签反算为像素框并保存预览。"""

import argparse
from pathlib import Path
import sys

import cv2

from video_utils import save_frame
from yolo_labels import draw_boxes, read_image, read_labels


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("label", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.image.stem != args.label.stem:
            raise ValueError("图片与标签必须同名，例如 sample.png 对应 sample.txt。")
        if args.label.suffix.lower() != ".txt":
            raise ValueError("标签必须是 .txt 文件。")
        frame = read_image(args.image)
        boxes = read_labels(args.label, frame.shape[1], frame.shape[0])
        save_frame(draw_boxes(frame, boxes), args.output)
        print(f"验证通过：{len(boxes)} 个 fly 框；预览：{args.output.resolve()}")
        for i, box in enumerate(boxes, 1):
            print(f"框 {i} 像素 xyxy：{[round(v, 6) for v in box]}")
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
