"""直接复用已有 ROI 配置，重新读取视频帧并输出六根试管的裁剪和预览。"""

import argparse
from pathlib import Path
import sys

import cv2

from roi_utils import export_rois


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame", type=int, default=100, help="代表帧号，默认 100")
    parser.add_argument("--config", type=Path, default=Path("config/tube_rois.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/roi_crops"))
    args = parser.parse_args()
    try:
        export_rois(args.video, args.frame, args.config, args.output_dir)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
