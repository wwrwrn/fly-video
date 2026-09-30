"""按照指定秒数间隔抽取视频帧，保存图片和便于后续标注的索引 CSV。"""

import argparse
import csv
import math
from pathlib import Path
import sys

import cv2

from video_utils import open_video, read_frame, save_frame, video_info


def main():
    # Windows 重定向输出时仍使用 UTF-8，避免中文提示乱码。
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="MP4 视频路径")
    parser.add_argument("--interval", type=float, default=1.0, help="抽帧间隔，单位秒，默认 1")
    parser.add_argument("--start", type=float, default=0.0, help="开始时间，单位秒，默认 0")
    parser.add_argument("--count", type=int, default=10, help="最多抽取帧数，默认 10")
    parser.add_argument("--output-dir", type=Path, help="输出目录，默认 outputs/frames/视频名")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval 必须是大于 0 的有限秒数。")
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("--start 必须是大于等于 0 的有限秒数。")
    if args.count <= 0:
        parser.error("--count 必须大于 0。")

    try:
        with open_video(args.video) as capture:
            info = video_info(capture)
            fps = info["fps"]
            if args.interval < 1 / fps:
                raise ValueError(f"抽帧间隔不能小于一帧的时长（{1 / fps:.6f} 秒）。")
            # 用 FPS 换算最近的帧号；精确的变帧率时间戳留待后续阶段处理。
            if args.start >= info["duration_seconds"]:
                raise ValueError("开始时间超出可读取视频帧的范围。")
            first_index = math.floor(args.start * fps + 0.5)
            if first_index >= info["frame_count"]:
                raise ValueError("开始时间超出可读取视频帧的范围。")
            output_dir = args.output_dir or Path("outputs/frames") / args.video.stem
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"输出目录非空，请指定新的目录：{output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            saved_count = 0
            # 实时记录已成功保存的帧，遇到损坏帧时也保留之前的索引。
            with (output_dir / "frames.csv").open("x", newline="", encoding="utf-8-sig") as manifest:
                writer = csv.writer(manifest)
                writer.writerow(["filename", "frame_index", "requested_seconds", "frame_seconds"])
                for number in range(args.count):
                    requested_seconds = args.start + number * args.interval
                    if not math.isfinite(requested_seconds) or requested_seconds >= info["duration_seconds"]:
                        break
                    frame_index = math.floor(requested_seconds * fps + 0.5)
                    if frame_index >= info["frame_count"]:
                        break
                    frame = read_frame(capture, frame_index, info["frame_count"])
                    filename = f"frame_{frame_index:06d}.png"
                    save_frame(frame, output_dir / filename)
                    writer.writerow([filename, frame_index, f"{requested_seconds:.6f}", f"{frame_index / fps:.6f}"])
                    saved_count += 1
                    print(f"已保存 {filename}，对应时间约 {frame_index / fps:.3f} 秒")
            print(f"完成：共保存 {saved_count} 帧，目录：{output_dir.resolve()}")
            if saved_count < args.count:
                print("已到达视频末尾，可抽取帧数少于请求数量。")
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
