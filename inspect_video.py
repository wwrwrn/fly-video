"""命令行查看视频参数，并保存或显示指定的一帧，不包含软件 GUI。"""

import argparse
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
    parser.add_argument("--frame", type=int, help="指定帧号，从 0 开始")
    parser.add_argument("--output", type=Path, help="单帧保存路径，默认保存到 outputs/preview")
    parser.add_argument("--show", action="store_true", help="通过 OpenCV 显示静态帧，按任意键关闭")
    args = parser.parse_args()
    if args.frame is None and (args.output or args.show):
        parser.error("使用 --output 或 --show 时需要同时指定 --frame。")

    try:
        with open_video(args.video) as capture:
            info = video_info(capture)
            print(f"视频：{args.video.resolve()}")
            print(f"分辨率：{info['width']} × {info['height']}")
            print(f"FPS：{info['fps']:.6f}")
            print(f"总帧数：{info['frame_count']}")
            print(f"视频总时长（帧数 / FPS）：{info['duration_seconds']:.6f} 秒")
            if args.frame is not None:
                frame = read_frame(capture, args.frame, info["frame_count"])
                output = args.output or Path("outputs/preview") / f"{args.video.stem}_frame_{args.frame:06d}.png"
                saved = save_frame(frame, output)
                print(f"已保存第 {args.frame} 帧：{saved}")
                if args.show:
                    # 仅显示静态图像，不实现交互式分析界面。
                    try:
                        cv2.imshow(f"Frame {args.frame} - press any key to close", frame)
                        cv2.waitKey(0)
                    finally:
                        cv2.destroyAllWindows()
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
