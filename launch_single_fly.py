"""Windows本地入口：接受拖入的视频路径；无参数时弹出文件选择框。"""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "outputs" / "single_fly_runs"


def choose_video():
    """双击命令文件时使用系统文件选择框，取消时不创建输出。"""
    import tkinter as tk
    from tkinter import filedialog

    window = tk.Tk()
    window.withdraw()
    window.attributes("-topmost", True)
    try:
        return filedialog.askopenfilename(
            parent=window,
            title="选择六管果蝇实验视频",
            initialdir=str(ROOT / "video" if (ROOT / "video").is_dir() else ROOT),
            filetypes=[("MP4 视频", "*.mp4"), ("全部文件", "*.*")],
        )
    finally:
        window.destroy()


def next_output(video, parent=RUNS):
    """为每次运行取新目录名，保留已有结果。"""
    first = parent / video.stem
    if not first.exists():
        return first
    number = 1
    while (parent / f"{video.stem}_run{number}").exists():
        number += 1
    return parent / f"{video.stem}_run{number}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", help="MP4 路径；不填写时弹出文件选择框")
    args = parser.parse_args(argv)
    try:
        selected = args.video if args.video is not None else choose_video()
    except Exception as exc:
        print(f"无法打开文件选择窗口：{exc}。可把视频拖到run_single_fly.cmd，或在命令行传入路径。", file=sys.stderr)
        return 2
    if not selected:
        print("已取消选择，没有运行分析。", flush=True)
        return 2
    video = Path(selected).expanduser().resolve()
    if not video.is_file():
        print(f"错误：视频文件不存在：{video}", file=sys.stderr, flush=True)
        return 2
    if video.suffix.lower() != ".mp4":
        print("错误：目前请选择 MP4 视频。", file=sys.stderr, flush=True)
        return 2
    python = ROOT / ".venv-train" / "Scripts" / "python.exe"
    if not python.is_file():
        print("首次使用请先双击 setup_env.cmd 安装环境。", file=sys.stderr, flush=True)
        return 2
    if not (ROOT / "assets/fly_yolo26n_best.pt").is_file():
        print("下载的项目缺少 assets/fly_yolo26n_best.pt，请重新下载完整仓库。", file=sys.stderr, flush=True)
        return 2
    output = next_output(video)
    print(f"输入视频：{video}", flush=True)
    print(f"本次结果：{output}", flush=True)
    result = subprocess.run(
        [str(python), "-u", str(ROOT / "analyze_single_fly_video.py"), str(video), "--output", str(output)],
        cwd=ROOT,
        check=False,
    )
    if result.returncode:
        print(f"分析未完成（退出码 {result.returncode}）。请查看上方错误信息。", flush=True)
        return result.returncode
    print(f"完成。带速率视频：{output / 'annotated_single_fly.mp4'}", flush=True)
    print(f"数据表：{output / 'first_passage_results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
