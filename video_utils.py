"""视频读取的公共工具：获取基本信息、读取指定帧、保存图像。"""

from contextlib import contextmanager
from pathlib import Path
import math
import os
import subprocess

import cv2


@contextmanager
def open_video(path):
    """打开本地视频，并在操作结束或报错时释放文件。"""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"视频不存在：{path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV 无法打开视频：{path}")
        yield capture
    finally:
        capture.release()


def video_info(capture):
    """读取容器报告的参数；时长按总帧数 / FPS 估算。"""
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not all(math.isfinite(v) and v > 0 for v in (fps, frame_count, width, height)):
        raise ValueError("视频元数据无效，无法获取 FPS、帧数或分辨率。")
    return {
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "frame_count": int(frame_count),
        "duration_seconds": int(frame_count) / fps,
    }


def read_frame(capture, frame_index, frame_count):
    """按从 0 开始的帧号读取图像；超出范围时给出明确提示。"""
    if not 0 <= frame_index < frame_count:
        raise ValueError(f"帧号应在 0 到 {frame_count - 1} 之间，收到 {frame_index}。")
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
        raise ValueError(f"无法定位到第 {frame_index} 帧。")
    success, frame = capture.read()
    if not success or frame is None:
        raise ValueError(f"第 {frame_index} 帧解码失败；元数据帧数可能与可解码帧数不一致，也可能存在编码兼容问题。")
    return frame


def open_encoder(path, width, height, fps, log):
    """向 FFmpeg 输送内存中的 BGR 帧，不落盘中间图片，编码为浏览器可播放的 H.264。"""
    import imageio_ffmpeg

    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "warning", "-n",
               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
               "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2",
               "-movflags", "+faststart", str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def save_frame(frame, path):
    """保存 PNG/JPEG；编码后写入以支持 Windows 中文目录。"""
    path = Path(path)
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("图像输出路径必须以 .png、.jpg 或 .jpeg 结尾。")
    if path.exists():
        raise FileExistsError(f"输出文件已存在，请更换路径：{path}")
    success, encoded = cv2.imencode(path.suffix, frame)
    if not success:
        raise ValueError(f"图像编码失败：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(encoded.tobytes())
    return path.resolve()
