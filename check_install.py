"""师兄电脑上的安装检查：依赖、随仓库模型完整性、类别及实际CUDA/CPU运行设备。"""

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
(ROOT / ".ultralytics").mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def main():
    model_path = ROOT / "assets/fly_yolo26n_best.pt"
    info = json.loads((ROOT / "assets/model_info.json").read_text(encoding="utf-8"))
    if not model_path.is_file() or hashlib.sha256(model_path.read_bytes()).hexdigest() != info["sha256"]:
        raise RuntimeError("随仓库提供的模型文件缺失或不完整，请重新下载。")
    import cv2
    import imageio_ffmpeg
    import numpy
    import torch
    import tkinter
    import ultralytics
    from ultralytics import YOLO
    from analyze_single_fly_video import resolve_device

    device = resolve_device("auto", torch)
    model = YOLO(str(model_path), task="detect")
    if model.names != {0: "fly"}:
        raise RuntimeError("模型类别应为0: fly。")
    report = {"python": sys.version.split()[0], "executable": sys.executable,
              "torch": torch.__version__, "torch_cuda_version": torch.version.cuda,
              "device": device, "gpu": torch.cuda.get_device_name(0) if device == "0" else None,
              "ultralytics": ultralytics.__version__, "opencv": cv2.__version__, "numpy": numpy.__version__,
              "tkinter": tkinter.TkVersion, "ffmpeg": imageio_ffmpeg.get_ffmpeg_exe(),
              "model": str(model_path), "model_sha256": info["sha256"], "classes": model.names}
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "install_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("环境和模型检查通过。CPU模式也可用，但处理速度较慢。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"检查失败：{exc}\n请重新运行 setup_env.cmd，或把错误信息发给维护者。", file=sys.stderr)
        raise SystemExit(1)
