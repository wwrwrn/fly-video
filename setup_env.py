"""一次性安装本地应用环境：自动选CUDA/CPU官方PyTorch wheel，保持依赖版本可复现。"""

import argparse
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def install_plan(accelerator):
    """NVIDIA可见时选择CUDA构建；无显卡或明确指定时选择CPU构建。"""
    if accelerator != "auto":
        return accelerator
    command = shutil.which("nvidia-smi")
    if command:
        probe = subprocess.run([command, "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, check=False)
        if probe.returncode == 0 and probe.stdout.strip():
            return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11) or struct.calcsize("P") != 8:
        print("请先安装 Python 3.11（64位），再运行 setup_env.cmd。", file=sys.stderr)
        return 2
    if not (ROOT / "assets/fly_yolo26n_best.pt").is_file():
        print("下载内容缺少 assets/fly_yolo26n_best.pt，请重新下载完整仓库。", file=sys.stderr)
        return 2
    environment = ROOT / ".venv-train"
    python = environment / "Scripts/python.exe"
    if not python.is_file():
        print("正在创建项目环境 .venv-train ...", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    mode = install_plan(args.accelerator)
    index = "https://download.pytorch.org/whl/cu128" if mode == "cuda" else "https://download.pytorch.org/whl/cpu"
    print(f"安装模式：{mode}。首次安装需要网络和下载时间。", flush=True)
    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(python), "-m", "pip", "install", "torch==2.9.1", "torchvision==0.24.1", "--index-url", index],
        [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements-app.txt")],
        [str(python), "-m", "pip", "check"],
        [str(python), str(ROOT / "check_install.py")],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            print("安装未完成。请保留上方错误信息；网络恢复后可重新运行 setup_env.cmd。", file=sys.stderr)
            return result.returncode
    print("安装和检查完成。现在双击 run_single_fly.cmd 选择视频。", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
