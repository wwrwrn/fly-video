"""用 dataset_v2 微调单类别果蝇检测器；从现有模型继续，权重写到 runs/，不改动 assets/ 里的发布模型。"""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
(ROOT / ".ultralytics").mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def main():
    import torch
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset_v2/dataset.yaml")
    parser.add_argument("--start", type=Path, default=ROOT / "assets/fly_yolo26n_best.pt")
    parser.add_argument("--name", default="fly_yolo26n_v2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    run_dir = ROOT / "runs" / "detect" / args.name
    if run_dir.exists():
        raise FileExistsError(f"训练目录已存在，请使用新的 --name：{run_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，停止训练，不静默改用 CPU。")
    torch.set_num_threads(4)
    model = YOLO(str(args.start), task="detect")
    # 推理时六管补成同尺寸后按长边640缩放，这里用相同尺度；果蝇方向任意，所以允许上下翻转。
    # 不用 mosaic：拼接会缩小本来就很小的目标。
    parameters = dict(
        data=str(args.data), epochs=args.epochs, imgsz=640, batch=args.batch, device=0,
        optimizer="AdamW", lr0=0.0008, lrf=0.02, weight_decay=0.0005, workers=2, patience=20,
        seed=42, deterministic=True, amp=True, cache=False, rect=False, multi_scale=False,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=5.0, translate=0.05, scale=0.25, shear=0.0,
        perspective=0.0, flipud=0.5, fliplr=0.5, hsv_h=0.0, hsv_s=0.1, hsv_v=0.25, close_mosaic=0,
        erasing=0.0, plots=True, save=True, val=True, project=str(ROOT / "runs/detect"), name=args.name,
        exist_ok=False,
    )
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / f"{args.name}_request.json").write_text(
        json.dumps({"start": str(args.start), "parameters": parameters}, ensure_ascii=False, indent=2), encoding="utf-8")
    model.train(**parameters)
    print(f"训练完成：{model.trainer.best}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
