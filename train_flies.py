"""第一次教学微调：本机 GPU、轻量预训练 Detection 模型、固定保守参数，不做跟踪。"""

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
    from qa_training_labels import digest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="fly_yolo26n_teaching")
    args = parser.parse_args()
    run_dir = ROOT / "runs" / "detect" / args.name
    if run_dir.exists():
        raise FileExistsError(f"训练目录已存在，请使用新的 --name：{run_dir}")
    qa = json.loads((ROOT / "outputs/training_qa/qa_report.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "dataset_yolo/split_report.json").read_text(encoding="utf-8"))
    if not qa["passed"] or qa["issues"]:
        raise ValueError("标签质检未通过，禁止训练。")
    # 再次验证原标签和训练副本，防止 QA 后数据被悄悄修改。
    for relative, signature in split["original_sha256"].items():
        if digest(Path(split["source"]) / relative) != signature:
            raise ValueError(f"原始数据已变化，停止训练并重新质检：{relative}")
    for entry in split["images"]:
        for folder, suffix in [("images", ".png"), ("labels", ".txt")]:
            filename = Path(entry["image"]).stem + suffix
            copied = ROOT / "dataset_yolo" / folder / entry["split"] / filename
            if digest(copied) != split["original_sha256"][f"{folder}/{filename}"]:
                raise ValueError(f"训练副本与原始人工数据不一致：{copied}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，停止训练，不静默改用 CPU。")
    torch.set_num_threads(4)
    model_path = ROOT / "models/yolo26n.pt"
    model_path.parent.mkdir(exist_ok=True)
    # 指定 .pt，自动从 Ultralytics 官方资产加载预训练权重；不从 YAML 随机初始化。
    model = YOLO(str(model_path), task="detect")
    parameters = dict(
        data=str(ROOT / "dataset_yolo/dataset.yaml"), epochs=60, imgsz=640, batch=4,
        device=0, optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
        workers=0, patience=20, seed=42, deterministic=True, pretrained=True, nbs=16,
        amp=True, cache=False, rect=True, multi_scale=False,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
        translate=0.02, scale=0.1, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,
        close_mosaic=0, erasing=0.0, plots=True, save=True, val=True,
        project=str(ROOT / "runs/detect"), name=args.name, exist_ok=False,
    )
    request = {"model": str(model_path), "pretrained_sha256": digest(model_path),
               "parameters": parameters, "reason": "YOLO26 nano 是正式支持的轻量预训练检测模型，适合 8GB GPU 的教学微调。"}
    (ROOT / "outputs/training_request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(request, ensure_ascii=False, indent=2), flush=True)
    model.train(**parameters)
    best = Path(model.trainer.best)
    last = Path(model.trainer.last)
    if not best.is_file() or not last.is_file():
        raise RuntimeError("训练结束但 best.pt 或 last.pt 缺失，请检查训练日志。")
    summary = {"run_dir": str(Path(model.trainer.save_dir).resolve()), "best": str(best.resolve()),
               "last": str(last.resolve()), "metrics": {k: float(v) for k, v in model.metrics.results_dict.items()},
               "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
               "peak_torch_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3}
    (ROOT / "outputs/training_completion.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
