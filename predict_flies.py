"""单图检测入口：用 best.pt 输出可见果蝇数量、置信度、像素框和结果图，不生成个体 ID。"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
(ROOT / ".ultralytics").mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def predict_one(model, image_path, output_dir, conf=0.25, device="0"):
    from yolo_labels import read_image
    from video_utils import save_frame

    image_path, output_dir = Path(image_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / f"{image_path.stem}_prediction.png"
    json_path = output_dir / f"{image_path.stem}_prediction.json"
    if preview_path.exists() or json_path.exists():
        raise FileExistsError(f"输出已存在，请更换输出目录：{preview_path}")
    image = read_image(image_path)
    result = model.predict(source=image, imgsz=640, conf=conf, iou=0.7, device=device,
                           max_det=100, verbose=False, save=False)[0]
    detections = []
    for box in result.boxes:
        if int(box.cls.item()) != 0:
            raise ValueError("权重输出了非 fly 类别，可能误用了通用预训练模型。")
        detections.append({"class": 0, "name": "fly", "confidence": float(box.conf.item()),
                           "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()]})
    save_frame(result.plot(labels=True, conf=True, line_width=1, font_size=8), preview_path)
    payload = {"image": str(image_path.resolve()), "count": len(detections), "confidence_threshold": conf,
               "iou_threshold": 0.7, "detections": detections, "preview": str(preview_path.resolve())}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    import torch
    from ultralytics import YOLO
    from analyze_single_fly_video import resolve_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "assets/fly_yolo26n_best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/single_predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", choices=["auto", "0", "cpu"], default="auto")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf 必须在 0～1 之间。")
    if not args.weights.is_file():
        parser.error(f"权重不存在：{args.weights}")
    model = YOLO(str(args.weights), task="detect")
    if model.names != {0: "fly"}:
        parser.error("模型类别必须恰好为 {0: fly}。")
    payload = predict_one(model, args.image, args.output_dir, args.conf, resolve_device(args.device, torch))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
