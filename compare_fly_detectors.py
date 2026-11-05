"""在留出实验组上比较两个检测器。标签是伪标签：差值可比，绝对数值不等于人工真值上的准确率。"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
(ROOT / ".ultralytics").mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
CONFIDENCES = (0.1, 0.25, 0.5)


def box_iou(box, boxes):
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    overlap = np.prod(np.clip(bottom_right - top_left, 0, None), axis=1)
    area = np.prod(box[2:] - box[:2]) + np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) - overlap
    return overlap / np.maximum(area, 1e-9)


def read_truth(label, width, height):
    rows = [list(map(float, line.split()[1:])) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    return np.array([[(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height]
                     for x, y, w, h in rows]).reshape(-1, 4)


def counts_at_conf(model, images, labels, device):
    """单蝇场景下真正影响测量的是：有果蝇时找到没有，空管时误报没有。"""
    # 逐张推理：测试图尺寸不一，且把整个列表传给 predict 会一次性堆进显存。
    predictions = [model.predict(str(image), conf=min(CONFIDENCES), iou=0.45, imgsz=640, device=device,
                                 rect=True, max_det=20, verbose=False)[0].cpu() for image in images]
    table = {}
    for confidence in CONFIDENCES:
        found = total = wrong = empty_images = empty_false = 0
        for prediction, label in zip(predictions, labels):
            height, width = prediction.orig_shape
            truth = read_truth(label, width, height)
            data = prediction.boxes.data.cpu().numpy()
            boxes = data[data[:, 4] >= confidence, :4]
            if len(truth) == 0:
                empty_images += 1
                empty_false += len(boxes) > 0
                continue
            total += len(truth)
            matched = np.zeros(len(boxes), bool)
            for item in truth:
                if len(boxes):
                    overlaps = box_iou(item, boxes)
                    best = int(np.argmax(overlaps))
                    if overlaps[best] >= 0.3 and not matched[best]:
                        matched[best] = True
                        found += 1
            wrong += int((~matched).sum())
        table[str(confidence)] = {"recall": found / max(1, total), "false_boxes_per_fly_image": wrong / max(1, total),
                                  "empty_tube_false_rate": empty_false / max(1, empty_images),
                                  "flies": total, "empty_images": empty_images}
    return table


def evaluate(path, data, images, labels, device):
    from ultralytics import YOLO

    model = YOLO(str(path), task="detect")
    metrics = model.val(data=str(data), split="test", imgsz=640, batch=16, device=device, conf=0.001, iou=0.7,
                        rect=False, workers=0, plots=False, verbose=False, project=str(ROOT / "runs/detect"),
                        name=f"compare_{path.parent.parent.name if path.parent.name == 'weights' else path.stem}", exist_ok=True)
    return {"weights": str(path), "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
            "at_confidence": counts_at_conf(model, images, labels, device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=ROOT / "assets/fly_yolo26n_best.pt")
    parser.add_argument("--new", type=Path, default=ROOT / "runs/detect/fly_yolo26n_v2/weights/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset_v2")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/detector_comparison.json")
    args = parser.parse_args()
    images = sorted((args.data / "images/test").glob("*.jpg"))
    labels = [args.data / "labels/test" / f"{image.stem}.txt" for image in images]
    if not images:
        raise FileNotFoundError("留出测试集为空。")
    report = {"test_images": len(images), "note": "标签为伪标签，只用于新旧模型相对比较。",
              "old": evaluate(args.old, args.data / "dataset.yaml", images, labels, args.device),
              "new": evaluate(args.new, args.data / "dataset.yaml", images, labels, args.device)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
