"""使用 best.pt 验证六张 frame 320 图片，保存默认指标、预测结果和人工数量对比。"""

import argparse
import csv
import json
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
    from predict_flies import predict_one
    from qa_training_labels import contact_sheet
    from yolo_labels import read_image, read_labels

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/detect/fly_yolo26n_teaching/weights/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/validation_predictions")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空：{args.output}")
    torch.set_num_threads(4)
    model = YOLO(str(args.weights), task="detect")
    if model.names != {0: "fly"}:
        raise ValueError("模型类别应为 {0: fly}。")
    # 指标使用验证模式默认的低置信度门槛，以计算 PR 曲线；数量比较固定使用 0.25。
    metrics = model.val(data=str(ROOT / "dataset_yolo/dataset.yaml"), split="val", imgsz=640,
                        batch=4, device=0, workers=0, conf=0.001, iou=0.7, plots=True,
                        project=str(ROOT / "runs/detect"), name="fly_yolo26n_best_val", exist_ok=False)
    rows, sheet_items = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((ROOT / "dataset_yolo/images/val").glob("*.png")):
        image = read_image(image_path)
        human = len(read_labels(ROOT / "dataset_yolo/labels/val" / f"{image_path.stem}.txt", image.shape[1], image.shape[0]))
        payload = predict_one(model, image_path, args.output, conf=0.25, device="0")
        row = {"image": image_path.name, "human_count": human, "prediction_count": payload["count"],
               "difference": payload["count"] - human}
        rows.append(row)
        tube = image_path.stem.split("tube_")[-1]
        sheet_items.append((f"tube_{tube} GT={human} Pred={payload['count']}", read_image(payload["preview"])))
    contact_sheet(sheet_items, args.output / "validation_contact_sheet.png")
    with (args.output / "count_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"weights": str(args.weights.resolve()), "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
              "validation_confidence_floor": 0.001, "count_comparison_confidence_threshold": 0.25,
              "counts": rows, "limitation": "验证图和训练图来自同一短视频，不能代表跨视频泛化能力；数量相同也不保证每个框都正确。"}
    (args.output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
