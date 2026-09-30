"""训练前只读质检：检查全部人工标签，统计数量，生成逐图预览和五张联系表。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from video_utils import save_frame
from yolo_labels import draw_boxes, read_image, read_labels

FRAMES = (0, 80, 160, 240, 320)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b):
    """计算矩形 IoU，用于发现几乎画在同一位置的疑似重复框。"""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def contact_sheet(items, path):
    """按一行六张排列，保留长宽比；仅缩放预览，不改变训练图片。"""
    cell_width, cell_height = 320, 910
    sheet = np.full((cell_height, cell_width * len(items), 3), 35, dtype=np.uint8)
    for column, (title, image) in enumerate(items):
        scale = min(300 / image.shape[1], 840 / image.shape[0])
        width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
        left = column * cell_width + (cell_width - width) // 2
        sheet[60:60 + height, left:left + width] = cv2.resize(image, (width, height))
        cv2.putText(sheet, title, (column * cell_width + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    save_frame(sheet, path)


def run_qa(dataset, output, duplicate_iou=0.85):
    dataset, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"QA 输出目录非空：{output}")
    output.mkdir(parents=True, exist_ok=True)
    expected = [f"frame_{frame:06d}_tube_{tube}" for frame in FRAMES for tube in range(1, 7)]
    issues, rows, hashes = [], [], {}
    for folder, suffix in [("images", ".png"), ("labels", ".txt")]:
        actual = {p.stem for p in (dataset / folder).glob(f"*{suffix}")}
        for stem in sorted(actual - set(expected)):
            issues.append({"file": f"{folder}/{stem}{suffix}", "reason": "存在预期 30 张之外的文件"})
    for frame in FRAMES:
        sheet_items = []
        for tube in range(1, 7):
            stem = f"frame_{frame:06d}_tube_{tube}"
            image_path, label_path = dataset / "images" / f"{stem}.png", dataset / "labels" / f"{stem}.txt"
            for path in [image_path, label_path]:
                if path.is_file():
                    hashes[path.relative_to(dataset).as_posix()] = digest(path)
            row = {"image": image_path.name, "frame": frame, "tube": tube, "count": None, "status": "ok"}
            try:
                image = read_image(image_path)
                boxes = read_labels(label_path, image.shape[1], image.shape[0])
                row.update(count=len(boxes), width=image.shape[1], height=image.shape[0])
                for i, box in enumerate(boxes):
                    for j, previous in enumerate(boxes[:i]):
                        iou = overlap(box, previous)
                        if iou >= duplicate_iou:
                            issues.append({"file": label_path.name, "reason": "疑似重复框，需人工确认", "rows": [j + 1, i + 1], "iou": iou})
                            row["status"] = "suspect"
                draft = dataset / "drafts" / f"{stem}.json"
                if draft.exists():
                    issues.append({"file": draft.name, "reason": "存在未完成草稿，需先确认正式标签"})
                    row["status"] = "suspect"
                preview = draw_boxes(image, boxes)
                save_frame(preview, output / "previews" / f"{stem}.png")
                sheet_items.append((f"tube_{tube}  n={len(boxes)}", preview))
            except (OSError, ValueError, cv2.error) as exc:
                row["status"] = "invalid"
                issues.append({"file": stem, "reason": str(exc)})
            rows.append(row)
        if sheet_items:
            contact_sheet(sheet_items, output / "contact_sheets" / f"frame_{frame:06d}.png")
    # 质检结束后核对文件哈希，确保输入数据没有在检查中被改动。
    unchanged = all(digest(dataset / name) == value for name, value in hashes.items())
    if not unchanged:
        issues.append({"reason": "QA 期间源数据发生变化，必须重新检查"})
    totals = {split: {"images": sum(row["count"] is not None for row in rows if (row["frame"] == 320) == (split == "val")),
                      "instances": sum(row["count"] or 0 for row in rows if (row["frame"] == 320) == (split == "val"))}
              for split in ["train", "val"]}
    report = {"dataset": str(dataset.resolve()), "passed": not issues, "source_unchanged": unchanged,
              "image_count": sum(row["count"] is not None for row in rows), "total_instances": sum(row["count"] or 0 for row in rows),
              "duplicate_iou_threshold": duplicate_iou, "issues": issues, "split_counts": totals,
              "images": rows, "source_sha256": hashes,
              "scope": "仅检测可见果蝇；棉花中完全不可见不猜框；边缘可辨认部分仍标 fly；不处理身份。"}
    (output / "qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "image_counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "frame", "tube", "count", "status", "width", "height"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: report[key] for key in ["passed", "image_count", "total_instances", "split_counts", "issues"]}, ensure_ascii=False, indent=2))
    return report


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset_sample"))
    parser.add_argument("--output", type=Path, default=Path("outputs/training_qa"))
    args = parser.parse_args()
    try:
        report = run_qa(args.dataset, args.output)
        if not report["passed"]:
            parser.exit(1, "发现标签异常，停止训练。人工标签未修改，详见 qa_report.json。\n")
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
