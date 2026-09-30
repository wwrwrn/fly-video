"""只抽取五个指定时间点，每帧裁剪六根试管，生成 30 张待人工标注图片。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2

from roi_utils import load_rois, metadata_path, representative_frame
from video_utils import save_frame

FRAME_INDICES = (0, 80, 160, 240, 320)


def build_dataset(video, config, output):
    """要求当前坐标具有活动区域确认记录；重新选框后旧确认自动失效。"""
    config, output = Path(config), Path(output)
    rois = json.loads(config.read_text(encoding="utf-8-sig"))
    meta = json.loads(metadata_path(config).read_text(encoding="utf-8-sig"))
    signature = hashlib.sha256(json.dumps(rois, sort_keys=True).encode("utf-8")).hexdigest()
    if meta.get("activity_region_confirmation", {}).get("roi_sha256") != signature:
        raise ValueError("当前 ROI 尚未确认覆盖完整活动区域，或确认后坐标已变化。请先检查并确认 ROI，停止生成数据。")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"数据目录非空，不能覆盖已有图片或标签：{output}")
    # 先读完五帧，避免最后一帧读取失败时已经产生不完整的数据集。
    frames = [(index, representative_frame(video, index)) for index in FRAME_INDICES]
    for _, frame in frames:
        load_rois(config, frame)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir()
    entries = []
    for index, frame in frames:
        for name, (x1, y1, x2, y2) in rois.items():
            stem = f"frame_{index:06d}_{name}"
            save_frame(frame[y1:y2, x1:x2], output / "images" / f"{stem}.png")
            entries.append({"image": f"images/{stem}.png", "label": f"labels/{stem}.txt",
                            "frame_index": index, "tube": name, "roi_xyxy": [x1, y1, x2, y2],
                            "image_size": [x2 - x1, y2 - y1]})
    manifest = {"source_video": str(Path(video).resolve()), "roi_snapshot": rois,
                "activity_region_confirmation": meta["activity_region_confirmation"],
                "frames": list(FRAME_INDICES), "candidate_count": len(entries),
                "note": "候选图不代表标注完成；labels 中的 txt 仅在人工保存完整标注后生成。", "images": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "dataset.example.yaml").write_text(
        "# 仅展示格式，不是可直接训练的配置；本阶段不建立训练/验证划分。\n"
        "# 将来标注审核完成后，再创建互不重叠的 train.txt 和 val.txt。\n"
        f"path: {json.dumps(output.resolve().as_posix(), ensure_ascii=False)}\n"
        "train: train.txt\nval: val.txt\nnames:\n  0: fly\n", encoding="utf-8")
    print(f"已生成 {len(entries)} 张候选图：{output.resolve()}")
    return manifest


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/tube_rois.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_sample"))
    args = parser.parse_args()
    try:
        build_dataset(args.video, args.config, args.output_dir)
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
