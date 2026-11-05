"""从六管单蝇视频自动生成检测训练集：背景差分定框，单目标与时间连续性过滤，按实验组留出测试集。

伪标签不是人工真值：只保留"恰好一个合理暗斑且前后位置连续"的帧，不确定的帧直接丢弃，绝不当负样本。
"""

import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np

from single_fly_geometry import TranslationRegistration, auto_calibrate, board_mask, dark_detail, load_manual_geometry, shifted, static_detail
from single_fly_tracker import REFERENCE_PIXELS_PER_CM
from video_utils import open_video, video_info

ROOT = Path(__file__).resolve().parent
MANUAL_DIR = ROOT / "config" / "manual_geometry"
TEST_GROUPS = {"60743 雌", "W118 雄"}
VAL_EVERY = 5
SAMPLE_STEP = 3
BACKGROUND_FRAMES = 41
PER_TUBE_LIMIT = 14
# 按高度分箱，保证底部、中段、线上方都有样本，不被静止帧占满。
HEIGHT_BINS = 6
# 成虫体长约0.3 cm；更长的暗斑是横线片段或白板边，不是果蝇。
MAX_FLY_LENGTH_CM = 0.42


def imwrite(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"图像编码失败：{path}")
    path.write_bytes(encoded.tobytes())


def read_video(path, reference):
    """顺序解码并配准；只保留抽样帧的灰度和彩色，缺帧不补。"""
    registration = TranslationRegistration(reference)
    frames = []
    with open_video(path) as capture:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            dx, dy, _, valid = registration.update(frame)
            if valid and index % SAMPLE_STEP == 0:
                frames.append((index, shifted(frame, dx, dy)))
            index += 1
    return frames


def find_fly(difference, scale, longest):
    """差分图中恰好一个果蝇大小的暗斑才返回框；多个大斑、没有斑或被ROI切断都放弃。"""
    _, mask = cv2.threshold(difference, 14, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    low, high = 22 * scale ** 2, 420 * scale ** 2
    rows, columns = difference.shape
    blobs = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        inside = x > 0 and y > 0 and x + width < columns and y + height < rows
        if inside and low <= area <= high and max(width, height) <= longest and max(width, height) <= 4.5 * max(1, min(width, height)):
            blobs.append((int(area), int(x), int(y), int(width), int(height), *map(float, centroids[label])))
    if len(blobs) != 1:
        return None
    # 另有接近大小的未通过形状检查的斑块时，也不确定。
    if (stats[1:, cv2.CC_STAT_AREA] > max(low, blobs[0][0] * 0.5)).sum() > 1:
        return None
    return blobs[0]


def label_video(path, geometry, frames, reference):
    scale = geometry["pixels_per_cm"] / REFERENCE_PIXELS_PER_CM
    board = cv2.erode(board_mask(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    longest = MAX_FLY_LENGTH_CM * geometry["pixels_per_cm"]
    size = max(15, round(31 * scale)) | 1
    details = [dark_detail(frame, size) for _, frame in frames]
    picks = np.linspace(0, len(details) - 1, min(BACKGROUND_FRAMES, len(details))).round().astype(int)
    background = static_detail([details[i] for i in picks])
    # 彩色空管图用中位数：去掉偶然经过的果蝇，亮度仍是真实帧的水平。
    colour = np.median(np.stack([frames[i][1] for i in picks[::4]]), axis=0).astype(np.uint8)
    samples, negatives = [], []
    pad = max(2, round(3 * scale))
    gate = 0.6 * geometry["pixels_per_cm"]
    for tube, item in geometry["tubes"].items():
        x1, y1, x2, y2 = item["roi"]
        track = []
        for (index, frame), detail in zip(frames, details):
            difference = np.clip(detail[y1:y2, x1:x2] - background[y1:y2, x1:x2], 0, 255).astype(np.uint8)
            difference[~board[y1:y2, x1:x2]] = 0
            found = find_fly(difference, scale, longest)
            if found:
                _, x, y, width, height, cx, cy = found
                box = [max(0, x - pad), max(0, y - pad), min(x2 - x1, x + width + pad), min(y2 - y1, y + height + pad)]
                track.append({"index": index, "frame": frame, "box": box, "cx": cx, "cy": cy})
        kept = []
        for i, point in enumerate(track):
            neighbours = [track[j] for j in (i - 1, i + 1) if 0 <= j < len(track)]
            near = [n for n in neighbours if abs(n["index"] - point["index"]) <= SAMPLE_STEP * 4
                    and np.hypot(n["cx"] - point["cx"], n["cy"] - point["cy"]) <= gate]
            if near:
                kept.append(point)
        bins = {}
        for point in kept:
            bins.setdefault(int(point["cy"] / (y2 - y1) * HEIGHT_BINS), []).append(point)
        chosen = []
        rng = random.Random(f"{path.stem}-{tube}")
        for group in bins.values():
            rng.shuffle(group)
        while len(chosen) < PER_TUBE_LIMIT and any(bins.values()):
            for group in bins.values():
                if group and len(chosen) < PER_TUBE_LIMIT:
                    chosen.append(group.pop())
        for point in chosen:
            samples.append({"tube": tube, "index": point["index"], "box": point["box"],
                            "image": point["frame"][y1:y2, x1:x2].copy(), "detected": len(track), "kept": len(kept)})
        negatives.append({"tube": tube, "image": colour[y1:y2, x1:x2].copy()})
    return samples, negatives


def write_item(output, split, stem, image, boxes):
    imwrite(output / "images" / split / f"{stem}.jpg", image)
    height, width = image.shape[:2]
    lines = [f"0 {(a + c) / 2 / width:.6f} {(b + d) / 2 / height:.6f} {(c - a) / width:.6f} {(d - b) / height:.6f}"
             for a, b, c, d in boxes]
    (output / "labels" / split / f"{stem}.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def split_for(group, position):
    if group in TEST_GROUPS:
        return "test"
    return "val" if position % VAL_EVERY == VAL_EVERY - 1 else "train"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, help="按“品系 性别”分子目录的视频文件夹")
    parser.add_argument("--extra", type=Path, help="九月旧单蝇视频文件夹，仅并入训练")
    parser.add_argument("--human-dataset", type=Path, default=ROOT / "dataset_sample", help="旧人工标注，全部并入训练")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_v2")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空：{args.output}")
    for folder in ("images", "labels"):
        for split in ("train", "val", "test"):
            (args.output / folder / split).mkdir(parents=True, exist_ok=True)
    videos = []
    for group_dir in sorted(p for p in args.videos.iterdir() if p.is_dir()):
        for position, video in enumerate(sorted(group_dir.glob("*.mp4"))):
            videos.append((video, group_dir.name, split_for(group_dir.name, position)))
    if args.extra:
        videos.extend((video, "九月旧视频", "train") for video in sorted(args.extra.glob("*.mp4")))
    manifest, failures = [], []
    for number, (video, group, split) in enumerate(videos, 1):
        try:
            manual = MANUAL_DIR / f"{video.stem}.json"
            reference, geometry = load_manual_geometry(video, manual) if manual.is_file() else auto_calibrate(video)
            frames = read_video(video, reference)
            samples, negatives = label_video(video, geometry, frames, reference)
        except (ValueError, cv2.error) as exc:
            failures.append({"video": str(video), "reason": str(exc)[:300]})
            print(f"[{number}/{len(videos)}] 跳过 {video.stem}：{str(exc)[:120]}", flush=True)
            continue
        for item in samples:
            stem = f"{video.stem}_{item['tube']}_{item['index']:06d}"
            write_item(args.output, split, stem, item["image"], [item["box"]])
            manifest.append({"stem": stem, "split": split, "group": group, "video": video.stem, "tube": item["tube"],
                             "frame": item["index"], "source": "pseudo", "pixels_per_cm": geometry["pixels_per_cm"],
                             "box": json.dumps(item["box"])})
        for item in negatives:
            stem = f"{video.stem}_{item['tube']}_empty"
            write_item(args.output, split, stem, item["image"], [])
            manifest.append({"stem": stem, "split": split, "group": group, "video": video.stem, "tube": item["tube"],
                             "frame": -1, "source": "background", "pixels_per_cm": geometry["pixels_per_cm"], "box": "[]"})
        print(f"[{number}/{len(videos)}] {group}/{video.stem} {split}: {len(samples)} 张", flush=True)
    if args.human_dataset and args.human_dataset.is_dir():
        for label in sorted((args.human_dataset / "labels").glob("*.txt")):
            image = args.human_dataset / "images" / f"{label.stem}.png"
            shutil.copyfile(image, args.output / "images" / "train" / image.name)
            shutil.copyfile(label, args.output / "labels" / "train" / label.name)
            manifest.append({"stem": label.stem, "split": "train", "group": "人工多蝇", "video": "20260817_202618",
                             "tube": "", "frame": -1, "source": "human", "pixels_per_cm": "", "box": ""})
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (args.output / "dataset.yaml").write_text(
        f"path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fly\n",
        encoding="utf-8")
    counts = {split: sum(row["split"] == split for row in manifest) for split in ("train", "val", "test")}
    summary = {"images": counts, "failed_videos": failures, "test_groups": sorted(TEST_GROUPS),
               "note": "pseudo 为背景差分自动标签，未经人工逐张确认；background 为去除果蝇后的空管图。"}
    (args.output / "build_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
