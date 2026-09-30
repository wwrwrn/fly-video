"""只分析检测数量与框重叠，不匹配个体；生成统计、曲线和最多30张异常前后帧截图。"""

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from qa_training_labels import contact_sheet, overlap
from video_utils import save_frame

TUBES = [f"tube_{i}" for i in range(1, 7)]
JUMP_THRESHOLD = 3
MAX_SCREENSHOTS = 30


def true_runs(mask, indices, times, fps, minimum=3):
    """找连续满足条件的源帧区间；不会跨越读取失败的帧拼接区间。"""
    runs, start = [], None
    for i in range(len(mask) + 1):
        active = i < len(mask) and bool(mask[i])
        gap = i < len(mask) and i > 0 and indices[i] != indices[i - 1] + 1
        if start is not None and (not active or gap):
            if i - start >= minimum:
                runs.append({"start_frame": int(indices[start]), "end_frame": int(indices[i - 1]),
                             "start_sec": float(times[start]), "end_sec": float(times[i - 1]),
                             "frames": i - start, "duration_sec": float(times[i - 1] - times[start] + 1 / fps)})
            start = None
        if active and start is None:
            start = i
    return runs


def analyze(output):
    output = Path(output)
    report_path = output / "stability_report.json"
    if report_path.exists():
        raise FileExistsError("统计报告已存在，请保留结果，或在新的检测输出目录分析。")
    performance = json.loads((output / "performance.json").read_text(encoding="utf-8"))
    fps = performance["source_metadata"]["fps"]
    with (output / "detection_counts.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("数量 CSV 为空。")
    indices = np.array([int(row["frame_idx"]) for row in rows])
    times = np.array([float(row["time_sec"]) for row in rows])
    count_matrix = np.array([[int(row[tube]) for tube in TUBES] for row in rows])
    if len(rows) != performance["processed_frames"] or not np.all(np.diff(indices) > 0):
        raise ValueError("CSV 帧数或源帧顺序不一致。")
    if not np.array_equal(count_matrix.sum(axis=1), [int(row["total"]) for row in rows]):
        raise ValueError("各试管数量之和与 total 不符。")
    grouped = defaultdict(list)
    detail_count = 0
    with (output / "full_video_detections.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[(int(row["frame_idx"]), row["tube_id"])].append(row)
            detail_count += 1
    for row in rows:
        for tube in TUBES:
            if len(grouped[(int(row["frame_idx"]), tube)]) != int(row[tube]):
                raise ValueError(f"检测明细与数量不一致：frame={row['frame_idx']} {tube}")
    if detail_count != performance["detection_rows"]:
        raise ValueError("检测明细总行数不符。")

    deltas = np.zeros_like(count_matrix)
    deltas[1:] = count_matrix[1:] - count_matrix[:-1]
    consecutive = np.r_[False, np.diff(indices) == 1]
    deltas[~consecutive] = 0
    events, duplicate_pairs, statistics = [], [], {}
    candidates = {}

    def nominate(frame, score, reason):
        entry = candidates.setdefault(int(frame), {"frame_idx": int(frame), "score": 0.0, "reasons": []})
        entry["score"] = max(entry["score"], float(score))
        if reason not in entry["reasons"]:
            entry["reasons"].append(reason)

    # 同一帧的框重叠只做疑似重复提示，不额外抑制或删除任何检测。
    for (frame, tube), detections in grouped.items():
        for i, item in enumerate(detections):
            box = [float(item[key]) for key in ["x1", "y1", "x2", "y2"]]
            for previous in detections[:i]:
                other = [float(previous[key]) for key in ["x1", "y1", "x2", "y2"]]
                iou = overlap(box, other)
                if iou >= 0.5:
                    duplicate_pairs.append({"frame_idx": frame, "time_sec": float(item["time_sec"]), "tube_id": tube,
                                            "detection_a": int(previous["detection_index"]),
                                            "detection_b": int(item["detection_index"]), "iou": iou})
                    nominate(frame, 2 + iou, f"{tube}: overlapping detections IoU={iou:.2f}")

    for column, tube in enumerate(TUBES):
        counts = count_matrix[:, column]
        for i in np.where(np.abs(deltas[:, column]) >= JUMP_THRESHOLD)[0]:
            event = {"frame_idx": int(indices[i]), "time_sec": float(times[i]), "tube_id": tube,
                     "previous_count": int(counts[i - 1]), "current_count": int(counts[i]),
                     "change": int(deltas[i, column]), "type": "drop" if deltas[i, column] < 0 else "increase"}
            events.append(event)
            nominate(indices[i], abs(deltas[i, column]) + 3, f"{tube}: {event['previous_count']} -> {event['current_count']}")
        # 对比过去约1秒的中位数，找连续低检出区间，避免假定真实数量始终固定。
        low = np.zeros(len(counts), dtype=bool)
        history_length, minimum_history = max(1, round(fps)), max(1, round(fps / 2))
        for i in range(minimum_history, len(counts)):
            history = counts[max(0, i - history_length):i]
            low[i] = counts[i] <= np.median(history) - JUMP_THRESHOLD
        low_runs = true_runs(low, indices, times, fps)
        zero_runs = true_runs(counts == 0, indices, times, fps)
        for run in zero_runs + low_runs:
            nominate(run["start_frame"], 4, f"{tube}: sustained low/zero detections for {run['frames']} frames")
        duplicates = [pair for pair in duplicate_pairs if pair["tube_id"] == tube]
        statistics[tube] = {"mean": float(np.mean(counts)), "min": int(np.min(counts)), "max": int(np.max(counts)),
                            "std_population": float(np.std(counts, ddof=0)),
                            "mean_absolute_adjacent_change": float(np.mean(np.abs(deltas[consecutive, column]))) if np.any(consecutive) else 0.0,
                            "abrupt_drop_events": sum(event["tube_id"] == tube and event["type"] == "drop" for event in events),
                            "abrupt_increase_events": sum(event["tube_id"] == tube and event["type"] == "increase" for event in events),
                            "zero_detection_runs": zero_runs, "sustained_low_count_runs": low_runs,
                            "suspect_overlap_pair_count": len(duplicates),
                            "frames_with_suspect_overlap": len({pair["frame_idx"] for pair in duplicates})}

    windows = []
    for second in sorted(set(np.floor(times).astype(int))):
        mask = (times >= second) & (times < second + 1)
        positions = np.where(mask)[0]
        change_mask = mask & consecutive
        per_tube = {tube: float(np.mean(np.abs(deltas[change_mask, column]))) if np.any(change_mask) else 0.0
                    for column, tube in enumerate(TUBES)}
        windows.append({"start_sec": float(times[positions[0]]), "end_sec": float(times[positions[-1]] + 1 / fps),
                        "frames": len(positions), "mean_absolute_step_per_tube": per_tube,
                        "instability_score": float(np.mean(list(per_tube.values()))),
                        "abrupt_change_events": int(sum(second <= event["time_sec"] < second + 1 for event in events))})
    top_windows = sorted((window for window in windows if window["frames"] >= round(fps / 2)),
                         key=lambda window: window["instability_score"], reverse=True)[:5]

    plots = output / "plots"
    plots.mkdir(exist_ok=True)

    def plot_one(axis, column):
        axis.plot(times, count_matrix[:, column], color="#176a99", linewidth=0.9)
        down = deltas[:, column] <= -JUMP_THRESHOLD
        up = deltas[:, column] >= JUMP_THRESHOLD
        axis.scatter(times[down], count_matrix[down, column], c="#c43232", s=14, label="drop >=3", zorder=3)
        axis.scatter(times[up], count_matrix[up, column], c="#138a59", s=14, label="rise >=3", zorder=3)
        axis.set(title=TUBES[column], xlabel="Time (s)", ylabel="Number of detections")
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        axis.set_ylim(bottom=0)

    for column, tube in enumerate(TUBES):
        figure, axis = plt.subplots(figsize=(11, 3.5))
        plot_one(axis, column)
        axis.legend(loc="upper right")
        figure.tight_layout()
        figure.savefig(plots / f"{tube}_counts.png", dpi=160)
        plt.close(figure)
    figure = plt.figure(figsize=(14, 13))
    grid = figure.add_gridspec(4, 2)
    for column in range(6):
        plot_one(figure.add_subplot(grid[column // 2, column % 2]), column)
    axis = figure.add_subplot(grid[3, :])
    axis.plot(times, count_matrix.sum(axis=1), color="#44346e", linewidth=1)
    axis.set(title="Total detections across six tubes", xlabel="Time (s)", ylabel="Number of detections")
    axis.grid(alpha=0.2)
    figure.suptitle("YOLO26n detection counts | conf=0.25 | count changes are review alerts, not confirmed errors", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(plots / "detection_counts_overview.png", dpi=160)
    plt.close(figure)

    def write_csv(path, entries, columns):
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(entries)

    write_csv(output / "count_change_alerts.csv", sorted(events, key=lambda event: (event["frame_idx"], event["tube_id"])),
              ["frame_idx", "time_sec", "tube_id", "previous_count", "current_count", "change", "type"])
    write_csv(output / "suspect_overlapping_detections.csv", duplicate_pairs,
              ["frame_idx", "time_sec", "tube_id", "detection_a", "detection_b", "iou"])

    # 选最多10个相隔至少约0.25秒的事件，每个事件保存前/中/后三帧，总数不超过30。
    selected = []
    for candidate in sorted(candidates.values(), key=lambda item: (-item["score"], item["frame_idx"])):
        if all(abs(candidate["frame_idx"] - other["frame_idx"]) >= max(3, round(fps / 4)) for other in selected):
            selected.append(candidate)
            if len(selected) == MAX_SCREENSHOTS // 3:
                break
    source_to_video = {int(source): i for i, source in enumerate(indices)}
    wanted = sorted({frame for item in selected for frame in [item["frame_idx"] - 1, item["frame_idx"], item["frame_idx"] + 1]
                     if frame in source_to_video})[:MAX_SCREENSHOTS]
    selected_video_indices = {source_to_video[frame]: frame for frame in wanted}
    screenshot_dir = output / "anomaly_frames"
    screenshot_dir.mkdir(exist_ok=True)
    screenshots = []
    capture = cv2.VideoCapture(str(output / "annotated_detection.mp4"))
    if not capture.isOpened():
        raise ValueError("无法打开已编码的结果视频以抽取异常帧。")
    try:
        for video_index in range(len(indices)):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"输出视频在第 {video_index} 帧解码失败。")
            if video_index in selected_video_indices:
                source = selected_video_indices[video_index]
                path = screenshot_dir / f"frame_{source:06d}.png"
                # 统计中断后允许复用完全相同的已生成截图，不覆盖不同内容。
                if path.exists():
                    existing = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if existing is None or not np.array_equal(existing, image):
                        raise FileExistsError(f"已有截图与视频不一致，拒绝覆盖：{path}")
                else:
                    save_frame(image, path)
                screenshots.append({"frame_idx": source, "time_sec": float(times[video_index]),
                                    "encoded_video_index": video_index, "path": str(path.resolve())})
        if capture.read()[0]:
            raise ValueError("输出视频帧数多于实际处理帧数。")
        video_verification = {"decoded_output_frames": len(indices), "reported_fps": capture.get(cv2.CAP_PROP_FPS),
                              "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        capture.release()
    manifest = {"selection_rule": "最大的数量跳变、持续低检出或高重叠候选；事件间隔至少约0.25秒，保存相邻三帧。",
                "selected_events": selected, "screenshots": screenshots, "maximum_screenshots": MAX_SCREENSHOTS}
    (output / "anomaly_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"processed_frames": len(indices), "detection_rows": detail_count,
              "definitions": {"jump_threshold": JUMP_THRESHOLD, "std": "population (ddof=0)",
                              "low_count": "至少连续3帧，比过去约1秒中位数低至少3；至少需0.5秒历史。",
                              "overlap_candidate": "同帧同管IoU>=0.5；可能为重复，也可能是真实相互遮挡，不自动删除。",
                              "unstable_window": "1秒区间中，相邻帧各管数量绝对变化的平均值；排名排除不足0.5秒的末尾区间。"},
              "tube_statistics": statistics, "most_unstable_windows": top_windows, "all_windows": windows,
              "count_change_alerts": len(events), "suspect_overlap_pairs": len(duplicate_pairs),
              "screenshots_saved": len(screenshots), "output_video_verification": video_verification,
              "limits": "数量下降或零检出不等于确认漏检；无逐帧完整真值，不能由数量推断隐藏个体或恢复身份。"}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["processed_frames", "detection_rows", "tube_statistics", "most_unstable_windows", "screenshots_saved", "output_video_verification"]}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/full_video_detection"))
    args = parser.parse_args()
    analyze(args.input)
