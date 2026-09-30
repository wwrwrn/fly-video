"""ByteTrack baseline审计：统计ID创建与确认，抽取重点视频/连续帧，准备人工检查，不纠正ID。"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detect_full_video import TUBES, open_encoder
from video_utils import save_frame


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, refresh=False):
    root = Path(root)
    if (root / "verification.json").exists() and not refresh:
        raise FileExistsError("分析结果已存在，请不要覆盖已有baseline。")
    report = json.loads((root / "tracking_report.json").read_text(encoding="utf-8"))
    tracks = read_csv(root / "tracks.csv")
    summaries = read_csv(root / "track_summary.csv")
    states = read_csv(root / "track_states.csv")
    events = read_csv(root / "track_events.csv")
    counts = read_csv(root / "active_track_counts.csv")
    detections = read_csv(root / "detections.csv")
    fps = report["source_metadata"]["fps"]
    summary_by_id = {row["track_id"]: row for row in summaries}
    track_frames = defaultdict(list)
    detection_keys = {(int(row["frame_idx"]), row["tube_id"], int(row["detection_index"])): row for row in detections}
    frame_tube_counts = Counter()
    for row in tracks:
        source = int(row["frame_idx"])
        tube = row["tube_id"]
        identity = row["track_id"]
        if not identity.startswith(f"T{tube.split('_')[-1]}-F"):
            raise ValueError("发现跨试管命名空间错误。")
        if summary_by_id[identity]["tube_id"] != tube:
            raise ValueError("同一个全局身份出现在不同试管。")
        track_frames[identity].append(source)
        frame_tube_counts[(source, tube)] += 1
        det = detection_keys[(source, tube, int(row["detection_index"]))]
        if abs(float(row["confidence"]) - float(det["confidence"])) > 1e-6:
            raise ValueError("轨迹与关联检测的置信度不一致。")
        ox, oy, _, _ = report["roi_snapshot"][tube]
        cx = (float(row["x1"]) + float(row["x2"])) / 2
        cy = (float(row["y1"]) + float(row["y2"])) / 2
        if abs(float(row["center_x"]) - cx) > 1e-6 or abs(float(row["global_center_x"]) - cx - ox) > 1e-6:
            raise ValueError("轨迹x坐标转换不一致。")
        if abs(float(row["center_y"]) - cy) > 1e-6 or abs(float(row["global_center_y"]) - cy - oy) > 1e-6:
            raise ValueError("轨迹y坐标转换不一致。")
    observed_frames = defaultdict(list)
    for row in states:
        if int(row["emitted"]) and row["state"] != "tracked":
            raise ValueError("当前有确认输出的轨迹不应被审计为lost或removed。")
        if int(row["has_detection"]):
            observed_frames[row["track_id"]].append(int(row["frame_idx"]))
    for row in summaries:
        identity = row["track_id"]
        observed = observed_frames[identity]
        output_frames = track_frames[identity]
        if len(observed) != len(set(observed)) or len(output_frames) != len(set(output_frames)):
            raise ValueError("同一身份同一帧重复记录。")
        if len(observed) != int(row["number_of_detected_frames"]) or len(output_frames) != int(row["number_of_output_frames"]):
            raise ValueError("生命周期观测数/确认输出数不一致。")
        duration = int(row["end_frame"]) - int(row["start_frame"]) + 1
        gaps = json.loads(row["missing_gaps"])
        if duration != int(row["duration_frames"]) or duration - len(observed) != sum(gap["frames"] for gap in gaps):
            raise ValueError("生命周期跨度与缺口不一致。")
    if len(summaries) != report["total_created_ids"]:
        raise ValueError("创建ID总数与汇总不一致，可能丢失失败轨迹。")
    for tube in TUBES:
        identities = [int(row["track_id"].split("F")[-1]) for row in summaries if row["tube_id"] == tube]
        if sorted(identities) != list(range(1, len(identities) + 1)):
            raise ValueError("存在未被保留的创建ID。")
    if [int(row["frame_idx"]) for row in counts] != list(range(425)):
        raise ValueError("处理帧不连续或数量错误。")
    for row in counts:
        if any(frame_tube_counts[(int(row["frame_idx"]), tube)] != int(row[tube]) for tube in TUBES):
            raise ValueError("活跃数量与tracks不一致。")

    review = root / "review"
    review.mkdir(exist_ok=True)
    # 重点窗口之外多保留少量后续状态，便于判定稍晚的重现是否恢复ID。
    for tube in ["tube_3", "tube_4"]:
        for name, rows in [("states", states), ("tracks", tracks), ("detections", detections), ("events", events)]:
            selected = [row for row in rows if row["tube_id"] == tube and 315 <= int(row["frame_idx"]) <= 360]
            write_csv(review / f"{tube}_frames315_360_{name}.csv", selected, list(rows[0]))
    early_events = [row for row in events if int(row["frame_idx"]) < 30]
    write_csv(review / "first_second_events.csv", early_events, list(events[0]))
    selected_frames = {0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 29, 318, 319, 320, 321, 322, 323, 324, 325, 330, 335, 340, 345}
    capture = cv2.VideoCapture(str(root / "annotated_tracking.mp4"))
    if not capture.isOpened():
        raise ValueError("跟踪结果视频无法打开。")
    decoded = 0
    encoder = None
    with (review / "clip_encoding.log").open("wb") as log:
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if decoded == 0 and not (review / "first_second_tracking.mp4").exists():
                    encoder = open_encoder(review / "first_second_tracking.mp4", image.shape[1], image.shape[0], fps, log)
                if decoded < 30 and encoder:
                    encoder.stdin.write(image.tobytes())
                if decoded in selected_frames:
                    image_path = review / "frames" / f"frame_{decoded:06d}.png"
                    if image_path.exists() and refresh:
                        existing = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if existing is None or not np.array_equal(existing, image):
                            raise ValueError("已有截图与跟踪视频不同，拒绝覆盖。")
                    else:
                        save_frame(image, image_path)
                decoded += 1
            video_metadata = {"frames": decoded, "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": capture.get(cv2.CAP_PROP_FPS)}
        finally:
            capture.release()
            if encoder:
                encoder.stdin.close()
                encoder.wait(timeout=60)
    if decoded != 425 or (encoder and encoder.returncode != 0):
        raise ValueError("结果视频帧数或片段编码不正常。")

    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    figure, axes = plt.subplots(3, 2, figsize=(13, 10))
    for tube, axis in zip(TUBES, axes.flat):
        values = [int(row[tube]) for row in counts]
        axis.plot(np.arange(425) / fps, values, linewidth=1)
        axis.set(title=tube, xlabel="Video time (s)", ylabel="Confirmed active tracks")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(plots / "active_tracks.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(3, 2, figsize=(13, 12))
    for tube, axis in zip(TUBES, axes.flat):
        rows = [row for row in summaries if row["tube_id"] == tube]
        for row in rows:
            number = int(row["track_id"].split("F")[-1])
            observed = observed_frames[row["track_id"]]
            axis.scatter(np.array(observed) / fps, [number] * len(observed), s=1,
                         color="#b93131" if int(row["unconfirmed_only"]) else "#1a739f")
        axis.set(title=f"{tube}: {len(rows)} created IDs", xlabel="Video time (s)", ylabel="Local track number")
        axis.grid(alpha=0.15)
    figure.suptitle("Track observations (red: never confirmed); duration is NOT movement time")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(plots / "track_lifecycles.png", dpi=150)
    plt.close(figure)
    findings = {"processed_frames": 425, "all_created_ids_preserved": True, "all_tube_namespaces_independent": True,
                "tracks_csv_rows": len(tracks), "all_output_tracks_have_current_detections": True,
                "observed_frames_gaps_and_lifecycles_validated": True, "video": video_metadata,
                "review_frames_saved": len(selected_frames), "first_second_clip_frames": 30,
                "first_second_created_ids_including_frame0": sum(row["event"] == "created" for row in early_events),
                "first_second_created_after_frame0": sum(row["event"] == "created" and int(row["frame_idx"]) > 0 for row in early_events),
                "reactivation_events": sum(row["event"] == "reactivated" for row in events),
                "warning": "ID switch需要连续画面或身份真值审查；创建ID和重现不自动等于ID switch。"}
    (root / "verification.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(findings, ensure_ascii=False, indent=2))
    return findings


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/bytetrack_baseline"))
    parser.add_argument("--refresh", action="store_true", help="仅刷新衍生统计和审计，不修改tracks或视频")
    args = parser.parse_args()
    analyze(args.input, args.refresh)
