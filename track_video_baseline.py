"""第4阶段：六管独立ByteTrack，保存带ID视频、短历史线、检测/轨迹/生命周期审计，不计算运动指标。"""

import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from detect_full_video import ROOT, TUBES, open_encoder
from roi_utils import draw_rois, load_rois
from video_utils import open_video, video_info


def format_row(row):
    return {key: f"{value:.9f}" if isinstance(value, float) else json.dumps(value) if isinstance(value, list) else value
            for key, value in row.items()}


def track_color(track_id):
    value = int(track_id.split("F")[-1])
    # 稳定颜色仅辅助阅读，不是额外身份判断。
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[value * 37 % 180, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def draw_tracking(frame, rois, visible, histories, index, fps):
    image = draw_rois(frame, rois)
    cv2.rectangle(image, (0, 0), (image.shape[1], 60), (25, 25, 25), -1)
    cv2.putText(image, f"ByteTrack | frame {index} | t={index/fps:.3f}s | active={len(visible)} | history 1.5s",
                (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    keep = round(fps * 1.5)
    for trail in histories.values():
        while trail and trail[0][0] < index - keep + 1:
            trail.popleft()
    for row in visible:
        track_id = row["track_id"]
        color = track_color(track_id)
        ox, oy, _, _ = rois[row["tube_id"]]
        x1, y1, x2, y2 = [round(row[key] + offset) for key, offset in [("x1", ox), ("y1", oy), ("x2", ox), ("y2", oy)]]
        trail = histories[track_id]
        trail.append((index, round(row["global_center_x"]), round(row["global_center_y"])))
        for before, after in zip(trail, list(trail)[1:]):
            # 丢失期间不插值、不连直线伪造运动路径。
            if after[0] == before[0] + 1:
                cv2.line(image, before[1:], after[1:], color, 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        text = f"{track_id} {row['confidence']:.2f}"
        point = (max(0, min(x1, image.shape[1] - 145)), max(15, y1 - 3))
        cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3)
        cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    return image


def run(video, roi_path, weights, output):
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    from ultralytics.engine.results import Boxes
    from tube_bytetrack import TubeByteTrack

    video, roi_path, weights, output = map(Path, [video, roi_path, weights, output])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空：{output}")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，停止，不静默更换实验设备。")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    config_path = Path(ultralytics.__file__).parent / "cfg/trackers/bytetrack.yaml"
    parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    (output / "bytetrack_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    model = YOLO(str(weights), task="detect")
    if model.names != {0: "fly"}:
        raise ValueError("模型类别必须为0: fly。")
    owners = {tube: TubeByteTrack(i, parameters) for i, tube in enumerate(TUBES, 1)}
    histories = defaultdict(deque)
    track_fields = ["frame_idx", "time_sec", "tube_id", "track_id", "confidence", "x1", "y1", "x2", "y2",
                    "center_x", "center_y", "global_center_x", "global_center_y", "detection_index", "association_tier",
                    "detection_x1", "detection_y1", "detection_x2", "detection_y2"]
    detection_fields = ["frame_idx", "time_sec", "tube_id", "detection_index", "confidence", "x1", "y1", "x2", "y2", "tier"]
    state_fields = ["frame_idx", "tube_id", "track_id", "state", "emitted", "has_detection", "last_detection_frame",
                    "missing_since_last_detection", "last_detection_confidence", "x1", "y1", "x2", "y2"]
    active_rows = []
    processed, output_rows, low_matches = 0, 0, 0
    with open_video(video) as capture:
        info = video_info(capture)
        fps = info["fps"]
        ok, first = capture.read()
        if not ok:
            raise ValueError("首帧无法读取。")
        rois = load_rois(roi_path, first)
        for x1, y1, x2, y2 in rois.values():
            model.predict(first[y1:y2, x1:x2], conf=0.1, imgsz=640, iou=0.7, rect=True,
                          device=0, max_det=100, verbose=False, save=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        from contextlib import ExitStack
        with ExitStack() as stack:
            writers = {}
            for name, fields in [("tracks", track_fields), ("detections", detection_fields), ("track_states", state_fields)]:
                stream = stack.enter_context((output / f"{name}.csv").open("w", newline="", encoding="utf-8-sig"))
                writers[name] = csv.DictWriter(stream, fieldnames=fields)
                writers[name].writeheader()
            log = stack.enter_context((output / "encoding.log").open("wb"))
            encoder = open_encoder(output / "annotated_tracking.mp4", first.shape[1], first.shape[0], fps, log)
            try:
                for index in range(425):
                    ok, frame = (True, first) if index == 0 else capture.read()
                    if not ok:
                        raise ValueError(f"源frame {index}读取失败；不能把跳帧伪装成连续跟踪。")
                    visible = []
                    active = {"frame_idx": index, "time_sec": index / fps}
                    for tube in TUBES:
                        ox, oy, ex, ey = rois[tube]
                        result = model.predict(frame[oy:ey, ox:ex], conf=0.1, imgsz=640, iou=0.7,
                                               rect=True, device=0, max_det=100, verbose=False, save=False)[0]
                        raw = result.boxes.cpu().numpy()
                        tracks, states = owners[tube].update(raw, index)
                        for row in states:
                            writers["track_states"].writerow(format_row(row))
                        for detection_index, detection in enumerate(raw.data, 1):
                            x1, y1, x2, y2, score, cls = map(float, detection)
                            if cls != 0:
                                raise ValueError("检测类别不是fly。")
                            writers["detections"].writerow(format_row(dict(frame_idx=index, time_sec=index / fps,
                                tube_id=tube, detection_index=detection_index, confidence=score, x1=x1, y1=y1, x2=x2, y2=y2,
                                tier="high" if score >= parameters["track_high_thresh"] else "low")))
                        active[tube] = len(tracks)
                        for values in tracks:
                            x1, y1, x2, y2, native_id, confidence, cls, det_index = map(float, values)
                            detection = raw.data[int(det_index)]
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            tier = "high" if confidence >= parameters["track_high_thresh"] else "low"
                            row = dict(frame_idx=index, time_sec=index / fps, tube_id=tube,
                                       track_id=owners[tube].name(native_id), confidence=confidence,
                                       x1=x1, y1=y1, x2=x2, y2=y2, center_x=cx, center_y=cy,
                                       global_center_x=cx + ox, global_center_y=cy + oy,
                                       detection_index=int(det_index) + 1, association_tier=tier,
                                       **{f"detection_{key}": float(value) for key, value in zip(["x1", "y1", "x2", "y2"], detection[:4])})
                            writers["tracks"].writerow(format_row(row))
                            visible.append(row)
                            low_matches += tier == "low"
                    active["total"] = len(visible)
                    active_rows.append(active)
                    encoder.stdin.write(draw_tracking(frame, rois, visible, histories, index, fps).tobytes())
                    processed += 1
                    output_rows += len(visible)
                    if processed % 50 == 0:
                        print(f"已跟踪{processed}/425帧，累计创建ID：" + str({tube: owner.next_number for tube, owner in owners.items()}), flush=True)
            finally:
                try:
                    encoder.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    code = encoder.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
                    raise RuntimeError("编码器关闭超时。")
            if code:
                raise RuntimeError("编码失败，见encoding.log。")
        seconds = time.perf_counter() - started

    def write_csv(name, rows, fields=None):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
            writer.writeheader()
            writer.writerows(format_row(row) for row in rows)

    summaries = [row for owner in owners.values() for row in owner.summaries(fps, processed - 1)]
    events = sorted([event for owner in owners.values() for event in owner.events], key=lambda row: (row["frame_idx"], row["tube_id"], row["track_id"]))
    write_csv("track_summary.csv", summaries)
    write_csv("track_events.csv", events)
    write_csv("active_track_counts.csv", active_rows)
    statistics = {}
    for tube, owner in owners.items():
        rows = [row for row in summaries if row["tube_id"] == tube]
        counts = [row[tube] for row in active_rows]
        longest = max((row["duration_frames"] for row in rows), default=0)
        shortest = min((row["duration_frames"] for row in rows), default=0)
        statistics[tube] = {"all_created_ids": owner.next_number, "confirmed_ids": sum(not row["unconfirmed_only"] for row in rows),
                            "unconfirmed_only_ids": sum(row["unconfirmed_only"] for row in rows),
                            "active_mean": float(np.mean(counts)), "active_min": min(counts), "active_max": max(counts),
                            "longest_duration_frames": longest, "longest_tracks": [row["track_id"] for row in rows if row["duration_frames"] == longest],
                            "shortest_duration_frames": shortest, "short_tracks_le_5_frames": sum(row["short_track_le_5_frames"] for row in rows),
                            "sparse_tracks_le_5_observations": sum(row["number_of_detected_frames"] <= 5 for row in rows),
                            "created_in_frames_0_29": sum(row["start_frame"] < 30 for row in rows),
                            "confirmed_in_frames_0_29": sum(0 <= row["first_confirmed_output_frame"] < 30 for row in rows)}
    report = {"processed_frames": processed, "source_frames": [0, processed - 1], "source_metadata": info,
              "weights": str(weights.resolve()), "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "roi_snapshot": rois,
              "implementation": "Ultralytics BYTETracker, unmodified association; per-tube counter + audit adapter",
              "ultralytics": ultralytics.__version__, "bytetrack_parameters": parameters,
              "detector_parameters": {"confidence_floor": 0.1, "imgsz": 640, "iou": 0.7, "max_det": 100, "rect": True, "device": 0},
              "history_seconds": 1.5, "tracks_csv_rows": output_rows, "low_confidence_matched_rows": low_matches,
              "tube_statistics": statistics, "total_created_ids": sum(owner.next_number for owner in owners.values()),
              "total_short_tracks_le_5_frames": sum(row["short_track_le_5_frames"] for row in summaries),
              "pipeline_seconds": seconds, "pipeline_fps": processed / seconds,
              "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
              "duration_definition": "首个匹配检测到最后匹配检测的帧跨度，含内部缺口；不含末尾lost缓冲等待；不是运动时间。",
              "tracks_csv_definition": "只含本帧有检测支持且已经确认的轨迹；xyxy/center为ByteTrack更新后框，另附原检测框。",
              "states_csv_definition": "含tentative/tracked/lost/removed/pruned_native；lost框为预测状态，未写作tracks中的观测。",
              "failure_preservation": "所有activate创建过的ID都进入summary，包括未确认、短命以及官方内部去重移除的轨迹。"}
    (output / "tracking_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/tube_rois.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/detect/fly_yolo26n_teaching/weights/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bytetrack_baseline")
    args = parser.parse_args()
    run(args.video, args.config, args.weights, args.output)
