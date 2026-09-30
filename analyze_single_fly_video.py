"""六管各一只果蝇：自动定位六管/背景线，固定身份逐帧检测，记录首次过线候选与轨迹。"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from detect_full_video import ROOT, open_encoder
from single_fly_geometry import TranslationRegistration, auto_calibrate, draw_geometry
from single_fly_tracker import SingleFlyTracker
from video_utils import open_video, save_frame, video_info


def fmt(row):
    return {key: f"{value:.9f}" if isinstance(value, float) else value for key, value in row.items()}


def annotate(frame, geometry, registration, states, trackers, index, time_sec):
    dx, dy = registration
    result = draw_geometry(frame, geometry, dx, dy)
    cv2.rectangle(result, (0, 0), (result.shape[1], 62), (25, 25, 25), -1)
    cv2.putText(result, f"Single fly per tube | frame {index} | t={time_sec:.3f}s | v*=5.5cm / first-passage time",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    for tube, state in states.items():
        tracker = trackers[tube]
        item = geometry["tubes"][tube]
        x1, y1, _, _ = item["roi"]
        origin = (round(x1 + dx + 5), round(y1 + dy + 28))
        speed_text = None
        if tracker.crossing is not None:
            quality = tracker.result(geometry["finish_line"]["timing_quality"])
            text = f"{tracker.track_id} @ {tracker.crossing['time_sec']:.2f}s"
            speed_text = f"v*={quality['candidate_speed_cm_s']:.2f}cm/s " + ("CHECK" if quality["speed_quality"] == "review" else "candidate")
            color = (0, 210, 210) if quality["speed_quality"] == "review" else (0, 210, 0)
        elif state["selected"] is not None:
            score = state["selected"]["confidence"]
            text = f"{tracker.track_id} {score:.2f}"
            color = (255, 80, 0)
        else:
            text = f"{tracker.track_id} {state['status']}"
            color = (0, 155, 255)
        cv2.putText(result, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(result, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if speed_text:
            speed_origin = (origin[0], origin[1] + 20)
            cv2.putText(result, speed_text, speed_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(result, speed_text, speed_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        point = state["selected"]
        if point is not None:
            box = point["box"]
            cv2.rectangle(result, (round(box[0] + x1 + dx), round(box[1] + y1 + dy)),
                          (round(box[2] + x1 + dx), round(box[3] + y1 + dy)), color, 2)
        recent = [p for p in tracker.points if p["frame_idx"] >= index - 60]
        for before, after in zip(recent, recent[1:]):
            if after["frame_idx"] == before["frame_idx"] + 1:
                cv2.line(result, (round(before["cx"] + dx), round(before["cy"] + dy)),
                         (round(after["cx"] + dx), round(after["cy"] + dy)), color, 1)
    return result


def resolve_device(requested, torch):
    """默认优先使用可实际运算的CUDA；没有可用GPU时允许CPU离线运行。"""
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            probe = torch.ones((16, 16), device="cuda:0")
            _ = probe @ probe
            torch.cuda.synchronize()
            return "0"
        except RuntimeError as exc:
            if requested == "0":
                raise RuntimeError(f"显式指定的CUDA无法运算：{exc}") from exc
            print("CUDA运行检查未通过，本次改用CPU（速度较慢）。", flush=True)
    elif requested == "0":
        raise RuntimeError("显式指定了GPU，但CUDA不可用。请使用--device auto或cpu。")
    return "cpu"


def run(video, weights, output, limit=None, device="auto"):
    import torch
    from ultralytics import YOLO

    video, weights, output = Path(video), Path(weights), Path(output)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空：{output}")
    selected_device = resolve_device(device, torch)
    print(f"运行设备：{'CUDA:0' if selected_device == '0' else 'CPU（较慢）'}", flush=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    reference, geometry = auto_calibrate(video)
    registration = TranslationRegistration(reference)
    model = YOLO(str(weights), task="detect")
    if model.names != {0: "fly"}:
        raise ValueError("当前模型不是单类别fly权重。")
    tubes = list(geometry["tubes"])
    trackers = {tube: SingleFlyTracker(i, item["bottom_y_reference"],
                lambda x, line=geometry["finish_line"]: line["slope"] * x + line["intercept"])
                for i, (tube, item) in enumerate(geometry["tubes"].items(), 1)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "automatic_geometry.json").write_text(json.dumps(geometry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_frame(draw_geometry(reference, geometry), output / "automatic_geometry_preview.png")
    selected_fields = ["frame_idx", "time_sec", "tube_id", "track_id", "status", "candidate_count", "confidence",
                       "x1", "y1", "x2", "y2", "roi_center_x", "roi_center_y", "global_center_x", "global_center_y",
                       "reference_center_x", "reference_center_y", "signed_line_distance", "camera_dx", "camera_dy", "registration_valid"]
    detection_fields = ["frame_idx", "time_sec", "tube_id", "detection_index", "confidence", "x1", "y1", "x2", "y2"]
    count_fields = ["frame_idx", "time_sec", *tubes, "total", "ambiguous_tubes", "finished_tubes"]
    processed = 0
    decoded_pts_invalid = 0
    registration_failures = 0
    start = time.perf_counter()
    with open_video(video) as capture:
        info = video_info(capture)
        previous_time = -1
        with (output / "single_fly_observations.csv").open("w", newline="", encoding="utf-8-sig") as observations, \
             (output / "raw_detection_candidates.csv").open("w", newline="", encoding="utf-8-sig") as raw_file, \
             (output / "per_frame_status.csv").open("w", newline="", encoding="utf-8-sig") as status_file, \
             (output / "encoding.log").open("wb") as log:
            selected_writer = csv.DictWriter(observations, fieldnames=selected_fields)
            raw_writer = csv.DictWriter(raw_file, fieldnames=detection_fields)
            count_writer = csv.DictWriter(status_file, fieldnames=count_fields)
            selected_writer.writeheader(); raw_writer.writeheader(); count_writer.writeheader()
            encoder = open_encoder(output / "annotated_single_fly.mp4", info["width"], info["height"], info["fps"], log)
            try:
                while limit is None or processed < limit:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    index = processed
                    pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if not np.isfinite(pts) or pts <= previous_time:
                        decoded_pts_invalid += 1
                        pts = index / info["fps"]
                    previous_time = pts
                    dx, dy, match_fraction, valid = registration.update(frame)
                    if not valid:
                        registration_failures += 1
                    states = {}
                    counts = {"frame_idx": index, "time_sec": pts, "ambiguous_tubes": 0}
                    for tube in tubes:
                        item = geometry["tubes"][tube]
                        x1, y1, x2, y2 = item["roi"]
                        rx1, ry1, rx2, ry2 = [round(value) for value in (x1 + dx, y1 + dy, x2 + dx, y2 + dy)]
                        rx1, rx2 = max(0, rx1), min(frame.shape[1], rx2)
                        ry1, ry2 = max(0, ry1), min(frame.shape[0], ry2)
                        if rx2 - rx1 < 40 or ry2 - ry1 < 120:
                            raise ValueError(f"{tube}自动ROI越界或过小。")
                        raw = []
                        if trackers[tube].crossing is None:
                            result = model.predict(frame[ry1:ry2, rx1:rx2], conf=0.1, iou=0.45,
                                                   imgsz=640, device=selected_device, rect=True, max_det=20, verbose=False, save=False)[0]
                            for det_index, values in enumerate(result.boxes.data.detach().cpu().tolist(), 1):
                                ax, ay, bx, by, confidence, cls = values
                                if int(cls) != 0:
                                    raise ValueError("检测器输出了非fly类别。")
                                row = {"x1": ax, "y1": ay, "x2": bx, "y2": by, "confidence": confidence}
                                raw.append(row)
                                raw_writer.writerow(fmt(dict(frame_idx=index, time_sec=pts, tube_id=tube,
                                                             detection_index=det_index, **row)))
                        # 坐标偏移以实际裁剪位置为准；基准平移单独存为参考坐标。
                        state = trackers[tube].update(index, pts, raw, (rx1, ry1), (dx, dy), valid)
                        states[tube] = state
                        counts[tube] = int(state["selected"] is not None)
                        counts["ambiguous_tubes"] += state["status"] == "ambiguous"
                        point = state["selected"]
                        row = {"frame_idx": index, "time_sec": pts, "tube_id": tube,
                               "track_id": trackers[tube].track_id, "status": state["status"],
                               "candidate_count": state["candidate_count"], "confidence": None,
                               "x1": None, "y1": None, "x2": None, "y2": None,
                               "roi_center_x": None, "roi_center_y": None,
                               "global_center_x": None, "global_center_y": None,
                               "reference_center_x": None, "reference_center_y": None,
                               "signed_line_distance": None, "camera_dx": dx, "camera_dy": dy,
                               "registration_valid": int(valid)}
                        if point is not None:
                            row.update(confidence=point["confidence"], x1=point["box"][0], y1=point["box"][1],
                                       x2=point["box"][2], y2=point["box"][3],
                                       roi_center_x=point["cx"] - x1, roi_center_y=point["cy"] - y1,
                                       global_center_x=point["global_x"], global_center_y=point["global_y"],
                                       reference_center_x=point["cx"], reference_center_y=point["cy"],
                                       signed_line_distance=point["signed_line_distance"])
                        selected_writer.writerow(fmt(row))
                    counts["total"] = sum(counts[tube] for tube in tubes)
                    counts["finished_tubes"] = sum(tracker.crossing is not None for tracker in trackers.values())
                    count_writer.writerow(fmt(counts))
                    encoder.stdin.write(annotate(frame, geometry, (dx, dy), states, trackers, index, pts).tobytes())
                    processed += 1
                    if processed % 100 == 0:
                        print(f"{video.name}: {processed}帧，已确认首次过线 {counts['finished_tubes']}/6", flush=True)
            finally:
                encoder.stdin.close()
                if encoder.wait(timeout=60) != 0:
                    raise RuntimeError("H.264编码失败，请查看encoding.log。")
    results = [tracker.result(geometry["finish_line"]["timing_quality"]) for tracker in trackers.values()]
    with (output / "first_passage_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        for item in results:
            writer.writerow({**item, "quality_flags": json.dumps(item["quality_flags"], ensure_ascii=False)})
    elapsed = time.perf_counter() - start
    report = {"video": str(video.resolve()), "weights": str(weights.resolve()),
              "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "source_metadata": info,
              "processed_frames": processed, "reached_end_of_video": limit is None,
              "decode_metadata_gap": max(0, info["frame_count"] - processed) if limit is None else None,
              "geometry_reference_frame": geometry["reference_frame"],
              "finish_line_timing_quality": geometry["finish_line"]["timing_quality"],
              "registration_uncertain_frames": registration_failures, "invalid_pts_fallback_frames": decoded_pts_invalid,
              "results": results, "processing_seconds": elapsed, "processing_fps": processed / elapsed,
              "device": selected_device, "cuda_used": selected_device == "0",
              "rules": {"one_identity_per_tube": True, "time_zero": "video frame 0", "distance_cm": 5.5,
                        "first_crossing": "bbox center from below to above background line; three consecutive observations confirm",
                        "missing_frames": "no synthetic observation or interpolated crossing",
                        "speed_interpretation": "nominal 5.5cm / first crossing time from video start; review flags must be checked"}}
    (output / "single_fly_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"video": video.name, "processed_frames": processed, "results": results}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "assets/fly_yolo26n_best.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="仅用于诊断，默认读取所有可解码帧")
    parser.add_argument("--device", choices=["auto", "0", "cpu"], default="auto")
    args = parser.parse_args()
    destination = args.output or ROOT / "outputs/single_fly_analysis" / args.video.stem
    run(args.video, args.weights, destination, args.limit, args.device)
