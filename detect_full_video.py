"""逐帧检测完整视频：六个 ROI 在内存中分别推理，流式保存 CSV 和 H.264 视频。"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from roi_utils import draw_rois, load_rois
from video_utils import open_video, video_info

ROOT = Path(__file__).resolve().parent
(ROOT / ".ultralytics").mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
TUBES = [f"tube_{i}" for i in range(1, 7)]
DETECTION_COLUMNS = ["frame_idx", "time_sec", "tube_id", "detection_index", "confidence",
                     "x1", "y1", "x2", "y2", "center_x", "center_y", "width", "height",
                     "global_center_x", "global_center_y"]


def detect_rois(model, frame, rois, device, conf, imgsz):
    """detection_index 在每帧、每根试管内从 1 重新编号，仅表示列表顺序，绝非身份。"""
    detections = {}
    for tube in TUBES:
        ox, oy, ex, ey = rois[tube]
        crop = frame[oy:ey, ox:ex]
        result = model.predict(source=crop, imgsz=imgsz, conf=conf, iou=0.7,
                               device=device, rect=True, max_det=100, verbose=False, save=False)[0]
        rows = []
        for index, values in enumerate(result.boxes.data.detach().cpu().tolist(), 1):
            x1, y1, x2, y2, confidence, class_id = values
            if int(class_id) != 0 or not all(np.isfinite(values)):
                raise ValueError("模型产生了非 fly 类别或无效坐标。")
            if not (0 <= x1 < x2 <= crop.shape[1] and 0 <= y1 < y2 <= crop.shape[0]):
                raise ValueError(f"模型框越界：{tube} {values}")
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rows.append(dict(tube_id=tube, detection_index=index, confidence=confidence,
                             x1=x1, y1=y1, x2=x2, y2=y2, center_x=cx, center_y=cy,
                             width=x2 - x1, height=y2 - y1,
                             global_center_x=cx + ox, global_center_y=cy + oy))
        detections[tube] = rows
    return detections


def annotate_frame(frame, rois, detections, frame_idx, time_sec):
    """原图上画 ROI、fly 和置信度，完全不画身份编号或轨迹。"""
    result = draw_rois(frame, rois)
    cv2.rectangle(result, (0, 0), (result.shape[1], 60), (25, 25, 25), -1)
    total = sum(len(rows) for rows in detections.values())
    cv2.putText(result, f"Frame {frame_idx} | t={time_sec:.3f}s | detections={total} | conf>=0.25",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    for tube, rows in detections.items():
        ox, oy, _, _ = rois[tube]
        for row in rows:
            x1, y1, x2, y2 = (round(row[key] + offset) for key, offset in
                              [("x1", ox), ("y1", oy), ("x2", ox), ("y2", oy)])
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 130, 0), 1)
            label = f"fly {row['confidence']:.2f}"
            origin = (max(0, min(x1, result.shape[1] - 75)), max(14, y1 - 3))
            cv2.putText(result, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3)
            cv2.putText(result, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
    return result


def open_encoder(path, width, height, fps, log):
    """向 FFmpeg 输送内存中的 BGR 帧，不落盘中间图片，编码为浏览器可播放的 H.264。"""
    import imageio_ffmpeg

    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "warning", "-n",
               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
               "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2",
               "-movflags", "+faststart", str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def run_detection(video, config, weights, output, device="0"):
    import torch
    import ultralytics
    from ultralytics import YOLO

    whole_start = time.perf_counter()
    video, config, weights, output = map(Path, [video, config, weights, output])
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，请更换目录：{output}")
    cuda = device != "cpu"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；如确需 CPU，请明确传入 --device cpu。")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    model = YOLO(str(weights), task="detect")
    if model.names != {0: "fly"}:
        raise ValueError("模型类别必须为 {0: fly}。")
    output.mkdir(parents=True, exist_ok=True)
    conf, imgsz = 0.25, 640
    read_failures, inference_times, pipeline_times = [], [], []
    processed, detection_count, frame_idx = 0, 0, 0
    with open_video(video) as capture:
        info = video_info(capture)
        ok, first_frame = capture.read()
        if not ok:
            raise ValueError("视频首帧读取失败。")
        rois = load_rois(config, first_frame)
        # 预热不计入稳定运行速度；正式处理仍包含 frame 0。
        detect_rois(model, first_frame, rois, device, conf, imgsz)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        setup_seconds = time.perf_counter() - whole_start
        pipeline_start = time.perf_counter()
        with (output / "full_video_detections.csv").open("w", newline="", encoding="utf-8-sig") as detail_file, \
             (output / "detection_counts.csv").open("w", newline="", encoding="utf-8-sig") as count_file, \
             (output / "encoding.log").open("wb") as log:
            detail_writer = csv.DictWriter(detail_file, fieldnames=DETECTION_COLUMNS)
            count_writer = csv.DictWriter(count_file, fieldnames=["frame_idx", "time_sec", *TUBES, "total"])
            detail_writer.writeheader()
            count_writer.writeheader()
            encoder = open_encoder(output / "annotated_detection.mp4", first_frame.shape[1], first_frame.shape[0], info["fps"], log)
            try:
                while True:
                    iteration_start = time.perf_counter()
                    if frame_idx == 0:
                        ok, frame = True, first_frame
                    else:
                        ok, frame = capture.read()
                    if not ok or frame is None:
                        if frame_idx < info["frame_count"]:
                            read_failures.append(frame_idx)
                        if frame_idx + 1 < info["frame_count"]:
                            # 对元数据范围内后续帧尝试重新定位，避免中间坏帧导致整段提前结束。
                            frame_idx += 1
                            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
                                raise ValueError(f"解码失败后无法定位 frame {frame_idx}。")
                            continue
                        break
                    if frame.shape != first_frame.shape:
                        raise ValueError("视频中途分辨率发生变化，停止处理以避免 ROI 错位。")
                    source_time = frame_idx / info["fps"]
                    if cuda:
                        torch.cuda.synchronize()
                    infer_start = time.perf_counter()
                    detections = detect_rois(model, frame, rois, device, conf, imgsz)
                    if cuda:
                        torch.cuda.synchronize()
                    inference_times.append(time.perf_counter() - infer_start)
                    counts = {tube: len(detections[tube]) for tube in TUBES}
                    total = sum(counts.values())
                    count_writer.writerow(dict(frame_idx=frame_idx, time_sec=f"{source_time:.9f}", **counts, total=total))
                    for rows in detections.values():
                        for row in rows:
                            formatted = {key: f"{value:.9f}" if isinstance(value, float) else value for key, value in row.items()}
                            detail_writer.writerow(dict(frame_idx=frame_idx, time_sec=f"{source_time:.9f}", **formatted))
                    annotated = annotate_frame(frame, rois, detections, frame_idx, source_time)
                    encoder.stdin.write(annotated.tobytes())
                    processed += 1
                    detection_count += total
                    pipeline_times.append(time.perf_counter() - iteration_start)
                    if processed % 50 == 0:
                        elapsed = time.perf_counter() - pipeline_start
                        print(f"已处理 {processed} 帧，source frame={frame_idx}，当前完整流程 {processed / elapsed:.2f} FPS", flush=True)
                    frame_idx += 1
            finally:
                if encoder.stdin:
                    try:
                        encoder.stdin.close()
                    except BrokenPipeError:
                        pass
                try:
                    return_code = encoder.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
                    raise RuntimeError("视频编码器关闭超时。")
            if return_code != 0:
                raise RuntimeError("视频编码失败，请查看 encoding.log。")
        pipeline_seconds = time.perf_counter() - pipeline_start
    whole_seconds = time.perf_counter() - whole_start
    report = {"video": str(video.resolve()), "weights": str(weights.resolve()),
              "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "roi_snapshot": rois,
              "source_metadata": info, "processed_frames": processed, "failed_source_frame_indices": read_failures,
              "detection_rows": detection_count,
              "inference_parameters": {"conf": conf, "imgsz": imgsz, "iou": 0.7, "max_det": 100, "device": device,
                                       "rect": True, "mode": "six separate ROI predictions per source frame"},
              "timing": {"setup_and_warmup_seconds": setup_seconds, "pipeline_seconds": pipeline_seconds,
                         "whole_detection_run_seconds": whole_seconds,
                         "average_frame_ms": pipeline_seconds / processed * 1000, "pipeline_fps": processed / pipeline_seconds,
                         "whole_run_fps_including_setup": processed / whole_seconds,
                         "roi_inference_total_seconds": sum(inference_times),
                         "six_roi_inference_mean_frame_ms": float(np.mean(inference_times)) * 1000,
                         "six_roi_inference_fps": processed / sum(inference_times),
                         "per_frame_pipeline_p95_ms": float(np.percentile(pipeline_times, 95)) * 1000},
              "cuda_used": cuda, "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
              "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3 if cuda else None,
              "peak_torch_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3 if cuda else None,
              "torch": torch.__version__, "ultralytics": ultralytics.__version__,
              "time_definition": "time_sec=source frame_idx / metadata FPS，估算时间，不是逐帧精确 PTS。",
              "output_video": "H.264/yuv420p，无音频；按输入名义 FPS 写入实际成功读取帧，失败帧不补造。",
              "identity_note": "detection_index 仅为当帧、当试管的检测列表序号，不是个体 ID。",
              "performance_scope": "完整流程包含读取、6次ROI推理、画框、CSV和视频编码及收尾；不含后续统计、截图与图表。"}
    (output / "performance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/detect/fly_yolo26n_teaching/weights/best.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "config/tube_rois.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/full_video_detection")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    run_detection(args.video, args.config, args.weights, args.output, args.device)
