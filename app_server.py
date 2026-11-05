"""本地交互界面：拖入视频排队分析，浏览单段和分组结果。只监听本机，不需要联网。"""

import argparse
import base64
import csv
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse
import uuid
import webbrowser

import cv2

from app_paths import DATA, FROZEN, RESOURCE
from calibrate_manual import MANUAL_DIR, click_problem
from launch_single_fly import next_output
from single_fly_geometry import draw_geometry, geometry_from_clicks
from video_utils import open_video, read_frame, video_info

OUTPUTS = DATA / "outputs"
PAGE = RESOURCE / "app" / "index.html"
RUNS = OUTPUTS / "app_runs"
UPLOADS = OUTPUTS / "app_uploads"
RESULT_ROOTS = (OUTPUTS / "batch_runs", RUNS)
SERVED_SUFFIXES = {".mp4", ".png", ".csv", ".json"}
CALIBRATION_FRAME = 30
CHUNK = 1 << 20
PROGRESS = re.compile(r"(\d+)帧，已确认首次过线 (\d)/6")
REPEAT_SUFFIX = re.compile(r"_(retry|run)\d+$")

jobs = {}
jobs_lock = threading.Lock()
work = queue.Queue()
current_process = None


def update_job(job_id, **changes):
    with jobs_lock:
        jobs[job_id].update(changes)


def job_list():
    with jobs_lock:
        return [{key: value for key, value in job.items() if key != "video"} for job in jobs.values()]


def add_job(video, group, limit=None, geometry=None):
    job_id = uuid.uuid4().hex[:10]
    with open_video(video) as capture:
        frames = video_info(capture)["frame_count"]
    job = {"id": job_id, "name": video.name, "group": group, "state": "queued", "video": str(video), "frames": frames,
           "done_frames": 0, "crossed": 0, "message": "", "run": None, "limit": limit, "geometry": geometry,
           "created": time.time()}
    with jobs_lock:
        jobs[job_id] = job
    work.put(job_id)
    return job_id


def analysis_command():
    if FROZEN:
        return [str(Path(sys.executable).with_name("fly_worker.exe"))]
    return [sys.executable, "-X", "utf8", "-u", str(RESOURCE / "analyze_single_fly_video.py")]


def busy():
    with jobs_lock:
        return any(job["state"] in ("uploading", "queued", "running") for job in jobs.values())


def stop_analysis():
    if current_process and current_process.poll() is None:
        current_process.kill()


def short_error(lines):
    """界面只显示错误本身，不显示调用栈。"""
    found = [line for line in lines if "Error" in line and "PYI-" not in line]
    return found[-1] if found else " | ".join(lines)[-600:]


def worker():
    """同一时间只分析一段：显存和解码都按单任务设计。"""
    global current_process
    while True:
        job_id = work.get()
        with jobs_lock:
            job = dict(jobs[job_id])
        video = Path(job["video"])
        output = next_output(video, RUNS / job["group"])
        command = analysis_command() + [str(video), "--device", "auto", "--output", str(output)]
        if job["geometry"]:
            command += ["--geometry", job["geometry"]]
        if job["limit"]:
            command += ["--limit", str(job["limit"])]
        update_job(job_id, state="running")
        tail = deque(maxlen=12)
        current_process = process = subprocess.Popen(command, cwd=DATA, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace",
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        for line in process.stdout:
            tail.append(line.rstrip())
            found = PROGRESS.search(line)
            if found:
                update_job(job_id, done_frames=int(found[1]), crossed=int(found[2]))
        process.wait()
        text = "\n".join(tail)
        if process.returncode == 0:
            update_job(job_id, state="done", done_frames=job["frames"], run=output.relative_to(OUTPUTS).as_posix())
            if video.is_relative_to(UPLOADS):
                shutil.rmtree(video.parent, ignore_errors=True)
            continue
        shutil.rmtree(output, ignore_errors=True)
        if "标定" in text or "ROI" in text:
            message = "上次标定不正确，需要重新点选。" if job["geometry"] else "自动标定没有通过，需要手动点选。"
            update_job(job_id, state="needs_calibration", geometry=None, message=message)
        else:
            update_job(job_id, state="failed", message=short_error(tail))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def find_reports():
    """同一视频重复运行时只取最新结果；旧目录保留但不进入统计。"""
    latest = {}
    for root in RESULT_ROOTS:
        for report in root.rglob("single_fly_report.json") if root.is_dir() else ():
            key = (report.parent.parent.name, REPEAT_SUFFIX.sub("", report.parent.name))
            if key not in latest or report.stat().st_mtime > latest[key].stat().st_mtime:
                latest[key] = report
    return sorted(latest.items())


def overview():
    rows, videos = [], []
    for (group, stem), path in find_reports():
        report = json.loads(path.read_text(encoding="utf-8"))
        run = path.parent.relative_to(OUTPUTS).as_posix()
        results = report["results"]
        videos.append({"group": group, "stem": stem, "run": run, "tubes": len(results),
                       "crossed": sum(item["outcome"] == "first_crossing_recorded" for item in results),
                       "review": sum(item["speed_quality"] == "review" for item in results)})
        rows += [{"group": group, "stem": stem, "run": run, "tube": item["tube_id"], "outcome": item["outcome"],
                  "quality": item["speed_quality"], "crossing": item["first_crossing_time_sec"],
                  "speed": item["candidate_speed_cm_s"], "net_speed": item["net_climb_speed_cm_s"]} for item in results]
    return {"videos": videos, "tubes": rows}


def result_dir(run):
    path = (OUTPUTS / run).resolve()
    if not path.is_relative_to(OUTPUTS) or not (path / "single_fly_report.json").is_file():
        raise ValueError("结果不存在。")
    return path


def run_detail(run):
    folder = result_dir(run)
    report = json.loads((folder / "single_fly_report.json").read_text(encoding="utf-8"))
    pixels = report["pixels_per_cm"]
    distance = report["results"][0]["distance_cm"]
    tracks = {}
    for row in read_rows(folder / "single_fly_observations.csv"):
        if row["signed_line_distance"] and int(row["frame_idx"]) % 2 == 0:
            # 高度从管底算起，终点线就是标定距离。
            tracks.setdefault(row["tube_id"], []).append(
                [round(float(row["time_sec"]), 3), round(distance - float(row["signed_line_distance"]) / pixels, 3)])
    return {"run": run, "report": report, "tracks": tracks, "distance_cm": distance,
            "duration": report["processed_frames"] / report["source_metadata"]["fps"]}


def calibration_frame(job_id):
    with jobs_lock:
        video = jobs[job_id]["video"]
    with open_video(video) as capture:
        info = video_info(capture)
        return read_frame(capture, CALIBRATION_FRAME, info["frame_count"]), info


def encoded_jpeg(frame):
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("图像编码失败。")
    return encoded.tobytes()


def apply_calibration(job_id, line, bottoms, confirm):
    """confirm 为假时只返回预览；确认后保存几何并重新排队。"""
    frame, info = calibration_frame(job_id)
    geometry = geometry_from_clicks(line, bottoms, (info["width"], info["height"]), CALIBRATION_FRAME)
    if not confirm:
        return {"preview": base64.b64encode(encoded_jpeg(draw_geometry(frame, geometry))).decode("ascii")}
    with jobs_lock:
        job = jobs[job_id]
    video = Path(job["video"])
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    path = MANUAL_DIR / f"{video.stem}.json"
    path.write_text(json.dumps(geometry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_job(job_id, state="queued", message="", geometry=str(path), done_frames=0)
    work.put(job_id)
    return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def send_json(self, value, status=HTTPStatus.OK):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body, content_type):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, relative):
        """支持 Range，视频才能在浏览器里拖动进度。"""
        path = (OUTPUTS / unquote(relative)).resolve()
        if not path.is_relative_to(OUTPUTS) or path.suffix.lower() not in SERVED_SUFFIXES or not path.is_file():
            return self.send_json({"error": "文件不存在。"}, HTTPStatus.NOT_FOUND)
        size = path.stat().st_size
        start, end = 0, size - 1
        requested = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if requested and (requested[1] or requested[2]):
            if requested[1]:
                start = int(requested[1])
                end = min(int(requested[2]), end) if requested[2] else end
            else:
                start = max(0, size - int(requested[2]))
        if start > end:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        self.send_response(HTTPStatus.PARTIAL_CONTENT if requested else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        return None

    def do_GET(self):
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self.send_bytes(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/jobs":
                self.send_json(job_list())
            elif url.path == "/api/overview":
                self.send_json(overview())
            elif url.path == "/api/run":
                self.send_json(run_detail(query["run"]))
            elif url.path == "/api/calibration-frame":
                frame, _ = calibration_frame(query["job"])
                self.send_bytes(encoded_jpeg(frame), "image/jpeg")
            elif url.path.startswith("/files/"):
                self.send_file(url.path[len("/files/"):])
            else:
                self.send_json({"error": "没有这个页面。"}, HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass
        except (KeyError, ValueError, OSError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self):
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        length = int(self.headers.get("Content-Length", 0))
        try:
            if url.path == "/api/upload":
                self.send_json({"job": self.receive_video(query, length)})
                return
            data = json.loads(self.rfile.read(length) or b"{}")
            if url.path == "/api/reveal":
                os.startfile(result_dir(data["run"]))
                self.send_json({"ok": True})
            elif url.path == "/api/check-click":
                self.send_json({"problem": click_problem([tuple(p) for p in data["points"]], tuple(data["point"]), tuple(data["size"]))})
            elif url.path == "/api/calibrate":
                self.send_json(apply_calibration(data["job"], data["line"], data["bottoms"], bool(data.get("confirm"))))
            else:
                self.send_json({"error": "没有这个接口。"}, HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass
        except (KeyError, ValueError, OSError, cv2.error) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def receive_video(self, query, length):
        name = Path(query["name"].replace("\\", "/")).name
        if Path(name).suffix.lower() != ".mp4":
            raise ValueError(f"目前只支持 MP4：{name}")
        group = Path(query.get("group", "").replace("\\", "/").strip()).name or "未分组"
        folder = UPLOADS / uuid.uuid4().hex[:10]
        folder.mkdir(parents=True)
        path = folder / name
        with path.open("wb") as stream:
            remaining = length
            while remaining:
                chunk = self.rfile.read(min(CHUNK, remaining))
                if not chunk:
                    raise ValueError("上传中断。")
                stream.write(chunk)
                remaining -= len(chunk)
        return add_job(path, group, int(query["limit"]) if query.get("limit") else None)


def start_server(port):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server, address = start_server(args.port)
    print(f"界面地址：{address}  关闭这个窗口即退出。", flush=True)
    if not args.no_browser:
        webbrowser.open(address)
    server.serve_forever()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
