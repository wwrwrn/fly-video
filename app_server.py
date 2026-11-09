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

from app_paths import DATA, FROZEN, RESOURCE, VERSION
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
BAD_GROUP = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
UNGROUPED = "未分组"
OUTPUT_VIDEO = "annotated_single_fly.mp4"

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


def clean_group(name):
    name = (name or "").strip().rstrip(".")
    if not name:
        return UNGROUPED
    if BAD_GROUP.search(name) or len(name) > 60:
        raise ValueError('分组名不能超过 60 个字，也不能包含 < > : " / \\ | ? * 这些符号。')
    return name


def videos_in(paths, group):
    """拖入的单个视频用分组栏的名字；拖入的文件夹按视频所在的文件夹命名分组。"""
    found = []
    for raw in paths:
        path = Path(raw or "")
        if not raw or not path.is_absolute():
            # 空路径会被当成当前目录，整盘视频都会被排进来。
            continue
        if path.is_dir():
            # 跳过本程序自己输出的标注视频，免得把结果文件夹里的视频再分析一遍。
            found += [(video, clean_group(video.parent.name)) for video in sorted(path.rglob("*"))
                      if video.suffix.lower() == ".mp4" and video.is_file() and video.name != OUTPUT_VIDEO]
        elif path.is_file() and path.suffix.lower() == ".mp4":
            found.append((path, clean_group(group)))
    return found


def add_paths(paths, group):
    """桌面版直接读原视频，不复制；原文件不会被移动或修改。"""
    found = videos_in(paths, group)
    if not found:
        raise ValueError("没有找到 MP4 视频。")
    for video, name in found:
        add_job(video, name)
    return {"added": len(found)}


def dismiss_job(job_id):
    """排队中的直接取消，分析中的先停掉，结束的从列表移走。"""
    with jobs_lock:
        job = jobs[job_id]
        job["cancelled"] = True
        running = job["state"] == "running"
        if not running:
            del jobs[job_id]
    if running:
        stop_analysis()
    elif Path(job["video"]).is_relative_to(UPLOADS):
        shutil.rmtree(Path(job["video"]).parent, ignore_errors=True)


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
            if job_id not in jobs:
                continue
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
        with jobs_lock:
            cancelled = jobs[job_id].get("cancelled")
            if cancelled:
                del jobs[job_id]
        if cancelled:
            shutil.rmtree(output, ignore_errors=True)
            if video.is_relative_to(UPLOADS):
                shutil.rmtree(video.parent, ignore_errors=True)
            continue
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
        crossed = [item for item in results if item["outcome"] == "first_crossing_recorded"]
        videos.append({"group": group, "stem": stem, "run": run, "tubes": len(results), "crossed": len(crossed),
                       "review": sum(item["speed_quality"] == "review" for item in results),
                       "crossed_review": sum(item["speed_quality"] == "review" for item in crossed),
                       "mean_speed": mean(item["candidate_speed_cm_s"] for item in crossed),
                       "mean_net_speed": mean(item["net_climb_speed_cm_s"] for item in crossed),
                       "editable": path.is_relative_to(RUNS)})
        rows += [{"group": group, "stem": stem, "run": run, "tube": item["tube_id"], "outcome": item["outcome"],
                  "quality": item["speed_quality"], "crossing": item["first_crossing_time_sec"],
                  "speed": item["candidate_speed_cm_s"], "net_speed": item["net_climb_speed_cm_s"]} for item in results]
    return {"videos": videos, "tubes": rows}


def mean(values):
    """每段视频的平均速度只算过线的管；没过线的管没有速度，不计入。"""
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


SUMMARY_COLUMNS = ["分组", "视频", "试管数", "过线管数", "过线管中需复核", "平均5.5cm速度(cm/s)", "平均净爬升速度(cm/s)"]


def summary_csv():
    lines = [SUMMARY_COLUMNS]
    for video in overview()["videos"]:
        lines.append([video["group"], video["stem"], video["tubes"], video["crossed"], video["crossed_review"],
                      *("" if value is None else f"{value:.3f}" for value in (video["mean_speed"], video["mean_net_speed"]))])
    return "".join(",".join(f'"{cell}"' if "," in str(cell) else str(cell) for cell in line) + "\r\n" for line in lines)


def result_dir(run):
    path = (OUTPUTS / run).resolve()
    if not path.is_relative_to(OUTPUTS.resolve()) or not (path / "single_fly_report.json").is_file():
        raise ValueError("结果不存在。")
    return path


def send_to_recycle_bin(path):
    """Windows 上放进回收站，误删还能还原。"""
    if os.name != "nt":
        shutil.rmtree(path)
        return
    import ctypes
    from ctypes import wintypes

    class Operation(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    # FO_DELETE；静默、不弹确认、允许撤销（进回收站）、不弹错误框。
    operation = Operation(None, 3, str(Path(path).resolve()) + "\0", None, 0x4 | 0x10 | 0x40 | 0x400, False, None, None)
    code = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if code or operation.fAnyOperationsAborted:
        raise OSError(f"放进回收站失败（代码 {code}），文件可能正在被使用。")


def video_runs(run):
    """同一视频的全部运行目录（含重跑）。只允许管理界面自己分析出的结果。"""
    folder = result_dir(run)
    if not folder.is_relative_to(RUNS.resolve()) or folder.parent.parent != RUNS.resolve():
        raise ValueError("只能修改界面里分析的结果。")
    stem = REPEAT_SUFFIX.sub("", folder.name)
    return stem, [item for item in folder.parent.iterdir() if item.is_dir() and REPEAT_SUFFIX.sub("", item.name) == stem]


def remove_if_empty(folder):
    try:
        folder.rmdir()
    except OSError:
        pass


def delete_video(run):
    _, folders = video_runs(run)
    for folder in folders:
        send_to_recycle_bin(folder)
    remove_if_empty(folders[0].parent)


def group_folder(group):
    folder = RUNS / clean_group(group)
    if not folder.is_dir():
        raise ValueError("只能修改界面里分析的分组。")
    return folder


def group_videos(group):
    """分组里每段视频取一个运行目录作代表。"""
    seen = {}
    for folder in sorted(group_folder(group).iterdir()):
        if (folder / "single_fly_report.json").is_file():
            seen.setdefault(REPEAT_SUFFIX.sub("", folder.name), folder.relative_to(OUTPUTS).as_posix())
    return list(seen.values())


def move_videos(runs, group):
    """先全部检查再移动，避免只移走一半。"""
    target = RUNS / clean_group(group)
    plans = [video_runs(run) for run in runs]
    existing = {REPEAT_SUFFIX.sub("", item.name) for item in target.iterdir()} if target.is_dir() else set()
    for stem, folders in plans:
        if folders[0].parent != target.resolve() and stem in existing:
            raise ValueError(f"“{target.name}”里已经有 {stem} 的结果，请先删掉其中一份。")
    target.mkdir(parents=True, exist_ok=True)
    moved = []
    for stem, folders in plans:
        source = folders[0].parent
        if source != target.resolve():
            for folder in folders:
                folder.rename(target / folder.name)
            remove_if_empty(source)
        moved.append(f"app_runs/{target.name}/{Path(runs[len(moved)]).name}")
    return moved


def delete_group(group):
    for run in group_videos(group):
        delete_video(run)


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
            elif url.path == "/api/info":
                self.send_json({"version": VERSION})
            elif url.path == "/api/video-summary.csv":
                body = summary_csv().encode("utf-8-sig")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=video_summary.csv; filename*=UTF-8''"
                                 "%E8%A7%86%E9%A2%91%E5%B9%B3%E5%9D%87%E9%80%9F%E5%BA%A6.csv")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
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
            elif url.path == "/api/add":
                self.send_json(add_paths(data["paths"], data.get("group", "")))
            elif url.path == "/api/dismiss":
                dismiss_job(data["job"])
                self.send_json({"ok": True})
            elif url.path == "/api/delete":
                delete_video(data["run"])
                self.send_json({"ok": True})
            elif url.path == "/api/move":
                self.send_json({"runs": move_videos([data["run"]], data["group"])})
            elif url.path == "/api/rename-group":
                self.send_json({"runs": move_videos(group_videos(data["group"]), data["name"])})
            elif url.path == "/api/delete-group":
                delete_group(data["group"])
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
        group = clean_group(Path(query.get("group", "").replace("\\", "/").strip()).name)
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
