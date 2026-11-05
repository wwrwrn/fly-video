"""批量分析“品系 性别”子目录下的全部视频：已完成的跳过，失败的记录下来，最后汇总成一张表和分组统计。"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

from analyze_single_fly_video import run
from app_paths import RESOURCE as ROOT

MANUAL_DIR = ROOT / "config" / "manual_geometry"
DEFAULT_OUTPUT = ROOT / "outputs" / "batch_runs"
SPEED_FIELDS = ["candidate_speed_cm_s", "net_climb_speed_cm_s"]


def parse_group(folder_name):
    """目录名形如“14541 雄”；其他写法整体记为品系，性别留空。"""
    parts = folder_name.rsplit(" ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 and parts[1] in {"雄", "雌"} else (folder_name, "")


def find_videos(root):
    return sorted(path for path in root.rglob("*") if path.suffix.lower() == ".mp4")


def summarize(rows):
    """按品系+性别汇总。只统计通过自动质量规则的结果，CHECK 另外计数，便于对照。"""
    groups = {}
    for row in rows:
        groups.setdefault((row["genotype"], row["sex"]), []).append(row)
    table = []
    for (genotype, sex), items in sorted(groups.items()):
        entry = {"genotype": genotype, "sex": sex, "tubes": len(items),
                 "crossed": sum(row["outcome"] == "first_crossing_recorded" for row in items),
                 "not_reached": sum(row["outcome"] == "not_reached" for row in items),
                 "review": sum(row["speed_quality"] == "review" for row in items)}
        for field in SPEED_FIELDS:
            values = [float(row[field]) for row in items if row[field] not in ("", None) and row["speed_quality"] != "review"]
            entry[f"{field}_n"] = len(values)
            entry[f"{field}_median"] = statistics.median(values) if values else None
            entry[f"{field}_mean"] = statistics.fmean(values) if values else None
            entry[f"{field}_sd"] = statistics.stdev(values) if len(values) > 1 else None
        table.append(entry)
    return table


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, help="默认 outputs/batch_runs/文件夹名")
    parser.add_argument("--weights", type=Path, default=ROOT / "assets/fly_yolo26n_best.pt")
    parser.add_argument("--device", choices=["auto", "0", "cpu"], default="auto")
    parser.add_argument("--distance-cm", type=float, default=5.5)
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUT / args.folder.name
    output.mkdir(parents=True, exist_ok=True)
    videos = find_videos(args.folder)
    if not videos:
        raise FileNotFoundError(f"文件夹内没有MP4：{args.folder}")
    rows, failures = [], []
    for number, video in enumerate(videos, 1):
        relative = video.relative_to(args.folder)
        genotype, sex = parse_group(relative.parts[0]) if len(relative.parts) > 1 else ("", "")
        target = output / relative.parent / video.stem
        report_path = target / "single_fly_report.json"
        print(f"[{number}/{len(videos)}] {relative}", flush=True)
        if not report_path.is_file():
            manual = MANUAL_DIR / f"{video.stem}.json"
            try:
                if target.exists():
                    # 上次中断留下的半成品没有报告文件，换目录重跑，不删除。
                    target = target.with_name(f"{video.stem}_retry{sum(1 for _ in target.parent.glob(video.stem + '_retry*')) + 1}")
                    report_path = target / "single_fly_report.json"
                run(video, args.weights, target, None, args.device, args.distance_cm, manual if manual.is_file() else None)
            except (ValueError, RuntimeError, OSError) as exc:
                failures.append({"video": str(relative), "reason": str(exc)})
                print(f"  失败：{exc}", flush=True)
                continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for result in report["results"]:
            rows.append({"genotype": genotype, "sex": sex, "video": video.stem, **result,
                         "quality_flags": ";".join(result["quality_flags"]),
                         "calibration": report.get("calibration_method", "")})
    write_csv(output / "all_tubes.csv", rows)
    write_csv(output / "group_summary.csv", summarize(rows))
    write_csv(output / "failed_videos.csv", failures)
    print(f"完成：{len(videos) - len(failures)}/{len(videos)} 段。汇总：{output / 'all_tubes.csv'}", flush=True)
    if failures:
        print(f"失败 {len(failures)} 段，见 failed_videos.csv；可用 calibrate_manual.py 手动点选后重新运行本命令。", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
