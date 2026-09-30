"""通过只读 QA 后按时间点复制训练数据：前四帧用于 train，320 帧用于 val。"""

import argparse
import json
from pathlib import Path
import shutil
import sys

from qa_training_labels import digest


def prepare(report_path, output):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    source, output = Path(report["dataset"]), Path(output)
    if not report["passed"] or report["issues"] or report["image_count"] != 30:
        raise ValueError("QA 未通过，不建立训练集。")
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("输出不得覆盖或放入原始 dataset_sample。")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空：{output}")
    for relative, expected_hash in report["source_sha256"].items():
        if digest(source / relative) != expected_hash:
            raise ValueError(f"质检后源文件发生变化，需重新 QA：{relative}")
    entries = []
    for row in report["images"]:
        split = "val" if row["frame"] == 320 else "train"
        stem = Path(row["image"]).stem
        for folder, suffix in [("images", ".png"), ("labels", ".txt")]:
            target = output / folder / split / f"{stem}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            original = source / folder / f"{stem}{suffix}"
            shutil.copy2(original, target)
            if digest(target) != digest(original):
                raise ValueError(f"复制校验失败：{target}")
        entries.append({**row, "split": split})
    yaml = f"# 同一视频的教学实验，验证指标不代表跨视频泛化能力。\npath: {json.dumps(output.resolve().as_posix(), ensure_ascii=False)}\ntrain: images/train\nval: images/val\nnames:\n  0: fly\n"
    (output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    split_report = {"source": str(source.resolve()), "qa_report": str(Path(report_path).resolve()),
                    "split_counts": report["split_counts"], "images": entries,
                    "original_sha256": report["source_sha256"],
                    "limitation": "同一短视频中的时间点划分，不能视为独立跨视频泛化验证。"}
    (output / "split_report.json").write_text(json.dumps(split_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(split_report["split_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-report", type=Path, default=Path("outputs/training_qa/qa_report.json"))
    parser.add_argument("--output", type=Path, default=Path("dataset_yolo"))
    args = parser.parse_args()
    try:
        prepare(args.qa_report, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"错误：{exc}\n")
