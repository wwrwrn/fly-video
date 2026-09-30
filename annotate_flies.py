"""简单的 OpenCV 人工标注窗口：逐图画果蝇框、撤销、保存 YOLO 标签并继续。"""

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from yolo_labels import draw_boxes, read_image, read_labels, write_labels, xyxy_to_yolo


def annotate_image(image, initial_boxes, title):
    """释放鼠标即增加一个框；Enter 保存下一张，U 撤销，D 存草稿退出，Esc 放弃本次修改。"""
    height, width = image.shape[:2]
    scale = min(3.0, 740 / height, 1100 / width)
    dw, dh = max(1, round(width * scale)), max(1, round(height * scale))
    canvas_width, top = max(760, dw), 85
    left = (canvas_width - dw) // 2
    boxes = [list(box) for box in initial_boxes]
    state = {"start": None, "end": None, "empty_confirmed": False}
    window = "Fly boxes - manual annotation"

    def point(x, y):
        return min(dw, max(0, x - left)), min(dh, max(0, y - top))

    def current_box():
        if state["start"] is None:
            return None
        (ax, ay), (bx, by) = state["start"], state["end"]
        if ax == bx or ay == by:
            return None
        return [math.floor(min(ax, bx) * width / dw), math.floor(min(ay, by) * height / dh),
                min(width, math.ceil(max(ax, bx) * width / dw)), min(height, math.ceil(max(ay, by) * height / dh))]

    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and left <= x < left + dw and top <= y < top + dh:
            state.update(start=point(x, y), end=point(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and state["start"] is not None:
            state["end"] = point(x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["start"] is not None:
            state["end"] = point(x, y)
            box = current_box()
            if box and box not in boxes:
                boxes.append(box)
                state["empty_confirmed"] = False
                print(f"新增框 {len(boxes)}：{box}", flush=True)
            state.update(start=None, end=None)

    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, mouse)
        while True:
            display = draw_boxes(image, boxes)
            pending = current_box()
            if pending:
                cv2.rectangle(display, tuple(pending[:2]), (pending[2] - 1, pending[3] - 1), (0, 170, 255), 1)
            canvas = np.full((dh + top, canvas_width, 3), 30, dtype=np.uint8)
            canvas[top:, left:left + dw] = cv2.resize(display, (dw, dh), interpolation=cv2.INTER_LINEAR)
            messages = [f"{title} | {width}x{height} | boxes: {len(boxes)}",
                        "Drag: add | U/Backspace: undo | Enter/N: save ALL + next",
                        "D: draft + quit | Esc: discard changes + quit | E: confirm empty"]
            if state["empty_confirmed"]:
                messages[2] = "Confirmed no flies: Enter saves an empty label."
            for row, message in enumerate(messages):
                cv2.putText(canvas, message, (10, 22 + row * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(30)
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return "quit", boxes
            if key in (8, 127, ord("u"), ord("U")):
                if state["start"] is None and boxes:
                    boxes.pop()
                state.update(start=None, end=None, empty_confirmed=False)
            elif key in (ord("d"), ord("D")):
                return "draft", boxes
            elif key in (ord("e"), ord("E")) and not boxes:
                state["empty_confirmed"] = True
            elif key in (10, 13, ord("n"), ord("N")) and state["start"] is None:
                if boxes or state["empty_confirmed"]:
                    return "complete", boxes
                print("没有框。若已确认整张图无果蝇，请先按 E 再按 Enter；未标完可按 D 存草稿。", flush=True)
    finally:
        cv2.destroyAllWindows()


def annotate_dataset(dataset, limit=None, review_existing=False):
    """同名 txt 只在用户确认完整标注后保存，草稿另存 JSON，避免混入训练标签。"""
    dataset = Path(dataset)
    images = sorted((dataset / "images").glob("*.png"))
    if not images:
        raise ValueError("images 目录中没有 PNG 候选图片，请先生成数据。")
    if len({path.stem for path in images}) != len(images):
        raise ValueError("图片名称重复。")
    completed = 0
    for path in images:
        label = dataset / "labels" / f"{path.stem}.txt"
        draft = dataset / "drafts" / f"{path.stem}.json"
        if label.exists() and not review_existing and not draft.exists():
            continue
        image = read_image(path)
        height, width = image.shape[:2]
        initial = []
        if draft.exists():
            data = json.loads(draft.read_text(encoding="utf-8"))
            if data.get("image_size") != [width, height]:
                raise ValueError(f"草稿尺寸不匹配：{draft}")
            initial = data["boxes_xyxy"]
            for box in initial:
                xyxy_to_yolo(box, width, height)
        elif label.exists():
            initial = [[round(v) for v in box] for box in read_labels(label, width, height)]
        print(f"正在标注：{path.name}；请框出全部可辨认果蝇，每只一个框。", flush=True)
        action, boxes = annotate_image(image, initial, path.name)
        if action == "quit":
            print("已退出，本张未保存的修改已放弃，之前保存的标签保留。")
            break
        record = {"image": path.name, "image_size": [width, height], "boxes_xyxy": boxes,
                  "status": action, "source": "human_mouse_annotation"}
        if action == "draft":
            draft.parent.mkdir(parents=True, exist_ok=True)
            draft.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"草稿已保存：{draft.resolve()}；未创建或修改训练标签。")
            break
        lines = write_labels(label, boxes, width, height)
        record["yolo_lines"] = lines
        audit = dataset / "annotations" / f"{path.stem}.json"
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if draft.exists():
            draft.unlink()
        print(f"已保存 {len(boxes)} 个 fly 框：{label.resolve()}", flush=True)
        completed += 1
        if limit is not None and completed >= limit:
            break
    print(f"本次完成 {completed} 张图片；当前共 {len(list((dataset / 'labels').glob('*.txt')))} / {len(images)} 张具有标签。")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset_sample"))
    parser.add_argument("--limit", type=int, help="本次最多保存多少张完整标注，例如 1")
    parser.add_argument("--review-existing", action="store_true", help="也打开已保存的标签以人工复查和修改")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须大于 0。")
    try:
        annotate_dataset(args.dataset, args.limit, args.review_existing)
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
