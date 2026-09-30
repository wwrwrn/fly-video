"""ROI 公共工具：检查坐标、保存配置、绘制编号并输出六根试管的裁剪图。"""

import json
from pathlib import Path

import cv2

from video_utils import open_video, read_frame, save_frame, video_info


def representative_frame(video, index=100):
    """重新打开视频获取代表帧，复用第 0 阶段的读取逻辑。"""
    with open_video(video) as capture:
        return read_frame(capture, index, video_info(capture)["frame_count"])


def validate_rois(rois, width, height, count=6):
    """坐标采用原图像素，右下角不包含在裁剪范围中。"""
    names = [f"tube_{i + 1}" for i in range(count)]
    if not isinstance(rois, dict) or set(rois) != set(names):
        raise ValueError(f"ROI 配置必须恰好包含 {', '.join(names)}。")
    for name in names:
        box = rois[name]
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError(f"{name} 必须是四个整数组成的 [x1, y1, x2, y2]。")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"{name} 坐标无效或超出 {width} × {height} 图像范围：{box}")
    if len({tuple(box) for box in rois.values()}) != count:
        raise ValueError("存在完全重复的 ROI，请检查是否重复选择了同一根试管。")
    return {name: rois[name] for name in names}


def metadata_path(path):
    return Path(path).with_suffix(".meta.json")


def save_rois(path, rois, frame, video, frame_index, *, overwrite=False, origin="mouse_selection"):
    """主 JSON 只存坐标；旁边的元数据文件记录原始尺寸，防止误用于其他分辨率。"""
    path = Path(path)
    if path.suffix.lower() != ".json" or path.name.endswith(".meta.json"):
        raise ValueError("配置文件应为普通 .json 文件，不能使用 .meta.json 后缀。")
    height, width = frame.shape[:2]
    rois = validate_rois(rois, width, height)
    meta_path = metadata_path(path)
    if not overwrite and (path.exists() or meta_path.exists()):
        raise FileExistsError(f"配置已存在：{path}，请更换文件名或使用 --overwrite。")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"image_size": [width, height], "source_video": str(Path(video).resolve()),
                "frame_index": frame_index, "origin": origin,
                "coordinate_convention": "[x1, y1, x2, y2], exclusive x2/y2"}
    # 先写临时文件，再替换，避免保存过程中留下半截 JSON。
    for target, data in [(meta_path, metadata), (path, rois)]:
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)


def load_rois(path, frame):
    """加载旧配置；不缩放坐标，拍摄位置和分辨率一致时才能直接复用。"""
    path = Path(path)
    height, width = frame.shape[:2]
    rois = json.loads(path.read_text(encoding="utf-8-sig"))
    meta_path = metadata_path(path)
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        if not isinstance(metadata, dict) or metadata.get("image_size") != [width, height]:
            raise ValueError("视频分辨率与 ROI 配置的原始尺寸不一致，请重新框选。")
    else:
        print("提示：缺少 .meta.json，当前仅检查坐标边界；请自行确认原始分辨率和拍摄位置相同。")
    return validate_rois(rois, width, height)


def draw_rois(frame, rois):
    """在副本上画矩形和英文编号，原图和裁剪图不带标注。"""
    preview = frame.copy()
    for name, (x1, y1, x2, y2) in rois.items():
        cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 220, 0), 3)
        position = (x1 + 5, max(25, y1 - 10))
        cv2.putText(preview, name, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(preview, name, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return preview


def export_rois(video, frame_index, config, output_dir):
    """从保存的 JSON 重新读取 ROI，输出六张原尺寸裁剪图、预览和尺寸报告。"""
    frame = representative_frame(video, frame_index)
    rois = load_rois(config, frame)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，请指定新目录：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, (x1, y1, x2, y2) in rois.items():
        crop = frame[y1:y2, x1:x2]
        save_frame(crop, output_dir / f"{name}.png")
        sizes[name] = [x2 - x1, y2 - y1]
        print(f"{name}：{x2 - x1} × {y2 - y1} 像素")
    save_frame(draw_rois(frame, rois), output_dir / "roi_preview.png")
    report = {"video": str(Path(video).resolve()), "frame_index": frame_index,
              "roi_config": str(Path(config).resolve()), "image_size": [frame.shape[1], frame.shape[0]],
              "crop_sizes_width_height": sizes}
    (output_dir / "crop_info.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存 6 张裁剪图、ROI 预览和尺寸报告：{output_dir.resolve()}")
    return report
