"""把 dist/FlyClimb 压成发布用的 zip，并放入使用说明。"""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "dist" / "FlyClimb"
ARCHIVE = ROOT / "dist" / "FlyClimb-Windows-CPU.zip"
GUIDE = r"""果蝇攀爬分析（Windows CPU 版）

1. 先把压缩包完整解压，不要在压缩包里直接运行。
2. 双击“果蝇攀爬分析.exe”。如果出现 Windows 安全提示，点“更多信息”→“仍要运行”。
3. 把 MP4 或整个文件夹拖到窗口左侧。拖文件夹时，文件夹名就是分组名。
4. 结果在“文档\果蝇攀爬分析\outputs\app_runs”，界面里点“打开结果文件夹”也能找到。

只适用于六管、每管一只、白板加一条终点线的拍摄布局；管底到终点线按 5.5 cm 计算。
自动标定失败时，在界面里点任务卡片，手动点选横线和管底。
不要删除 _internal 文件夹，也不要单独移动 exe。
"""

with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    bundle.writestr("FlyClimb/使用说明.txt", GUIDE.encode("utf-8-sig"))
    for path in sorted(FOLDER.rglob("*")):
        if path.is_file():
            bundle.write(path, Path("FlyClimb") / path.relative_to(FOLDER))
print(ARCHIVE, f"{ARCHIVE.stat().st_size / 2**20:.0f} MB")
