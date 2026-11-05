"""区分只读资源和可写数据：源码运行时都在项目目录，打包后数据写到用户的文档文件夹。"""

import os
from pathlib import Path
import sys

FROZEN = getattr(sys, "frozen", False)
RESOURCE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
DATA = Path(os.environ.get("FLY_APP_DATA") or Path.home() / "Documents" / "果蝇攀爬分析") if FROZEN else RESOURCE
