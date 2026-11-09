"""桌面程序入口：在独立窗口里打开分析界面，窗口组件不可用时退回默认浏览器。"""

import ctypes
import json
import logging
import threading
import time
import webbrowser

import app_server
from app_paths import DATA, VERSION

TITLE = "果蝇攀爬分析"
QUIT_QUESTION = "还有视频在排队或分析，退出会中断它们。确定退出吗？"
LOG = DATA / "logs" / "app.log"
log = logging.getLogger("fly_app")


class DesktopApi:
    """供界面调用：用系统对话框选视频，直接拿到原文件路径，不复制视频。"""

    def __init__(self):
        self._window = None

    def choose_videos(self):
        import webview

        paths = self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True,
                                                file_types=("MP4 视频 (*.mp4)",))
        return list(paths or [])

    def choose_folder(self):
        import webview

        paths = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        return list(paths or [])


def keep_close_prompt_current(window):
    """只有任务未完成时才询问，空闲时直接关闭。"""
    while True:
        window.confirm_close = app_server.busy()
        time.sleep(0.5)


def bind_drop(window):
    """拖入时由窗口组件给出原文件的完整路径，再交给页面按路径排队。"""
    from webview.dom import DOMEventHandler

    def on_drop(event):
        files = event.get("dataTransfer", {}).get("files", [])
        paths = [item["pywebviewFullPath"] for item in files if item.get("pywebviewFullPath")]
        log.info("拖入 %d 项，取得路径 %d 个", len(files), len(paths))
        if paths:
            window.evaluate_js(f"desktopDrop({json.dumps(paths)})")

    window.dom.document.events.drop += DOMEventHandler(on_drop, prevent_default=True)


def open_window(address):
    import webview
    from webview.platforms import winforms

    if winforms.renderer != "edgechromium":
        # 没有 WebView2 时 pywebview 会退回旧版 IE 内核，页面无法正常显示。
        raise RuntimeError(f"窗口组件是 {winforms.renderer}，不是 WebView2")
    webview.settings["ALLOW_DOWNLOADS"] = True
    api = DesktopApi()
    window = webview.create_window(TITLE, address, js_api=api, width=1360, height=860, min_size=(960, 620),
                                   maximized=True)
    api._window = window
    window.events.loaded += lambda: bind_drop(window)
    threading.Thread(target=keep_close_prompt_current, args=(window,), daemon=True).start()
    log.info("以桌面窗口打开")
    webview.start(localization={"global.quitConfirmation": QUIT_QUESTION})


def open_browser(address):
    log.info("以浏览器打开：%s", address)
    webbrowser.open(address)
    ctypes.windll.user32.MessageBoxW(None, f"已在浏览器中打开：{address}\n点“确定”退出程序。", TITLE, 0x40)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LOG, encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log.info("启动 v%s", VERSION)
    server, address = app_server.start_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        open_window(address)
    except Exception:
        log.exception("桌面窗口打不开，改用浏览器")
        open_browser(address)
    app_server.stop_analysis()
    log.info("退出")


if __name__ == "__main__":
    main()
