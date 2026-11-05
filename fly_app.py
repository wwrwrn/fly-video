"""桌面程序入口：在独立窗口里打开分析界面，窗口组件不可用时退回默认浏览器。"""

import ctypes
import threading
import time
import webbrowser

import app_server

TITLE = "果蝇攀爬分析"
QUIT_QUESTION = "还有视频在排队或分析，退出会中断它们。确定退出吗？"


def keep_close_prompt_current(window):
    """只有任务未完成时才询问，空闲时直接关闭。"""
    while True:
        window.confirm_close = app_server.busy()
        time.sleep(0.5)


def open_window(address):
    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True
    window = webview.create_window(TITLE, address, width=1360, height=860, min_size=(960, 620))
    threading.Thread(target=keep_close_prompt_current, args=(window,), daemon=True).start()
    webview.start(localization={"global.quitConfirmation": QUIT_QUESTION})


def open_browser(address):
    webbrowser.open(address)
    ctypes.windll.user32.MessageBoxW(None, f"已在浏览器中打开：{address}\n点“确定”退出程序。", TITLE, 0x40)


def main():
    server, address = app_server.start_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        open_window(address)
    except Exception:
        open_browser(address)
    app_server.stop_analysis()


if __name__ == "__main__":
    main()
