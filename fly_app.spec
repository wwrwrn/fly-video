# -*- mode: python ; coding: utf-8 -*-
"""打包：python -m PyInstaller fly_app.spec --noconfirm。窗口程序不含 torch，分析进程单独成 fly_worker.exe。"""

from PyInstaller.utils.hooks import collect_data_files

EXCLUDED = ["tensorboard", "IPython", "jupyter", "notebook", "pytest", "tkinter"]

window = Analysis(["fly_app.py"], datas=[("app/index.html", "app")],
                  excludes=EXCLUDED + ["torch", "torchvision", "ultralytics", "matplotlib"])
worker = Analysis(["analyze_single_fly_video.py"],
                  datas=[("assets/fly_yolo26n_best.pt", "assets"), ("assets/model_info.json", "assets")]
                  + collect_data_files("ultralytics") + collect_data_files("imageio_ffmpeg"),
                  hiddenimports=["ultralytics.nn.tasks", "lap"], excludes=EXCLUDED)

window_exe = EXE(PYZ(window.pure), window.scripts, exclude_binaries=True, name="果蝇攀爬分析", console=False)
worker_exe = EXE(PYZ(worker.pure), worker.scripts, exclude_binaries=True, name="fly_worker", console=True)

COLLECT(window_exe, window.binaries, window.datas, worker_exe, worker.binaries, worker.datas, name="FlyClimb")
