# 果蝇视频分析：六管各一只

本地分析六根试管、每管一只果蝇的视频。程序自动寻找试管、管底参考和背景深色横线，检测并跟踪可见果蝇，记录首次过线时间，输出带速率的视频和数据表。

**模型已包含在仓库：`assets/fly_yolo26n_best.pt`，约5.1 MB。下载完整仓库后无需另找best.pt，也无需重新训练。**

这是教学原型。速率是 `5.5 cm ÷ 从视频第0帧到首次过线的时间` 的候选平均爬升指标；`CHECK`结果需要人工复核，不应直接当作最终生物学测量。

## 下载即用（推荐）

不用装 Python，也不用敲命令，只支持 64 位 Windows 10/11：

1. 到本仓库的 [Releases](https://github.com/wwrwrn/fly-video/releases) 页面下载 `FlyClimb-Windows-CPU.zip`。
2. **完整解压**到任意文件夹，路径可以有中文和空格。不要在压缩包里直接运行。
3. 双击 `果蝇攀爬分析.exe`。第一次运行时如果出现 Windows 安全提示，点“更多信息”→“仍要运行”。
4. 把 MP4 或整个文件夹拖到窗口左侧，等待分析完成。

结果保存在“文档\果蝇攀爬分析\outputs\app_runs\分组名”，界面里点“打开结果文件夹”也能找到。关闭窗口即退出；还有任务时会先询问。全部分析都在本机完成，视频不会上传到网络，原视频也不会被复制、移动或修改。

左上角显示版本号。程序打不开或出错时，请把“文档\果蝇攀爬分析\logs\app.log”和截图一起发给作者。

此版本用 CPU 分析，一段约 16 秒、476 帧的视频，在作者的 20 线程电脑上约 45 秒，普通电脑会更慢。这段视频的结果与 GPU 版完全一致。窗口依赖 Windows 自带的 WebView2；如果缺失，程序会改用默认浏览器打开界面。

只适用于六管、每管一只、白板加一条终点线的拍摄布局；管底到终点线按 5.5 cm 计算。自动标定失败时，在界面里手动点选横线和管底即可。

## 从源码运行

适合要改代码或要用 NVIDIA 显卡加速的人：

1. 从GitHub下载完整ZIP并**解压**，或使用Git克隆。不要在压缩包内部直接运行。
2. 安装 **Python 3.11、64位**。安装时勾选pip、Tcl/Tk和Python Launcher；建议勾选“Add python.exe to PATH”。
3. 双击项目根目录的 **`setup_env.cmd`**。它会创建 `.venv-train`、安装依赖、检查随仓库模型和运行设备。首次安装需要联网，CUDA版PyTorch下载体积较大，请等待窗口显示完成。
4. 安装完成后，双击 **`run_single_fly.cmd`**，在弹出窗口选择MP4。也可把MP4拖到这个文件上。
5. 窗口会显示本次输出文件夹。等待进度完成后，打开其中的 **`annotated_single_fly.mp4`** 和 **`first_passage_results.csv`**。

项目可以解压到任意可写位置，例如 `D:\fly-video` 或桌面文件夹；路径中有空格或中文也可以。**不需要存在作者的 `F:\fly video` 目录。** 视频可放在任意目录，不会被移动或修改。

安装选择：检测到NVIDIA显卡时安装已验证版本的CUDA构建，其他电脑安装CPU构建。运行时实际检查CUDA运算；不可用时自动使用CPU，处理较慢。没有NVIDIA显卡也能做离线分析。

如果自动安装的GPU构建不适合电脑，可在项目目录的PowerShell中明确安装CPU版本：

```powershell
.\setup_env.cmd --accelerator cpu
```

已安装环境后可重复双击 `check_install.cmd` 检查模型、依赖、FFmpeg和设备。检查记录保存在 `outputs/install_check.json`。

## 交互界面

桌面版直接打开窗口；源码运行时双击 `run_app.cmd`，浏览器会打开本地页面 `http://127.0.0.1:8765/`。把视频或整个文件夹拖到左侧就会排队分析；拖文件夹时，文件夹名就是分组名（如 `W118 雄`）。

- 总览：各组过线率、速度分布和样本数，点一个点打开对应视频。
- 单段：标注视频、六管结果、爬升曲线、质量标记和标定预览，点曲线或结果卡片跳到对应时刻。
- 每段视频的平均速度：六管中只算过线的管，没过线的管没有速度，不计入；需复核的管计入平均并单独列出数量。单段页面和总览表都有，总览表可导出 CSV。
- 管理结果：左侧视频或分组旁的“⋯”（或右键）可以改分组、重命名分组、删除结果。删除的结果放进回收站，原视频不动。任务卡片上的 × 可以取消排队或停止分析。
- 自动标定失败时点任务卡片，在页面里按提示点横线两端和六个管底，确认预览后继续分析。

界面只在本机运行，视频不外传。桌面版直接读取原视频；源码版用浏览器时会上传一份副本，分析成功后自动删除。源码运行时结果在 `outputs/app_runs/分组名/`，界面也会显示 `outputs/batch_runs/` 下的批量结果；关闭黑色命令窗口即退出。

## 视频要求与实验口径

- 六根直立试管横向排列，每管只有一只果蝇。
- 浅色背景、上部白色塞子，背景有清楚的深色横线。
- 当前终点是**背景深色横线**，不是试管白色环线。
- 当前实验标定距离固定5.5 cm，计时起点固定视频第0帧。
- 首次过线按检测框中心越到线以上约0.04 cm（按画面比例换算，76像素/厘米时约3像素）、连续3帧确认；过线后停止该管目标跟踪。
- 每管固定身份 `T1-F001`～`T6-F001`。缺失时留空，不虚构轨迹，也不反复创建新个体。

结果同时给两种速度：`candidate_speed_cm_s` 是5.5 cm除以从第0帧起的过线用时，与旧结果可比；`net_climb_speed_cm_s` 是实际起步高度到终点线的距离除以起步后的用时（上升超过0.3 cm算起步），不含起步前的停留。

相机位置和缩放可以变化（已验证约37–74像素/厘米）。自动标定失败或不确定时会停止并提示原因，这时用 `calibrate_manual.py` 手动点选。当前规则不是通用场景识别模型，不能保证任意不同拍摄布局都能成功。

部分果蝇如果在首帧已经离开底部，完整5.5 cm除以视频起点用时可能高估速率。程序会标记明显偏高的情况；即使没有自动标记，仍应检查起始位置及终点。

## 结果在哪里

启动入口每次选择新的文件夹：

```text
outputs/single_fly_runs/视频文件名/
outputs/single_fly_runs/视频文件名_run1/
outputs/single_fly_runs/视频文件名_run2/
```

已有结果不覆盖。主要文件：

| 文件 | 用途 |
|---|---|
| `annotated_single_fly.mp4` | 带固定身份、轨迹、终点线、首次过线时间和候选速率的视频 |
| `automatic_geometry_preview.png` | 自动ROI、红色终点线、蓝色管底参考，优先检查这张图 |
| `first_passage_results.csv` | 每管首次过线时间、候选cm/s、状态和风险标记 |
| `single_fly_observations.csv` | 每帧每管状态与有效位置，缺失帧不补点 |
| `raw_detection_candidates.csv` | 原始检测候选，方便排查误检 |
| `single_fly_report.json` | 参数、设备、处理帧数和结果 |

画面中的 `v*` 是候选平均爬升指标。`CHECK` 表示需要复核；`candidate` 只表示未触发自动风险规则，仍需抽查。没有过线的管不输出零速率，而是保留空值和 `not_reached`。

OpenCV元数据帧数可能比实际能解码的帧数多，程序记录实际处理帧数。处理速度与视频播放FPS不同，视频时间用于计算，不使用程序运行耗时。

## 命令行使用

在项目根目录运行，Windows不需要激活环境：

```powershell
.\.venv-train\Scripts\python.exe launch_single_fly.py
.\.venv-train\Scripts\python.exe launch_single_fly.py 'D:\实验视频\新样本.mp4'
```

直接调用分析器，可指定设备、输出或另一份权重：

```powershell
.\.venv-train\Scripts\python.exe analyze_single_fly_video.py 'D:\实验视频\新样本.mp4' --device auto --output 'outputs/my_run'
.\.venv-train\Scripts\python.exe analyze_single_fly_video.py 'D:\实验视频\新样本.mp4' --device cpu --output 'outputs/cpu_run'
```

批量分析一个文件夹（子目录名形如 `14541 雄`）：

```powershell
.\.venv-train\Scripts\python.exe batch_single_fly.py 'D:\实验视频\年轻攀爬测试' --device auto
```

结果在 `outputs/batch_runs/文件夹名/`：`all_tubes.csv` 是每管结果，`group_summary.csv` 是分组统计（只统计未标记复核的结果），`failed_videos.csv` 是失败视频。已完成的视频再次运行时跳过。

某段视频自动标定失败时，手动点选终点线两端和六个管底，保存后重新运行批量命令即可：

```powershell
.\.venv-train\Scripts\python.exe calibrate_manual.py 'D:\实验视频\失败的视频.mp4'
```

默认模型已在仓库中，完整性信息见 `assets/model_info.json`。`--weights` 只在维护者明确更换检测模型时使用。

## 常见问题

**双击没有环境 / 找不到python**：先安装Python 3.11（64位），再运行 `setup_env.cmd`。安装失败时保留窗口错误信息，不要仅安装原阶段的OpenCV `requirements.txt`。

**缺少best.pt**：请下载完整仓库，确认 `assets/fly_yolo26n_best.pt` 存在。运行 `check_install.cmd` 会检查哈希。此版本不再要求作者本机的 `runs/` 训练目录。

**拖放没有反应**：双击实际文件 `run_single_fly.cmd`，用文件选择窗口。不要把MP4拖到网页或聊天里的链接上。

**CUDA不可用**：默认auto会使用CPU，较慢但可以分析。可用 `setup_env.cmd --accelerator cpu` 安装较小的CPU构建；当前不需要实验室服务器。

**标定失败或CHECK很多**：检查视频布局、亮度、背景线和首帧位置。这是输入/测量问题，不能仅凭换一块GPU解决。

Windows命令文件已以CRLF字节保存，Git属性禁止换行归一化，以兼容Git克隆和GitHub ZIP下载。

## 维护与历史实验

主要入口为 `launch_single_fly.py` → `analyze_single_fly_video.py`，自动标定在 `single_fly_geometry.py`，固定单目标关联在 `single_fly_tracker.py`。

重新训练检测器的流程：`build_pseudo_labels.py` 从视频生成伪标签数据集 `dataset_v2/`（标签不是人工真值），`train_fly_detector.py` 在现有模型上微调，`compare_fly_detectors.py` 在留出实验组上比较新旧模型。

`requirements-app.txt`固定应用依赖；PyTorch/torchvision从官方CPU或CUDA索引安装，版本为2.9.1/0.24.1。`requirements-train.txt`与训练脚本属于旧教学实验，不是师兄首次使用的安装入口。

原始阶段记录保留在 [docs/PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md)。旧多目标训练、QA、跟踪测试会引用作者的实验数据或训练输出；使用当前应用不需要运行这些历史脚本。

安装依据：[PyTorch官方版本与安装组合](https://pytorch.org/get-started/previous-versions/)。当前模型是在教学微调的YOLO26n基础上，用2026年10月60段视频的伪标签再次微调的版本，类别仅 `0: fly`；上一版保留为 `assets/fly_yolo26n_v1.pt`。

## 打包桌面版

在已有 Python 3.11 的 Windows 上，用只含 CPU 版 PyTorch 的单独环境打包：

```powershell
python -m venv .venv-build
.\.venv-build\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-build\Scripts\python -m pip install -r requirements-app.txt lap pywebview pyinstaller
.\.venv-build\Scripts\python -m PyInstaller fly_app.spec --noconfirm
.\.venv-build\Scripts\python package_release.py
```

产物是 `dist/FlyClimb-Windows-CPU.zip`，上传到 GitHub Releases，不要提交进仓库。
