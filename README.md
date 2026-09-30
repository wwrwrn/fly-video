# 果蝇视频分析：六管各一只

本地分析六根试管、每管一只果蝇的视频。程序自动寻找试管、管底参考和背景深色横线，检测并跟踪可见果蝇，记录首次过线时间，输出带速率的视频和数据表。

**模型已包含在仓库：`assets/fly_yolo26n_best.pt`，约5.1 MB。下载完整仓库后无需另找best.pt，也无需重新训练。**

这是教学原型。速率是 `5.5 cm ÷ 从视频第0帧到首次过线的时间` 的候选平均爬升指标；`CHECK`结果需要人工复核，不应直接当作最终生物学测量。

## 首次使用：师兄只需要做这几步

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

## 视频要求与实验口径

- 六根直立试管横向排列，每管只有一只果蝇。
- 浅色背景、上部白色塞子，背景有清楚的深色横线。
- 当前终点是**背景深色横线**，不是试管白色环线。
- 当前实验标定距离固定5.5 cm，计时起点固定视频第0帧。
- 首次过线按检测框中心越到线以上约3像素、连续3帧确认；过线后停止该管目标跟踪。
- 每管固定身份 `T1-F001`～`T6-F001`。缺失时留空，不虚构轨迹，也不反复创建新个体。

自动标定失败会停止并提示原因。当前规则不是通用场景识别模型，不能保证任意不同拍摄布局都能成功。

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

`requirements-app.txt`固定应用依赖；PyTorch/torchvision从官方CPU或CUDA索引安装，版本为2.9.1/0.24.1。`requirements-train.txt`与训练脚本属于旧教学实验，不是师兄首次使用的安装入口。

原始阶段记录保留在 [docs/PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md)。旧多目标训练、QA、跟踪测试会引用作者的实验数据或训练输出；使用当前应用不需要运行这些历史脚本。

安装依据：[PyTorch官方版本与安装组合](https://pytorch.org/get-started/previous-versions/)。模型来源为已完成教学微调的YOLO26n，类别仅 `0: fly`。
