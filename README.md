# 果蝇运动视频分析：视频读取、人工数据与首次教学检测训练

当前实现视频读取、试管 ROI、人工检测标签、YOLO Detection 教学训练，以及六管独立 ByteTrack baseline。
当前不包含最终运动指标、首次过终点线测量或软件 GUI。
下文第 0–2 阶段的限制和数量是当时的阶段记录；最新训练阶段说明见文末。

## 环境与文件

已在 Windows、Python 3.11.1 下使用独立 `.venv` 验证。依赖版本记录在
`requirements.txt`；OpenCV 安装包版本是 4.14.0.94，运行时版本显示 4.14.0。

| 文件 | 用途 |
| --- | --- |
| `video_utils.py` | 打开和释放视频、读取元数据和指定帧、保存图片 |
| `inspect_video.py` | 输出视频信息，保存指定帧，可选 OpenCV 静态预览 |
| `extract_frames.py` | 按秒数间隔抽帧，并保存 `frames.csv` 索引 |
| `requirements.txt` | 固定本阶段实际使用的依赖版本 |
| `.gitignore` | 忽略虚拟环境、缓存、视频和生成结果 |
| `README.md` | 使用说明、限制和实际验证结果 |

输入视频保留原位。图片和验证报告保存在 `outputs/` 下。

## 安装

当前项目的 `.venv` 已安装好，可以直接运行后面的命令。
在新环境重新安装时，使用 PowerShell：

```powershell
Set-Location 'F:\fly video'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

无需激活虚拟环境，直接使用它的 Python，避免 PowerShell 执行策略问题。

## 使用

下面命令均在 `F:\fly video` 目录运行。视频路径有空格时必须加引号。

仅输出视频参数：

```powershell
.\.venv\Scripts\python.exe inspect_video.py '20260817_202618.mp4'
```

读取、保存并显示第 150 帧（**帧号从 0 开始**，150 是第 151 张图像）：

```powershell
.\.venv\Scripts\python.exe inspect_video.py '20260817_202618.mp4' --frame 150 --show
```

在 OpenCV 静态图片窗口中按任意键关闭。不需要窗口时去掉 `--show`，
仍会保存 PNG，可以用图片查看器打开。这里仅使用 OpenCV 图像预览功能，
没有搭建软件 GUI。运行验证时已保存帧号 100 的图片。

自定义单帧输出位置：

```powershell
.\.venv\Scripts\python.exe inspect_video.py '20260817_202618.mp4' --frame 200 --output 'outputs/custom/frame_200.png'
```

从第 1 秒开始，每隔 2 秒抽一帧，最多保存 6 帧：

```powershell
.\.venv\Scripts\python.exe extract_frames.py '20260817_202618.mp4' --start 1 --interval 2 --count 6 --output-dir 'outputs/sample_from_1s'
```

默认参数为 `--start 0 --interval 1 --count 10`，默认抽帧目录为
`outputs/frames/视频名/`。到达视频末尾会提前结束并提示。

为保护后续标注，单帧保存遇到同名文件会报错，抽帧输出目录非空时也会报错。
重复执行时请指定新输出路径或新目录。

`frames.csv` 使用 UTF-8 BOM 编码，可用 Excel 打开。列含义：

- `filename`：保存的图片文件名。
- `frame_index`：从 0 开始的帧号。
- `requested_seconds`：用户设定的抽帧时间。
- `frame_seconds`：实际选中帧的估算时间，即帧号 / FPS。

抽帧时间按 FPS 换算为最近的帧号，可能存在不超过约半帧的取整偏差。
间隔不能小于一帧时长，以免重复抽取同一帧。
元数据帧数由 OpenCV 报告，时长采用帧数 / FPS 估算；对于变帧率视频，
这些时间不是逐帧的精确媒体时间戳。

## 提供的视频实测

视频：`20260817_202618.mp4`。

| 参数 | 实际读取结果 |
| --- | --- |
| 分辨率 | 1920 × 1080 |
| FPS | 29.998271（约 30） |
| 总帧数 | 428 |
| 时长 | 14.267489 秒（帧数 / FPS） |

单帧结果：`outputs/preview/20260817_202618_frame_000100.png`。
抽帧结果：`outputs/frames/20260817_202618/`，从 0 秒起每隔 2 秒，
共保存 6 帧，帧号为 0、60、120、180、240、300，并附带 `frames.csv`。

完整顺序解码实际读到 **425 帧**，与容器报告的 428 帧存在差异；默认
FFmpeg 后端无法读取帧号 425–427，脚本会明确报错，不会保存空图像。
OpenCV 5.0.0 和 4.14.0 均复现这一现象。Windows MSMF 后端虽可读到
428 帧，但报告了不同的分辨率和帧数，因此本阶段保留默认后端，
不使用它替换结果。差异原因尚未确定，不能据此断言原视频损坏。
后续精确计时前需进一步核对视频编码和时间戳。

验证覆盖已保存 PNG 与顺序解码图像逐像素一致、CSV 索引、静态帧显示、
越界帧号、非法参数、缺失文件、重复输出保护和到达视频末尾的处理。
机器可读验证记录见 `outputs/verification.json`。

## 第 0 阶段结束时的建议

先检查抽帧图片，确认果蝇大小、背景、光照、遮挡和成像清晰度，
选取少量代表性帧制定标注规则，再评估简单背景分割能否满足单帧检测需求。
完成这些准备后再决定检测方案，暂不进入个体跟踪和软件界面开发。

## 第 1 阶段：试管 ROI

新增 `annotate_rois.py`（鼠标框选）、`crop_rois.py`（直接复用配置裁剪）和
`roi_utils.py`（坐标校验、配置读写、预览和裁剪）。继续使用现有 OpenCV 和
NumPy，不增加依赖，不包含果蝇识别、YOLO、跟踪或软件 GUI。
标注窗口仅为 OpenCV 的临时图片交互窗口。

### 首次选择

```powershell
Set-Location 'F:\fly video'
.\.venv\Scripts\python.exe annotate_rois.py '20260817_202618.mp4' --frame 100 --config 'config/tube_rois.json' --output-dir 'outputs/roi_selected'
```

1. 从左到右，用鼠标左键拖动框选每根试管。建议保持一致口径，包含完整试管、塞子和底部。
2. 每框好一根，按 Enter（或空格）确认，依次生成 `tube_1` 至 `tube_6`。
3. 画错可重新拖动；Backspace 先清除当前未确认框，没有未确认框时撤销上一根。
4. 六根全部确认后，再按一次 Enter 保存。Esc 或关闭窗口取消，不改动配置。

窗口默认最大 1280 × 800，按比例缩小显示；所有保存坐标都会换算回原图像素。
保存后，脚本重新打开视频、重新加载 JSON，自动输出六张裁剪图和编号预览。
选框顺序就是编号顺序，程序不会自行判断或识别试管。

### JSON 格式与坐标

主配置 `config/tube_rois.json` 只有以下六个键，下面是用户在第 100 帧上鼠标框选后实际保存的值：

```json
{
  "tube_1": [274, 400, 474, 956],
  "tube_2": [505, 391, 696, 963],
  "tube_3": [756, 372, 935, 951],
  "tube_4": [982, 382, 1166, 941],
  "tube_5": [1209, 378, 1383, 939],
  "tube_6": [1446, 375, 1617, 936]
}
```

`[x1, y1, x2, y2]` 是原始图像的像素坐标，左上角原点为 `(0, 0)`，
x 向右、y 向下。右下边界不包含在 ROI 中，裁剪代码为
`frame[y1:y2, x1:x2]`，宽为 `x2-x1`，高为 `y2-y1`。

同时保存 `config/tube_rois.meta.json`，记录原图尺寸、视频路径、帧号和标注来源。
复用时请保留这两个文件；程序会拒绝尺寸不匹配的视频。
手工编写的纯坐标 JSON 也可使用，但缺少元数据时只能检查坐标边界，会提示人工确认尺寸。

### 重新选择 ROI

使用 `--overwrite` 明确替换原配置，同时选择新的图片输出目录：

```powershell
.\.venv\Scripts\python.exe annotate_rois.py '20260817_202618.mp4' --frame 100 --config 'config/tube_rois.json' --overwrite --output-dir 'outputs/roi_reselected_01'
```

也可换一个 `--config` 文件名保留旧配置。取消标注不会覆盖旧配置。

### 直接复用已有 ROI

```powershell
.\.venv\Scripts\python.exe crop_rois.py '20260817_202618.mp4' --frame 100 --config 'config/tube_rois.json' --output-dir 'outputs/roi_reuse_01'
```

复用到另一个视频时，只需替换视频路径；前提是**相机位置、试管位置、画面裁剪和分辨率一致**。
尺寸相同并不代表位置相同，程序不会自动校正相机移动或试管位移，请检查每次生成的预览。
此步骤不弹标注窗口，也不会重新选择 ROI。

输出包括 `tube_1.png` 到 `tube_6.png`、`roi_preview.png` 和 `crop_info.json`。
裁剪图保留原始像素，不缩放、不画框；预览图单独绘制六个矩形和编号。
尺寸报告中的 `crop_sizes_width_height` 记录每张图的 `[宽, 高]`。
图片目录非空时不会覆盖，重复运行请换新目录。

第 1 阶段只建立矩形区域；矩形大小不等于试管的实际物理尺寸。
本阶段完成后暂停，不开始检测训练、个体跟踪或 GUI 开发。

### 本次实际结果与验证

用户已在真实 OpenCV 窗口完成六个框选并确认保存。实际选择覆盖试管下部和底部附近，
不含塞子和上部管身；部分底部外缘贴近或略超出框线。这里忠实保存用户选择，未自动扩大 ROI。

| ROI | 宽 × 高（像素） |
| --- | --- |
| tube_1 | 200 × 556 |
| tube_2 | 191 × 572 |
| tube_3 | 179 × 579 |
| tube_4 | 184 × 559 |
| tube_5 | 174 × 561 |
| tube_6 | 171 × 561 |

六根试管横向排列，位置并非完全等距，管身有轻微倾斜，底部高度也不完全相同。
本次 ROI 宽度范围为 171–200 像素，高度为 556–579 像素，上边界相差 28 像素，
下边界相差 27 像素。选框、透视和试管摆放都会影响这些数值，不能把 ROI 差异直接视为物理尺寸差异。

实际裁剪及预览保存在 `outputs/roi_selected/`。再次调用 `crop_rois.py` 的结果保存在
`outputs/roi_reuse_verified/`，已确认六张图与首次输出、原始帧对应区域逐像素一致，预览也一致。
同时检查了另一帧加载配置、坐标越界、原图尺寸变化、非法帧号、取消和撤销、重复输出保护等。
鼠标坐标换算与快捷键分支另用模拟事件验证，报告为 `outputs/roi_script_verification.json`。

## 第 2 阶段：候选图片与人工果蝇框

本阶段只制作、标注和验证数据，不安装 YOLO，不下载模型，不训练模型，不做跟踪或运动指标计算。
继续使用已有 OpenCV 和 NumPy，标注窗口为简单的临时图片窗口。

### 活动范围确认

用户已重新框选 ROI，并明确说明：未框入的上部为棉花，果蝇无法到达。
因此完整活动区域不等于整个外观管身；此次按用户确认的活动范围生成数据。
之前关于“必须包含上部管身”的判断已由这项实验信息修正。
已检查 frame 0、80、160、240、320 的 ROI 预览，文件在 `outputs/stage2_roi_review/`。

`config/tube_rois.meta.json` 已记录这项确认及坐标摘要。候选图生成脚本检查摘要；
重新选择或修改 ROI 后，旧确认失效，必须重新检查其是否覆盖实际活动区域。
不要仅为绕过检查而修改确认记录。

### 新文件与数据目录

| 文件 | 用途 |
| --- | --- |
| `build_dataset_sample.py` | 从五个固定帧裁剪六根试管，恰好生成 30 张候选图片 |
| `annotate_flies.py` | 人工逐只画框、撤销、存草稿、保存标签并切换图片 |
| `yolo_labels.py` | 图像读取、坐标转换、标签校验与读写、框反绘 |
| `verify_yolo_labels.py` | 独立读取一张图片和同名 txt，保存反绘预览 |
| `test_stage2.py` | 运行候选图、标签及交互逻辑的自动验证 |

```text
dataset_sample/
  images/frame_000000_tube_1.png   # 候选原图，不带标注线
  images/...                      # 共 30 张
  labels/frame_000000_tube_1.txt  # 人工完成并保存后才创建
  annotations/...json            # 保存时的像素框，方便审计与验证
  drafts/...json                 # 未完成标注，不作为训练标签
  manifest.json                  # 每张图的来源帧、试管编号、ROI、尺寸
  dataset.example.yaml           # 单类别数据配置格式示例
```

固定帧号为 0、80、160、240、320，均从 0 开始。
六根试管图片宽高依次为 219×597、204×594、204×601、195×575、184×575、177×575 像素。
保留原始像素，不统一拉伸为同一尺寸。

生成命令（当前已运行，重复时指定新的空目录）：

```powershell
Set-Location 'F:\fly video'
.\.venv\Scripts\python.exe build_dataset_sample.py '20260817_202618.mp4' --output-dir dataset_sample
```

### Bounding Box 和 YOLO 标签

Bounding Box 是包住一只果蝇的矩形框。每只果蝇一个框，尽量紧贴可见轮廓，
避免把相邻两只合成一个框。不可辨认的重叠个体不要猜测，可保存草稿待人工复查。
完成一张图时，应确认所有可辨认个体均已标注；只画一个示例框不代表整图标注完成。

标签每只果蝇一行，共五列：

```text
class x_center y_center width height
```

- `class`：类别编号，本项目固定 `0`，表示 `fly`，不是个体 ID。
- `x_center`、`y_center`：框中心相对于当前试管裁剪图宽、高的位置。
- `width`、`height`：框宽、框高相对于当前裁剪图宽、高的比例。

如果当前图片宽为 W、高为 H，像素框为 `[x1, y1, x2, y2]`：

```text
x_center = (x1 + x2) / (2 * W)
y_center = (y1 + y2) / (2 * H)
width    = (x2 - x1) / W
height   = (y2 - y1) / H
```

四个坐标数值应在 0～1 内，宽高必须大于 0。归一化使不同尺寸图片都可以使用同一种比例表示。
分母必须使用当前裁剪图的宽高，不能使用整段视频的 1920×1080，也不能使用窗口放大后的尺寸。
例如 100×200 的图片中 `[10,20,30,60]` 对应 `0 0.200000000 0.200000000 0.200000000 0.200000000`。
这只是公式示例，实际人工标注的像素框和文本记录在 `annotations/`。

文件按同名对应：`images/frame_000000_tube_1.png` 对应
`labels/frame_000000_tube_1.txt`。尚未标注的图片没有 txt，不能因此认定没有果蝇。
只有人工确认整图无目标后，才允许保存空 txt。本阶段不建立训练集清单，也不使用未标注候选图训练。
格式依据：[Ultralytics 官方 Detection 数据格式](https://docs.ultralytics.com/datasets/detect/)。

### 人工标注操作

```powershell
.\.venv\Scripts\python.exe annotate_flies.py --dataset dataset_sample
```

每次显示一张图片，自动从未完成的图片继续；已有标签会跳过，有草稿会优先恢复。

- 鼠标左键拖动并松开：添加一个框，可连续添加多个框。
- `U` 或 Backspace：撤销上一个框。
- Enter 或 `N`：确认整张图标注完成，保存同名 txt 并切换下一张。
- `D`：保存未完成草稿并退出；不创建或修改训练标签。
- Esc 或关闭窗口：退出，放弃当前未保存修改，之前保存的内容不变。
- `E` 后 Enter：仅当整张图确实没有果蝇时，确认并保存空标签。

图像会适度放大显示，坐标自动映射回原图。要修改已保存的标签，使用
`--review-existing`；主动保存会替换该图旧标签。只测试一张可加 `--limit 1`。

### 独立标签反绘验证

```powershell
.\.venv\Scripts\python.exe verify_yolo_labels.py 'dataset_sample/images/frame_000000_tube_1.png' 'dataset_sample/labels/frame_000000_tube_1.txt' --output 'outputs/yolo_check_01.png'
.\.venv\Scripts\python.exe -X utf8 test_stage2.py
```

验证脚本检查类别、列数、有限数值、归一化范围及整个矩形是否越界，再将坐标反算为像素，
在原图副本上画框并标注 `fly`。打印像素坐标便于对照原始手工框。
这能验证格式与坐标，不能自动判定是否漏标、重叠目标判断是否准确或框是否足够紧。
输出图片同名时拒绝覆盖，请换新路径。

`dataset.example.yaml` 使用 `names: {0: fly}`，示意以后通过 `train.txt` 和 `val.txt` 指定划分。
本阶段没有生成这些划分清单，示例不能直接训练。之后需在标签审核完成后设计划分，
避免把同一短视频的相邻帧随意分入训练和验证集而高估效果。本次不开始该工作。

### 本次真实人工标注结果

30 张候选图已生成。用户在真实窗口中完成 `frame_000000_tube_1.png` 的标注，
保存了 10 个框；其他 29 张仍待人工标注，没有伪造空标签。

该图片尺寸为 **219×597**。用户保存的第一个像素框为 `[151,515,173,535]`，
中心为 `(162,525)`，宽高为 `(22,20)`，对应：

```text
x_center = 162 / 219 = 0.739726027
y_center = 525 / 597 = 0.879396985
width    = 22 / 219  = 0.100456621
height   = 20 / 597  = 0.033500838
```

实际标签第一行：

```text
0 0.739726027 0.879396985 0.100456621 0.033500838
```

独立验证工具已运行。十个框的反算浮点误差最大约 `3.41e-7` 像素，
取整后与原始人工矩形完全一致，反绘图片也与原始框绘图逐像素一致。
预览为 `outputs/stage2_label_check/frame_000000_tube_1_preview.png`，
验证报告为 `outputs/stage2_verification.json`。
7 项自动测试通过；本阶段验证了数据与标注流程，尚未完成全部候选图片的人工标注。

## 首次教学检测训练

### 本阶段实验规则

只检测当前帧中可见的果蝇，类别始终只有 `0: fly`。
棉花中完全不可见时不猜测 Bounding Box；棉花边缘有明确可辨认部分时仍按可见目标标注。
是否为同一个体、遮挡后身份恢复和轨迹都属于未来跟踪阶段，本阶段没有实现。

### 人工标签 QA 与时间划分

目前 30 张图片均已有人工作出的同名 TXT，共 **278 个标注实例**，不是 278 只独立个体。
只读检查覆盖文件配对、类别、有限数值、归一化范围、正宽高、整框边界和疑似重复框。
疑似重复阈值设为两框 IoU ≥ 0.85，发现异常则停止，不自动修复标签。
已生成 30 张反绘预览、5 张联系表和逐图数量 CSV：`outputs/training_qa/`。
程序校验不能取代对低清晰度、重叠目标和漏标的人工判断。

| 划分 | 帧号 | 图片数 | 标注实例数 |
| --- | --- | --- | --- |
| train | 0、80、160、240 | 24 | 229 |
| val | 320 | 6 | 49 |

`dataset_yolo/` 包含 `images/train`、`images/val`、`labels/train`、`labels/val` 和 `dataset.yaml`。
图片与标签均从原数据复制，保留全部原始 `dataset_sample` 文件；复制结果按 SHA-256 核对。
**训练与验证来自同一短视频，只能验证流程，不能代表跨视频、跨拍摄条件的泛化能力。**

重新质检和准备新的副本时：

```powershell
Set-Location 'F:\fly video'
.\.venv\Scripts\python.exe qa_training_labels.py --output outputs/training_qa_new
.\.venv\Scripts\python.exe prepare_yolo_dataset.py --qa-report outputs/training_qa_new/qa_report.json --output dataset_yolo_new
```

这两条命令不会覆盖已有输出。训练脚本默认使用 `dataset_yolo`，开始前还会验证 QA 和原始文件哈希。

### 两个独立 Python 环境

- `.venv`：原有 OpenCV 视频读取、标注和数据 QA 环境，保持原依赖。
- `.venv-train`：独立训练与推理环境，使用官方 CUDA 版 PyTorch 和 Ultralytics。

本机为 Windows、Python 3.11，GPU 实际名称为 NVIDIA GeForce RTX 3070 Ti Laptop GPU，8GB 显存。
使用 PyTorch 官方提供的 `torch==2.9.1`、`torchvision==0.24.1` CUDA 12.8 wheel。
`nvidia-smi` 的 CUDA 版本表示驱动能力，`torch.version.cuda` 才是此 PyTorch 构建使用的 CUDA 运行时版本。

```powershell
python -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install --upgrade pip
.\.venv-train\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv-train\Scripts\python.exe -m pip install -r requirements-train.txt
```

实际环境信息、全部依赖版本、训练参数和训练结果分别记录在
`outputs/training_environment.json`、`requirements-train.lock.txt`、
`outputs/training_request.json`、训练目录 `args.yaml` 和 `outputs/training_completion.json`。

### 模型与运行入口

本次选择 `yolo26n.pt` 预训练 Detection 权重，使用轻量 nano 规模，加载预训练权重后微调单类别检测头。
固定教学设置为 epochs=60、imgsz=640、batch=4、device=0、AdamW、workers=0、patience=20、nbs=16。
使用矩形输入、AMP，关闭 mosaic、mixup、copy-paste、旋转和上下翻转；仅保留轻微平移、尺度变化、亮度变化和左右翻转。
具体实际生效参数以运行目录 `args.yaml` 为准。没有参数搜索或服务器训练。

```powershell
.\.venv-train\Scripts\python.exe train_flies.py
.\.venv-train\Scripts\python.exe evaluate_flies.py
.\.venv-train\Scripts\python.exe predict_flies.py 'dataset_yolo/images/val/frame_000320_tube_1.png'
```

训练默认保留在 `runs/detect/fly_yolo26n_teaching/`，包括 `results.csv`、`results.png`、
`weights/best.pt`、`weights/last.pt` 及 Ultralytics 生成的其他诊断图。
已有同名训练目录时会拒绝覆盖。单图入口默认加载这里的 `best.pt`；其他权重可用 `--weights` 指定。
输出为数量、每个检测的置信度与像素框，以及带 `fly` 和 confidence 的预测图片；不生成个体身份编号。

独立验证使用 best.pt。指标计算采用验证模式的低置信度门槛 0.001，保留 PR 曲线；
逐图数量比较固定使用 `conf=0.25`、`iou=0.7`。PR/mAP 指标与固定阈值下的数量比较不是同一统计口径。
预测数量相同也不意味着所有框都匹配正确。漏检、误检均不用于自动修改人工标签。

预测输出在 `outputs/validation_predictions/`，包括六张结果图、联系表、逐图 JSON、数量对比 CSV 和指标 JSON。
重复运行请指定新输出目录。运行完成后停止，不继续跟踪阶段。

参考：[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)、
[训练参数](https://docs.ultralytics.com/modes/train/)、
[PyTorch 官方 CUDA 安装组合](https://pytorch.org/get-started/previous-versions/)。

### 本次已完成的实际结果

详细报告见 [TRAINING_REPORT.md](TRAINING_REPORT.md)，机器可读结果为 `outputs/final_training_report.json`。
共完成一次训练：上限60轮，第51轮早停，best.pt 来自第31轮。
独立 best.pt 验证（batch=4）得到 Precision=0.736250、Recall=0.740677、mAP50=0.701954、mAP50-95=0.212657。
训练结束时的自动验证采用其原有加载配置，数值另列于报告，没有与独立验证结果混用。

在 confidence≥0.25 时，六张验证图人工数量为 `[7,7,9,8,9,9]`，预测为 `[7,7,12,7,9,9]`。
原始图片和人工标签哈希全部保持不变。已实际运行单图入口，所有训练结果和预测图均已保留。
本次工作已停止在检测阶段，没有启动个体跟踪或后续运动分析。

## 第 3.5 阶段：完整视频逐帧检测与数量稳定性

继续使用当前 `best.pt` 和六根试管 ROI，不训练模型，不分配个体 ID。
固定 conf=0.25、imgsz=640、IoU=0.7、max_det=100、rect=True，每根试管单独在内存中推理，
保持与此前单图验证一致的输入处理。不保存逐帧 ROI 图片。

新增 `detect_full_video.py`（视频流检测及CSV/视频输出）、
`analyze_detection_stability.py`（统计、图表和少量异常截图）、
`test_full_video_detection.py`（坐标与输出一致性验证）。
训练环境新增 `imageio-ffmpeg==0.6.0` 用于 H.264 编码，原 `.venv` 不变。

```powershell
Set-Location 'F:\fly video'
.\.venv-train\Scripts\python.exe detect_full_video.py '20260817_202618.mp4' --output 'outputs/full_video_detection'
.\.venv-train\Scripts\python.exe analyze_detection_stability.py --input 'outputs/full_video_detection'
.\.venv-train\Scripts\python.exe -X utf8 test_full_video_detection.py
```

当前上述任务已运行完成。重新检测请指定新输出目录；已有结果不覆盖。
输出包括：

- `annotated_detection.mp4`：1920×1080、约29.998271 FPS、H.264，保留ROI、fly框、confidence、frame和time，不含音频。
- `full_video_detections.csv`：每行一条检测，包含frame_idx、time_sec、tube_id、detection_index、confidence、局部xyxy、局部中心、宽高及全局中心。
- `detection_counts.csv`：每个成功处理帧一行，六根试管数量及total。
- `performance.json`：处理数量、参数、时间、CUDA状态和PyTorch显存统计。
- `stability_report.json`：每管均值、最小/最大、总体标准差、数量异常提示及最不稳定时间段。
- `count_change_alerts.csv`：相邻源帧同一试管增减至少3个检测框的提示。
- `suspect_overlapping_detections.csv`：同帧同管IoU≥0.5的重叠框候选，不自动判错或去重。
- `plots/`：六管各一张曲线，加一张总览，共7张。
- `anomaly_frames/`：10个代表性事件的前/中/后三帧，共30张完整截图。
- `review_panels/`：3张抽查局部对照图，上方原图、下方已有检测框，用于核对漏检与重复。
- `anomaly_manifest.json`、`visual_review.json`、`verification.json`：截图来源、人工查看结论和校验记录。

`detection_index` 在每帧、每根试管内从1重新开始，仅是检测列表序号，**不是Fly身份编号**。
局部框坐标相对于当前tube裁剪图。全局中心等于局部中心加ROI左上角偏移。
`time_sec=frame_idx/FPS` 为估算时间，不是变帧率视频的精确逐帧PTS。

此次处理425帧（0–424），元数据仍报告428帧；425、426、427尝试读取失败，没有补造图像。
输出视频已完整顺序解码验证为425帧，时长约14.167秒。若其他视频出现中间坏帧，
脚本会尝试后续帧，输出仅包括成功读取帧；失败位置写入报告，画面中的frame/time保留源帧位置。

数量标准差为总体标准差（ddof=0）。不稳定时间段按1秒窗口内平均相邻帧数量绝对变化排名，
排除不足0.5秒的尾部窗口。持续低检出是相对过去约1秒中位数减少至少3、连续至少3帧，
不假定真实可见数量恒定。所有异常提示都需要人工判断，不能直接等同于漏检。

实际完整处理流程48.45秒，约8.77源视频帧/秒，平均114.00毫秒/源帧；
仅六管推理合计32.62秒，约13.03源帧/秒。加载和预热另用2.79秒，包含这部分约8.29 FPS。
统计、截图和图表不计入上述检测流程速度。PyTorch allocator峰值约50.5MiB，
不包含驱动上下文、其他CUDA分配和其他应用显存，不能解释为整块GPU的总峰值。

最不稳定区间为0–1秒，其次1–3秒和5–6秒。没有整管连续零检出，但抽查tube_4的frame320–321
能看到局部连续漏检，tube_3顶部及tube_4开头有明显重复框。详见
[FULL_VIDEO_DETECTION_REPORT.md](FULL_VIDEO_DETECTION_REPORT.md)。
当前可用于离线教学Tracker baseline和失败案例分析，尚不足以保证可靠身份或运动量化。
本阶段没有运行或实现任何Tracker、轨迹或运动指标。

## 第4阶段：六管独立ByteTrack baseline

本轮只运行当前 Ultralytics 8.4.143 的一组默认ByteTrack，保持YOLO26n best.pt不变。
六根试管各有独立tracker、轨迹池、状态估计和本地ID计数器，显示ID为 `T1-F001`、`T2-F001` 等。
适配层只隔离ID计数和记录审计，关联与滤波规则沿用官方实现；对六管425帧的原生结果复核一致。

### 基本概念

- detection：某一帧上模型给出的一个可见果蝇框及置信度。
- track：tracker认为属于同一目标的一串跨帧观测，中间可能短暂缺失。这个关联判断可能出错。
- track_id：这条轨迹的身份标签，在同一试管内持续使用；不是经过生物学确认的个体身份。
- detection_index：每帧检测列表中的序号，框的排序和数量变化就会改变，因此不能直接当track_id。

ByteTrack先用高置信度检测与现有轨迹关联，再尝试用低置信度检测维持未匹配的已跟踪目标，
减少直接丢弃低分框造成的断裂。完全不可见、长期遮挡、外观相似或重复检测仍可能导致失败。
参考：[ByteTrack论文](https://arxiv.org/abs/2110.06864)、[Ultralytics跟踪文档](https://docs.ultralytics.com/modes/track/)。

### 本次参数与输出含义

| 参数 | 实际值 |
| --- | --- |
| track_high_thresh | 0.25 |
| track_low_thresh | 0.10 |
| new_track_thresh | 0.25 |
| track_buffer | 30个源帧，约1秒 |
| match_thresh | 0.8（关联代价阈值，不是IoU必须≥0.8） |
| fuse_score | True |
| detector conf / imgsz / IoU | 0.10 / 640 / 0.7 |
| history | 最近1.5秒；缺帧处不连线伪造观测 |

相比第3.5阶段，检测最低置信度从0.25改为0.10，以提供ByteTrack第二次关联需要的低分框。
没有重新训练、参数搜索或同时比较其他tracker。新增依赖只有训练环境中的 `lap==0.5.13`，原 `.venv` 不变。

`outputs/bytetrack_baseline/` 中：

- `annotated_tracking.mp4`：425帧，1920×1080，约29.998271FPS，ID、置信度、ROI、frame/time和短历史线。
- `tracks.csv`：22,447行当帧有检测支持的确认轨迹。xyxy/center为ByteTrack更新后的ROI局部框，另附原检测框与全局中心。
- `track_summary.csv`：全部1,871个创建过的ID，包括未确认、极短和被原生规则移除的轨迹。
- `track_states.csv`：tentative/tracked/lost/removed/pruned_native状态、最后观测帧及当前是否有观测。
- `track_events.csv`：创建、状态变化、原生重新关联事件。
- `detections.csv`：这次保留到0.10的检测输入，便于复核低分关联。
- `active_track_counts.csv`：每帧每管确认活跃轨迹数量。
- `bytetrack_used.yaml`、`tracking_report.json`：实际参数、计数和性能。
- `review/`：前1秒片段、23张重点帧截图、4张原图/ID对照图、指定窗口的逐帧审计CSV。
- `plots/`：活跃数量与轨迹生命周期图。

summary中的 `duration_frames=end_frame-start_frame+1`，从首次匹配检测到最后匹配检测，包含内部缺口，
不包含末尾等待超时的lost缓冲。`duration_sec=duration_frames/FPS`。
**track duration ≠ movement time。** 本轮没有计算运动/静止、速度或距离。
`number_of_detected_frames`包含候选激活帧，`number_of_output_frames`只计正式确认输出，二者可能不同。
`missing_gaps`等字段记录内部缺口；`censored_at_video_end`说明视频结束时轨迹是否仍在保留中。
极短轨迹定义为生命周期≤5帧，另记观测帧数≤5的稀疏轨迹，不自动删除。

当前Ultralytics在过期边界存在一个实现细节：历史removed列表中的对象可能在下一次更新被原生重新关联。
审计以当前live池/对象状态为准，不能仅看历史列表。已用边界测试验证，并从保存的检测重建审计。
`tracks.csv`、检测CSV和跟踪视频保持不变，原生算法没有修改。

### 实际结果与失败模式

| Tube | 全部创建ID | 曾确认ID | 活跃均值（最小–最大） | 最长生命周期/帧 | ≤5帧轨迹 |
| --- | ---: | ---: | --- | ---: | ---: |
| tube_1 | 262 | 58 | 8.58（5–12） | 408 | 214 |
| tube_2 | 374 | 60 | 8.75（6–11） | 404 | 328 |
| tube_3 | 349 | 63 | 8.39（5–11） | 425 | 295 |
| tube_4 | 296 | 65 | 8.07（5–18） | 415 | 245 |
| tube_5 | 184 | 33 | 9.41（7–12） | 425 | 157 |
| tube_6 | 406 | 74 | 9.61（6–13） | 404 | 348 |

最短生命周期均为1帧；共1,587条≤5帧轨迹。全部1,871个创建ID中，1,518个从未确认，353个曾确认。
这些都不是独立果蝇数量。即便只看曾确认ID，每管33–74个仍明显超过实验中约8–12只的规模，
因此不能把本次关联当作可靠个体身份。

- 0–1秒（frame0–29）：创建434个ID，其中首帧72个、之后362个；119个在这一秒内曾确认输出。
  底部重叠和重复检测使起点编号就存在问题。首帧原有人工作出的标签共60个，不能把72个tracker输出都当成真实个体。
- tube_4 frame319–322：`T4-F240`保持原ID，置信度依次约0.317、0.235、0.205、0.105；后三帧依靠低分关联维持，
  不是先完全丢失再恢复。frame323进入lost，保留到352，353移除；截至424没有恢复该ID。
- tube_3 frame319–322：`T3-F229`已在跟踪顶部目标；重复检测在320创建`T3-F278`，321确认，随后两个ID覆盖同一可见目标区域。
- 起步另一例：tube_3的同一孤立目标在frame0–1同时显示`T3-F005`与`T3-F008`，之后后者丢失。

本次明确观察到虚假重复ID和轨迹碎片化。没有连续帧身份真值，不能据此给出严格ID switch次数；
抽查也不足以可靠确认两个真实个体的ID互换。不能把每次新建ID都算作ID switch，也不能把425帧长轨迹当成身份正确的证明。

### 运行与后续实验边界

```powershell
.\.venv-train\Scripts\python.exe track_video_baseline.py '20260817_202618.mp4' --output 'outputs/bytetrack_baseline_new'
.\.venv-train\Scripts\python.exe analyze_tracking_baseline.py --input 'outputs/bytetrack_baseline_new'
.\.venv-train\Scripts\python.exe -X utf8 test_bytetrack_baseline.py
```

本次已经运行完成。完整流程约44.47秒、9.56源帧/秒；没有上服务器。
详见 [BYTETRACK_BASELINE_REPORT.md](BYTETRACK_BASELINE_REPORT.md)。

用户已将后续实验目标明确为“从底部到终点线的首次到达平均爬升速度”，过线后不再纳入该个体测量。
这能规避过线后的顶部遮挡，但不能解决起点到终点之前的身份错误。
终点位置、标定距离和计时规则尚未正式配置，没有实施任何过线或速度计算。
当前baseline不满足直接量化的要求。更需要起步和到线前的连续帧人工身份核对；若人工也无法辨认重叠个体，
应补拍更清晰的新样本。现阶段不需要更强服务器。具体实验约定见 [EXPERIMENT_NOTES.md](EXPERIMENT_NOTES.md)。

## 六管各一只的新视频：本地自动分析原型

用户的新拍摄协议仍为六管，每管一只。终点已明确为**背景深色横线**，计时从**视频第0帧**开始，
管底到横线标定距离为5.5 cm。新视频位置不同，程序自动重新寻找六个白色塞子、六个管底和背景横线；
标定不满足六管/横线规则时停止，不复用旧视频的ROI坐标。

`single_fly_geometry.py`负责图像标定及相机小幅平移检查；`single_fly_tracker.py`负责每管固定一个ID、
按位置选择检测、记录缺失/歧义及确认首次过线；`analyze_single_fly_video.py`是视频文件入口。
它们使用已有YOLO26n best.pt，只做离线分析。`test_single_fly.py`验证标线、空间连续性、缺失、固定身份和跨线计时。

```powershell
Set-Location 'F:\fly video'
.\.venv-train\Scripts\python.exe analyze_single_fly_video.py 'video/20260921_193819.mp4'
.\.venv-train\Scripts\python.exe analyze_single_fly_video.py 'video/20260921_193846.mp4'
.\.venv-train\Scripts\python.exe -X utf8 test_single_fly.py
```

默认结果存入 `outputs/single_fly_analysis/视频文件名/`，已有目录非空时拒绝覆盖；换视频可用 `--output` 指定新目录。
在Windows上，建议用文件管理器进入 `F:\fly video`，**双击 `run_single_fly.cmd`**，在弹出的窗口选择MP4；
如果文件管理器支持，也可把MP4拖到该命令文件上。入口由 `launch_single_fly.py` 接收文件路径，
每次自动选择 `outputs/single_fly_runs/视频名` 或追加 `_runN` 的新目录，不覆盖已有结果。
输出视频在试管旁同时显示首次过线时间、`v*`候选cm/s和状态：`CHECK`表示需要复核，`candidate`表示未触发自动风险规则；都不是未经人工核验的最终实验值。
生成 `annotated_single_fly.mp4`、`automatic_geometry_preview.png`、
`single_fly_observations.csv`（每帧每管状态和有效中心点）、`raw_detection_candidates.csv`、
`per_frame_status.csv`、`first_passage_results.csv`、`single_fly_report.json` 及复核图片。
六管身份固定为 `T1-F001`～`T6-F001`，不再因漏检自动创建新编号。

起点和终点像素位置来自图像估计。背景线较淡、首帧果蝇已高于管底、失去目标后重新选择、
或首次跨线处画面配准不稳，都会在 `quality_flags` 中说明，`speed_quality=review`。
`candidate_speed_cm_s` 是 `5.5 / 首次过线视频秒数` 的候选值；`auto_candidate` 只表示程序规则未触发风险标记，
仍应抽查图像后才作为实验结果。没过线或首次看到时已经在线上方的个体不输出零速率。
前两秒选择标定参考帧，检测时只在内存中裁剪，不保存全部ROI帧。轨迹缺口不插值，首次跨线需要连续三帧确认。

这套固定场景自动定位已在当前两段视频通过；这是原型，不能推断其他位置、亮度和拍摄条件下都能自动成功。
先看 `outputs/single_fly_analysis/` 的标定图、过线图和结果质量标记。完整检查见
[SINGLE_FLY_REPORT.md](SINGLE_FLY_REPORT.md)。目前两段离线处理均在本机完成，不需要实验室服务器。
