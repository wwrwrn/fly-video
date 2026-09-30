# 第3.5阶段：完整视频检测与时间稳定性报告

已使用现有YOLO26n best.pt对整段可解码视频逐帧检测。ROI仅在内存中裁剪，没有重新训练，没有身份编号、关联、滤波、轨迹或运动量化。

## 处理范围与数据

实际处理 **425帧（frame 0–424）**，保存 **23,646条检测记录**。视频元数据报告428帧，末尾425、426、427逐帧尝试均失败，没有补造检测。

固定参数：conf=0.25、imgsz=640、IoU=0.7、max_det=100、rect=True，每个源帧的6个ROI分别推理。模型和ROI均保持不变。

输出视频为1920×1080、29.998271 FPS、H.264/yuv420p，无音频。完整重新解码得到425帧，时长约14.167秒。画面标出源frame_idx和估算时间，不显示任何永久身份。

[annotated_detection.mp4](<F:/fly video/outputs/full_video_detection/annotated_detection.mp4>)；[每帧每管数量CSV](<F:/fly video/outputs/full_video_detection/detection_counts.csv>)；[全部检测明细CSV](<F:/fly video/outputs/full_video_detection/full_video_detections.csv>)

## 每管检测数量

| Tube | 平均/帧 | 最小 | 最大 | 总体标准差 | 相邻帧平均绝对变化 |
|---|---:|---:|---:|---:|---:|
| tube_1 | 9.021 | 5 | 14 | 1.759 | 0.509 |
| tube_2 | 9.252 | 6 | 14 | 1.409 | 0.663 |
| tube_3 | 9.021 | 6 | 13 | 1.341 | 0.724 |
| tube_4 | 8.511 | 5 | 18 | 1.782 | 0.604 |
| tube_5 | 9.673 | 8 | 13 | 0.896 | 0.420 |
| tube_6 | 10.160 | 7 | 14 | 1.120 | 0.712 |

数量代表当帧通过置信度门槛的框数，不是真实独立果蝇数；可见数量变化也可能来自顶部遮挡。整体标准差还包含缓慢数量趋势，不能单独解释为检测抖动。

## 最不稳定的时间段

按1秒窗口内各管相邻帧数量绝对变化的平均值排序；不足0.5秒的尾部窗口不参与排名。

| 时间段（秒） | 波动分数 | 增减≥3的提示次数 |
|---|---:|---:|
| 0.00–1.00 | 1.115 | 18 |
| 1.00–2.00 | 0.761 | 4 |
| 2.00–3.00 | 0.717 | 5 |
| 5.00–6.00 | 0.717 | 3 |
| 4.00–5.00 | 0.606 | 2 |

全视频共46条数量骤变提示：下降22次、增加24次。最明显的早期例子是tube_4从frame0的18框变成frame1的13框；这些提示不自动判定为模型错误。

![检测数量总览](<F:/fly video/outputs/full_video_detection/plots/detection_counts_overview.png>)

## 连续漏检与重复框

没有发现整根试管连续零检出的区间，这不代表不存在局部漏检。统计上的持续低检出提示有：tube_1的frame17–20（约0.57–0.67秒）和tube_5的frame15–17（约0.50–0.57秒）；重复框减少或可见数量变化也会触发。

人工抽查发现：**tube_4的frame320–321（约10.67–10.70秒），顶部一处仍可辨认的小目标连续没有检测框；frame319该处有框。** 这是局部图像证据，不是全视频逐帧真值评估，不能据此给出完整漏检率或所有漏检持续时长。

[局部连续漏检对照：tube_4 frame319–321](<F:/fly video/outputs/full_video_detection/review_panels/tube_4_frames_319_321.png>)

明显重复的例子：

- tube_4 frame0–2：底部目标出现重复覆盖框，数量18→13→12。
- tube_3 frame319–321：顶部原图变化较小，右侧目标由一个框变成多个高重叠框，该管数量8→12→11。

自动检出1572对IoU≥0.5的重叠候选，涉及415个源帧。这个数字仅表示重叠候选，不等于确认了这么多错误；真实相互遮挡也会产生框重叠。没有再次做NMS或删框。

![tube_3重复框对照](<F:/fly video/outputs/full_video_detection/review_panels/tube_3_frames_319_321.png>)

[30张异常前后帧截图](<F:/fly video/outputs/full_video_detection/anomaly_frames>)；[截图选择原因和时间](<F:/fly video/outputs/full_video_detection/anomaly_manifest.json>)；[3张局部对照图](<F:/fly video/outputs/full_video_detection/review_panels>)

## 性能

| 项目 | 实际值 |
|---|---|
| CUDA | True，NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| 实际处理帧 | 425 |
| 完整检测流程时间 | 48.451秒 |
| 平均每源视频帧耗时 | 114.003毫秒 |
| 完整检测流程速度 | 8.772 FPS |
| 加载和预热 | 2.790秒 |
| 含加载预热的检测总时间 | 51.270秒 |
| 含加载预热的平均速度 | 8.290 FPS |
| 每源帧六管推理平均耗时 | 76.751毫秒 |
| 仅六管推理速度 | 13.029源帧/秒 |
| PyTorch已分配显存峰值 | 50.50MiB |
| PyTorch保留显存峰值 | 66.00MiB |

完整流程包含解码、六个ROI推理、CSV写入、画框、H.264编码及编码收尾。统计绘图与人工检查时间不计入检测速度。显存数字来自预热后重置的PyTorch allocator统计，包含当时存活的模型分配，但不包含CUDA上下文、驱动、外部分配和其他桌面应用；不是GPU整体显存峰值。

输出播放速度仍为原始约30FPS；8.77FPS是处理速度，因此本实现尚未达到原视频的实时速度。

## 是否可以开始第一个Tracker baseline

可用于离线、教学性质的首个Tracker baseline及失败案例分析；尚不足以直接用于可靠的个体身份或运动量化。

理由：已经能够逐帧提供有时间、试管和坐标信息的检测结果，大部分可见目标能检出；但重复框会造成虚假新轨迹，连续漏检和顶部遮挡会导致轨迹中断。后续baseline应以软件流程和失败案例分析为目标，不能直接承诺可靠身份恢复。当前模型又仅由同一条视频的少量帧微调，本结果不代表跨视频泛化能力。

**没有启动Tracker，没有分配Fly01/Fly02，没有做Kalman Filter、身份恢复、轨迹线、距离或运动时间。**

## 文件与运行

- detect_full_video.py：原始视频检测、CSV、H.264输出。
- analyze_detection_stability.py：数量统计、曲线、异常抽帧和输出视频解码校验。
- test_full_video_detection.py：坐标偏移、逐帧序号、连续区间及实际CSV一致性测试。
- requirements-train.txt / requirements-train.lock.txt：新增imageio-ffmpeg编码依赖，原OpenCV环境不变。

```powershell
.\.venv-train\Scripts\python.exe detect_full_video.py '20260817_202618.mp4' --output 'outputs/full_video_detection_new'
.\.venv-train\Scripts\python.exe analyze_detection_stability.py --input 'outputs/full_video_detection_new'
```

CSV中的detection_index从1开始，每帧每管重新编号，只是检测列表顺序。time_sec=frame_idx/元数据FPS；不是精确媒体PTS。局部坐标以ROI左上角为原点，全局中心加上ROI偏移。

3项测试通过，核对了全部23,646条检测的坐标与数量、输出425帧视频、30张截图及7张曲线。模型、ROI和原始人工图片/标签均通过不变性检查。

[结构化统计报告](<F:/fly video/outputs/full_video_detection/stability_report.json>)；[性能报告](<F:/fly video/outputs/full_video_detection/performance.json>)；[输出验证记录](<F:/fly video/outputs/full_video_detection/verification.json>)
