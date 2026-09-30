# 首次 YOLO 果蝇检测教学实验报告

本次完成标签 QA、按时间点复制划分、本机 GPU 微调、best.pt 验证、六图预测及单图推理。只检测可见果蝇，完全遮挡不猜框；没有跟踪或个体 ID。

## 数据与质检

30 张图片全部有同名人工标签，共 **278 个标注实例**。这些是跨帧框的总数，并非独立果蝇数量。类别均为 0，坐标和正宽高合法，未发现越界框或 IoU ≥ 0.85 的疑似重复人工框。全部原图和标签哈希保持不变。

已查看五张 QA 联系表，未发现明确需要阻断训练的错位或重复框；低清晰度、重叠和局部遮挡仍可能影响人工判断。

| 划分 | 帧号 | 图片 | 实例 |
|---|---|---:|---:|
| Train | 0、80、160、240 | 24 | 229 |
| Validation | 320 | 6 | 49 |

[逐图人工数量 CSV](<F:/fly video/outputs/training_qa/image_counts.csv>)；[QA 联系表目录](<F:/fly video/outputs/training_qa/contact_sheets>)；[QA 报告](<F:/fly video/outputs/training_qa/qa_report.json>)

## 环境

| 项目 | 实际结果 |
|---|---|
| Python | 3.11.1，独立 .venv-train |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU，8 GB |
| NVIDIA 驱动 | 616.56，nvidia-smi 正常 |
| torch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| torch.cuda.is_available() | True |
| torch.version.cuda | 12.8 |
| Ultralytics | 8.4.143 |
| GPU 运算检查 | CUDA 矩阵乘法通过 |

原 `.venv` 保持独立，继续用于 OpenCV 读取和人工标注。[环境记录](<F:/fly video/outputs/training_environment.json>)；[完整依赖锁定](<F:/fly video/requirements-train.lock.txt>)

## 实际训练

选择 **YOLO26n Detection 预训练权重 yolo26n.pt**，nano 规模适合 8GB 显存的小样本教学实验。权重从 Ultralytics 官方发布地址下载，迁移 606/708 项预训练参数，不是从随机模型开始。

| 参数 | 值 |
|---|---|
| epochs | 上限60，实际51；第31轮最佳 |
| batch / nbs | 4 / 16 |
| imgsz / rect | 640 / True，按长宽比填充 |
| device / workers | 0 / 0 |
| optimizer | AdamW，lr0=0.001，lrf=0.01，weight_decay=0.0005 |
| patience | 20，触发早停 |
| AMP / cache | True / False |
| seed / deterministic | 42 / True |
| 平移 / 尺度 / 水平翻转 / 亮度 | 0.02 / 0.1 / 0.5 / hsv_v=0.1 |
| 强增强 | mosaic、mixup、copy-paste、旋转、透视、上下翻转均为0 |
| PyTorch 峰值显存 | 1.281 GiB（不含其他桌面应用） |

训练CSV记录约 77.6 秒；该时间不含首次环境下载安装。

[实际全部训练参数 args.yaml](<F:/fly video/runs/detect/fly_yolo26n_teaching/args.yaml>)；[results.csv](<F:/fly video/runs/detect/fly_yolo26n_teaching/results.csv>)；[best.pt](<F:/fly video/runs/detect/fly_yolo26n_teaching/weights/best.pt>)；[last.pt](<F:/fly video/runs/detect/fly_yolo26n_teaching/weights/last.pt>)

## 损失变化

| 损失 | 第1轮 | 第51轮 | 最低值 |
|---|---:|---:|---:|
| train/box_loss | 2.75682 | 1.78154 | 1.76950 |
| train/cls_loss | 3.93892 | 1.12084 | 1.07247 |
| train/l1_loss | 0.01046 | 0.00631 | 0.00616 |
| val/box_loss | 2.90816 | 2.54189 | 2.01400 |
| val/cls_loss | 4.19653 | 1.77200 | 1.56498 |
| val/l1_loss | 0.01273 | 0.01066 | 0.00804 |

训练损失总体下降；验证 box/L1 损失先降后升，后期定位表现没有持续改善。这是早停后采用 best.pt、而不是只取末轮的重要原因。YOLO26 此次记录的是 L1 loss，不是 DFL loss。

![训练曲线](<F:/fly video/runs/detect/fly_yolo26n_teaching/results.png>)

## best.pt 最终指标

以下主结果来自单独运行 evaluate_flies.py：batch=4、imgsz=640、conf=0.001、iou=0.7。Precision/Recall 为 Ultralytics 根据 PR/F1 曲线报告的点，mAP 使用置信度排序，不等于下方 conf=0.25 数量统计。

| 指标 | 独立验证 | 训练结束自动验证（另存） |
|---|---:|---:|
| Precision | 0.736250（73.62%） | 0.773036 |
| Recall | 0.740677（74.07%） | 0.764713 |
| mAP50 | 0.701954（70.20%） | 0.736547 |
| mAP50-95 | 0.212657（21.27%） | 0.225244 |

训练结束自动验证复用了训练器的验证加载配置（日志为1批）；独立验证为batch=4共2批。分别保留两次结果，不混用。

## 六张验证图：人工 vs 预测

固定预测 confidence ≥ 0.25，IoU 阈值0.7。数量相同不代表逐个框都匹配正确。

| 图片 | 人工 | 预测 | 差值 |
|---|---:|---:|---:|
| [frame_000320_tube_1.png](<F:/fly video/outputs/validation_predictions/frame_000320_tube_1_prediction.png>) | 7 | 7 | +0 |
| [frame_000320_tube_2.png](<F:/fly video/outputs/validation_predictions/frame_000320_tube_2_prediction.png>) | 7 | 7 | +0 |
| [frame_000320_tube_3.png](<F:/fly video/outputs/validation_predictions/frame_000320_tube_3_prediction.png>) | 9 | 12 | +3 |
| [frame_000320_tube_4.png](<F:/fly video/outputs/validation_predictions/frame_000320_tube_4_prediction.png>) | 8 | 7 | -1 |
| [frame_000320_tube_5.png](<F:/fly video/outputs/validation_predictions/frame_000320_tube_5_prediction.png>) | 9 | 9 | +0 |
| [frame_000320_tube_6.png](<F:/fly video/outputs/validation_predictions/frame_000320_tube_6_prediction.png>) | 9 | 9 | +0 |

人工共49个实例，预测共51个。第3管顶部可见重复预测，第4管数量偏少，保留结果供人工检查，没有据此改动标签。

![六张验证预测](<F:/fly video/outputs/validation_predictions/validation_contact_sheet.png>)

## 单图使用

在项目目录下运行：

```powershell
.\.venv-train\Scripts\python.exe predict_flies.py 'dataset_yolo/images/val/frame_000320_tube_1.png' --output-dir outputs/my_prediction
```

实际已运行单图CLI，返回7个fly框，每个框均输出confidence和像素xyxy，并保存图片及JSON。修改图片路径即可使用；需避免覆盖已有结果目录。

## 实验边界

**本次训练集和验证集来自同一条短视频，因此这些指标不能代表跨视频泛化能力。** 本次是教学与流程验证，不是最终性能结论。原始 dataset_sample 人工数据没有被删除、移动或修改。

已停止训练实验；未实现身份编号、跟踪、遮挡身份恢复、轨迹、距离、运动时间或 GUI。

模型与参数依据：[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)；[官方训练文档](https://docs.ultralytics.com/modes/train/)；[PyTorch 安装组合](https://pytorch.org/get-started/previous-versions/)。
