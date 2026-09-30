# 六管各一只：本地自动分析原型结果

已对两段新视频自动定位六根试管、背景深色终点线和管底参考；使用现有YOLO26n检测可见果蝇，每管只有固定身份 Tn-F001。视频第0帧为计时起点，标定距离5.5cm。

## 视频与校验

| 视频 | 实际处理帧 | 处理时间 | 速度 | 横线质量 |
|---|---:|---:|---:|---|
| 20260921_193819.mp4 | 499 | 99.7s | 5.01FPS | review |
| 20260921_193846.mp4 | 927 | 105.1s | 8.82FPS | ok |

输入元数据分别报告504、931帧；实际顺序可解码499、927帧，结果视频已验证能解码相同数量，保持1920×1080及原始名义约30FPS。新标定没有复用旧视频ROI。

## 六管首次过线候选

**速率列是候选平均爬升指标**：5.5cm除以从第0帧到首次确认过线的秒数。它不等于沿曲线累计的路程速度或瞬时速度。标注“review”的值不能直接当正式实验结果。

| 视频 | Tube/固定ID | 首次过线帧 | 用时/s | 候选cm/s | 质量 |
|---|---|---:|---:|---:|---|
| 20260921_193819 | T1-F001 | 116 | 3.867 | 1.422 | review（finish_line_low_contrast_review） |
| 20260921_193819 | T2-F001 | 139 | 4.634 | 1.187 | review（finish_line_low_contrast_review） |
| 20260921_193819 | T3-F001 | 131 | 4.367 | 1.259 | review（finish_line_low_contrast_review） |
| 20260921_193819 | T4-F001 | 457 | 15.234 | 0.361 | review（finish_line_low_contrast_review） |
| 20260921_193819 | T5-F001 | 78 | 2.600 | 2.115 | review（finish_line_low_contrast_review；initial_position_above_bottom） |
| 20260921_193819 | T6-F001 | 301 | 10.034 | 0.548 | review（finish_line_low_contrast_review） |
| 20260921_193846 | T1-F001 | 147 | 4.900 | 1.122 | auto_candidate（无自动风险标记） |
| 20260921_193846 | T2-F001 | 257 | 8.567 | 0.642 | auto_candidate（无自动风险标记） |
| 20260921_193846 | T3-F001 | 64 | 2.133 | 2.578 | review（initial_position_above_bottom） |
| 20260921_193846 | T4-F001 | 275 | 9.167 | 0.600 | review（initial_position_above_bottom） |
| 20260921_193846 | T5-F001 | 122 | 4.067 | 1.352 | auto_candidate（无自动风险标记） |
| 20260921_193846 | T6-F001 | 128 | 4.267 | 1.289 | auto_candidate（无自动风险标记） |

短片标线对比度较低（自动拟合9个有效横向支撑点，拟合RMSE约1.62px），全部六管记为review。tube_5首帧距离估算管底较高，完整5.5cm/视频起点的解释尤其需谨慎。
长片标线更清楚（23个支撑点，RMSE约1.73px）；tube_1、2、5、6未触发自动风险标记，tube_3、4首帧位置偏高，记为review。即使auto_candidate也需要抽查人工视频和标定图。

## 观看与复核


**20260921_193819**：

- [带固定ID、过线时间和候选速率的视频（修复版）](<F:/fly video/outputs/single_fly_runs/20260921_193819/annotated_single_fly.mp4>)
- [末帧六管速率截图](<F:/fly video/outputs/single_fly_runs/20260921_193819/speed_overview.png>)
- [六管首次过线图](<F:/fly video/outputs/single_fly_analysis/20260921_193819/review/crossing_contact_sheet.png>)
- [自动标定图](<F:/fly video/outputs/single_fly_analysis/20260921_193819/automatic_geometry_preview.png>)
- [每管结果CSV](<F:/fly video/outputs/single_fly_analysis/20260921_193819/first_passage_results.csv>)
- [逐帧选中目标与缺口CSV](<F:/fly video/outputs/single_fly_analysis/20260921_193819/single_fly_observations.csv>)


**20260921_193846**：

- [带固定ID、过线时间和候选速率的视频（修复版）](<F:/fly video/outputs/single_fly_runs/20260921_193846/annotated_single_fly.mp4>)
- [末帧六管速率截图](<F:/fly video/outputs/single_fly_runs/20260921_193846/speed_overview.png>)
- [六管首次过线图](<F:/fly video/outputs/single_fly_analysis/20260921_193846/review/crossing_contact_sheet.png>)
- [自动标定图](<F:/fly video/outputs/single_fly_analysis/20260921_193846/automatic_geometry_preview.png>)
- [每管结果CSV](<F:/fly video/outputs/single_fly_analysis/20260921_193846/first_passage_results.csv>)
- [逐帧选中目标与缺口CSV](<F:/fly video/outputs/single_fly_analysis/20260921_193846/single_fly_observations.csv>)


过线事件要从线下到线上，连续3帧有真实检测确认；事件时间记首次越过约3像素边界的那一帧。完成后停止对该管继续选择目标。轨迹缺帧不补点，固定身份不会反复新建。标线是背景深色横线，图中红色；蓝色管底仅供初始位置QA，管身白色环线不是终点。

## 范围与下一步

当前程序是本地命令行原型。它只对六根直立试管、上部白塞、浅背景深横线这类固定画面进行规则自动标定。输入布局或线对比度不满足条件会停下报错，不会默默按错误位置计算。两段视频通过不能证明对任意新视频都可靠。

当前无需实验室服务器；RTX 3070 Ti已完成两段全程推理。若将来要求实时或大批量处理，再比较本机和服务器。做交互式应用前，应先让人工核对更多独立视频的ROI、终点线、首次过线帧及首帧位置。

`run_single_fly.cmd`已修复并用两段MP4从命令行实际运行：拖入视频后，每次选择新的 `outputs/single_fly_runs/视频名[_runN]/`，不会因同名旧结果而失败或覆盖。
修复版画面显示首次过线时间和 `v*` 候选cm/s；`CHECK`表示需要复核，`candidate`表示未触发自动风险标记。
