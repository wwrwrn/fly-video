# 第4阶段：ByteTrack baseline报告

六管独立ByteTrack已经运行并验证，但身份稳定性不满足直接进行逐只果蝇首次到线速度或运动距离计算的要求。保留全部失败轨迹，没有自动修正ID、没有调参搜索、没有重新训练。

## 实际运行与参数

实际处理425帧（0–424），生成22,447行确认轨迹观测。输出视频为1920×1080、约29.998271FPS，绘制ROI、Tn-Fxxx、confidence、frame/time及最近1.5秒短历史线。缺帧处不插值连线。

| 参数 | 值 |
|---|---|
| tracker_type | bytetrack |
| track_high_thresh | 0.25 |
| track_low_thresh | 0.1 |
| new_track_thresh | 0.25 |
| track_buffer | 30 |
| match_thresh | 0.8 |
| fuse_score | True |
| detector conf / imgsz / IoU | 0.10 / 640 / 0.7 |
| device | CUDA:0，RTX 3070 Ti Laptop GPU |
| 实现 | Ultralytics 8.4.143 BYTETracker，lap 0.5.13 |
| 完整运行时间 / FPS | 44.47秒 / 9.56源帧每秒 |

为了启用低分检测关联，将检测保留阈值从第3.5阶段的0.25降到0.10。新轨迹仍需达到0.25；低分框不会独立新建轨迹。track_buffer是30帧，当前实现不会按FPS另行缩放。match_thresh为关联代价阈值，并非IoU必须大于0.8。

ByteTrack使用高分和低分两阶段关联，减少低分框被直接丢弃造成的断裂；其运动状态模型和超时管理不能保证遮挡后一定恢复正确身份。[官方文档](https://docs.ultralytics.com/modes/track/)；[论文](https://arxiv.org/abs/2110.06864)。

## 各试管ID与生命周期

“全部创建”包括待确认即失败的候选，不能只保留成功轨迹。极短定义为生命周期≤5帧（约0.167秒），各管最短均为1帧。

| Tube | 全部创建 | 曾确认 | 从未确认 | 活跃均值（范围） | 最长/帧 | 极短≤5帧 |
|---|---:|---:|---:|---|---:|---:|
| tube_1 | 262 | 58 | 204 | 8.58（5–12） | 408 | 214 |
| tube_2 | 374 | 60 | 314 | 8.75（6–11） | 404 | 328 |
| tube_3 | 349 | 63 | 286 | 8.39（5–11） | 425 | 295 |
| tube_4 | 296 | 65 | 231 | 8.07（5–18） | 415 | 245 |
| tube_5 | 184 | 33 | 151 | 9.41（7–12） | 425 | 157 |
| tube_6 | 406 | 74 | 332 | 9.61（6–13） | 404 | 348 |

**共创建1,871个ID，353个曾确认输出，1,518个从未确认；1,587条生命周期≤5帧。** 即使只看曾确认轨迹，每管33–74个仍明显高于约8–12只果蝇的规模，说明身份碎片化和重复创建明显。长轨迹也不自动意味着其始终对应同一只果蝇。

最长轨迹：

- tube_1：T1-F046，408帧。
- tube_2：T2-F056，404帧。
- tube_3：T3-F001，425帧。
- tube_4：T4-F022，415帧。
- tube_5：T5-F004, T5-F005, T5-F010，425帧。
- tube_6：T6-F067，404帧。

summary按首次匹配检测到最后匹配检测定义duration，包含内部缺口，不包含尾部lost缓冲等待。number_of_detected_frames包含候选激活观测；number_of_output_frames仅计确认输出。missing_gaps记录缺失区间，censored_at_video_end区分录像结束时仍保留的轨迹。**track duration ≠ movement time。**

## A：0–1秒

frame0–29共创建434个ID，其中72个来自首帧，之后又创建362个；119个在前1秒内曾确认。对照已有首帧60个人工框，72个初始tracker输出已不符合“一只一个正确初始ID”。

明确例子：tube_3一个孤立可见目标在frame0–1同时具有T3-F005和T3-F008，之后后者lost。这是重复身份，不能解释为两只真实果蝇。底部密集区域也有大量短命候选及碎片。

[前1秒跟踪视频](<F:/fly video/outputs/bytetrack_baseline/review/first_second_tracking.mp4>)

![初始重复身份](<F:/fly video/outputs/bytetrack_baseline/review/panels/A_initial_duplicate_identity.png>)

## B：tube_4 frame319–322

目标ID为 **T4-F240**。四帧均保持这个ID：

| Frame | 检测confidence | 状态/关联 |
|---|---:|---|
| 319 | 0.317 | tracked，高分 |
| 320 | 0.235 | tracked，低分补救 |
| 321 | 0.205 | tracked，低分补救 |
| 322 | 0.105 | tracked，低分补救 |

第3.5阶段阈值0.25下看起来是漏检，但本次0.10输入下仍有检测。因此这里证明了低分补救保持ID，而不是凭空生成框，也不是完全丢失后的身份恢复。

该ID在frame323–352处于lost保留状态，frame353移除；截至424没有恢复T4-F240。后续相近位置出现其他ID，不能只凭位置断言是原果蝇回来。

![低分补救](<F:/fly video/outputs/bytetrack_baseline/review/panels/B_low_confidence_rescue.png>)

[lost保留及超时移除对照](<F:/fly video/outputs/bytetrack_baseline/review/panels/B_buffer_and_removal.png>)

## C：tube_3 frame319–322

已有 **T3-F229** 对应顶部右侧目标。frame320的重复检测创建T3-F278，frame321得到确认；321–322同时出现两个重叠ID。T3-F278不是仅存在于日志中的候选，而是实际显示过的第二条轨迹。

![重复检测形成额外确认ID](<F:/fly video/outputs/bytetrack_baseline/review/panels/C_duplicate_confirmed_track.png>)

## ID switch判断与验证限制

已经明确观察到重复ID、候选反复创建和轨迹碎片化。抽查不足以可靠确认两个真实个体互换ID；没有连续帧人工身份真值，因此严格IDSW次数记为未知，不记为零，也不把所有新ID计为ID switch。

六管、425帧均与原生BYTETracker输出复核一致，ID创建数量相同。高ID数不是本地命名空间或适配层造成。7项接入/边界测试通过。

状态审计曾发现当前原生实现的过期边界行为：历史removed对象可能下一帧又被重新关联。本次出现T5-F042（frame57）和T6-F205（frame177），缺口均为31帧。审计已改为以当前live状态为准，并用保存检测重建；没有改动原生关联算法。tracks.csv、detections.csv和跟踪视频哈希保持不变，旧审计备份在audit_before_fix，仅当前顶层审计文件用于结论。

## 对首次过终点线实验的影响

用户的新目标合理：每只果蝇从底部到第一次到达终点线，过线后不再纳入该个体。上部横向环线可作为待确认终点参考；六管终点位置、计时起点和实际距离尚未正式配置。

终点规则可以避开过线后顶部隐藏和再次出现的问题，但无法纠正起点ID重复或到线前断裂。**当前baseline还不足以直接计算可靠的逐只首次到达速度或运动距离。**

本机运行本次baseline约44.47秒，当前不需要实验室服务器。更需要起步0–1秒及到线前少量连续帧的人工身份核对；若现有重叠连人工也无法辨认，则应补拍更清晰起步样本。不是增加GPU算力就能恢复缺失的身份信息。

后续使用标定距离除以首次到达用时得到的是平均爬升速度，不是实际曲折路径长度除以时间。未到达或身份失效应单独记录，具体测量规则另行确定。本轮没有实现终点线判断、过线计时、距离、速度、运动/静止或GUI。

## 交付文件与运行

[annotated_tracking.mp4](<F:/fly video/outputs/bytetrack_baseline/annotated_tracking.mp4>)；[tracks.csv](<F:/fly video/outputs/bytetrack_baseline/tracks.csv>)；[track_summary.csv](<F:/fly video/outputs/bytetrack_baseline/track_summary.csv>)
 
[状态审计](<F:/fly video/outputs/bytetrack_baseline/track_states.csv>)；[事件日志](<F:/fly video/outputs/bytetrack_baseline/track_events.csv>)；[全部重点帧和对照图](<F:/fly video/outputs/bytetrack_baseline/review>)；[生命周期图](<F:/fly video/outputs/bytetrack_baseline/plots/track_lifecycles.png>)

新增程序：tube_bytetrack.py、track_video_baseline.py、analyze_tracking_baseline.py、test_bytetrack_baseline.py。新增依赖lap=0.5.13，仅安装在.venv-train。

```powershell
.\.venv-train\Scripts\python.exe track_video_baseline.py '20260817_202618.mp4' --output 'outputs/bytetrack_baseline_new'
.\.venv-train\Scripts\python.exe analyze_tracking_baseline.py --input 'outputs/bytetrack_baseline_new'
```

已有输出不会覆盖。单独的--refresh仅用于从现有跟踪结果刷新分析，不重新检测或改ID。

**本阶段已停止，未进入运动量化。**
