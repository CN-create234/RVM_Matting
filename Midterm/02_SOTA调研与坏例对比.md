# 人体视频抠像 SOTA 调研、复现与坏例对比

## 1. 最终结论

本次选择 **MatAnyone 2（CVPR 2026）** 对现有 RVM 坏例进行实测。测试结果表明：

- 在白色衣物、浅色背景和高曝光组合的 `Test1228_1_lab` 第 0166 帧上，MatAnyone 2 明显减少了 RVM 在肩部、右臂、袖口和手部周围的大面积半透明光晕；
- 人物主体没有明显缩小，手部轮廓比 RVM 更干净；
- 代价是 Alpha 边缘更硬，半透明细节明显减少，部分头发和衣物软边界可能被过度收紧；
- 8 帧短序列上的无配准时序变化代理指标与 RVM 基本持平，没有证据证明时序稳定性显著提升；
- MatAnyone 2 需要首帧人物 Mask、1080p 实测约 5 FPS，并且官方许可证只允许非商业用途，不能直接作为当前实时商业链路的无条件替代品。

**Baseline 决策：坏例的目标问题有明确改善，因此按照“有改善则作为后续改进基础”的规则，将 MatAnyone 2 设为后续的画质 baseline；RVM 保留为实时、无提示部署 baseline 和速度对照组。**

这不是两个互相冲突的结论：后续算法优化以 MatAnyone 2 的 Alpha 质量为目标，实时工程验收仍需同时达到 RVM 的无需提示、速度和许可证条件。

## 2. 任务范围

当前项目需要的不是普通人体语义分割，而是能够输出连续 Alpha 的**人体视频抠像**。候选模型需要重点满足：

1. 对头发、皮肤、手指、运动模糊和半透明边缘有较好处理；
2. 帧间稳定，不能出现明显闪烁；
3. 最好无需绿幕、Trimap 或逐帧人工提示；
4. 能处理 1080p/4K，并有实时或接近实时的部署可能；
5. 有公开代码和权重，能够在现有 RTX A4500 20 GB 上复现；
6. 如果用于实际制作，还必须考虑商业许可证。

## 3. 2025—2026 代表性工作

| 方法 | 时间 | 输入条件 | 主要优势 | 当前可复现性 | 对本项目的判断 |
|---|---:|---|---|---|---|
| RVM | WACV 2022 | 仅 RGB 视频 | 无提示、速度快、输出 `fgr + alpha` | 已部署 | 继续作为实时部署对照 |
| MatAnyone | CVPR 2025 | 视频 + 首帧 Mask | 记忆传播稳定、目标可指定 | 代码和权重公开 | 质量优于老式无提示方案，但需要初始化 |
| MatAnyone 2 | CVPR 2026 | 视频 + 首帧 Mask | VMReal 大规模真实数据、MQE 质量评估、复杂光照更稳 | 代码和权重公开 | 本次可运行的质量 SOTA 候选 |
| µMatting | CVPR 2026 | 人体视频 | 4K 两阶段处理、稀疏 3D 细节细化 | 官方仓库要求自行训练 CAP/FAR，没有可直接运行的完整预训练流程 | 暂不适合本轮快速替换 |
| VideoMaMa | CVPR 2026 | 视频 + 粗 Mask | 利用视频生成先验将粗 Mask 细化为 Matte | 代码和权重公开，但依赖 Stable Video Diffusion，计算重 | 更适合离线精修，不适合现有实时链路 |
| MatAnyone-E | 2026-10 预印本 | 视频 + 首帧 Mask | MatAnyone 2 的稀疏高分辨率加速版，论文报告质量近似但显著提速 | 官方仓库的推理代码和权重仍为 TODO | 当前最新、最值得跟踪，但本轮无法真实复现 |

### 3.1 为什么没有直接测试 MatAnyone-E

截至 2026-11-10，MatAnyone-E 是时间上最新且最贴合本项目方向的工作。论文通过稀疏记忆传播和轻量解码器减少高分辨率推理开销，并报告了接近 MatAnyone 2 的质量。但其官方仓库尚未发布推理代码和 checkpoint，因此无法做可信的本地复现。仅按论文重新实现不适合作为本轮“直接替换”测试。

### 3.2 为什么选择 MatAnyone 2

MatAnyone 2 的官方代码、官方权重和推理脚本均已发布；论文报告其在合成和真实视频基准上达到 SOTA，并针对背光、复杂背景、细节丢失和分割式硬边界做了改进。它也是 MatAnyone-E 的直接质量基础，因此在 MatAnyone-E 尚未开放时，是当前最合理的可运行替代候选。

## 4. 实验设置

### 4.1 硬件与环境

- GPU：NVIDIA RTX A4500，20 GB；
- 驱动：553.35，CUDA 12.4；
- 隔离环境：Python 3.10、PyTorch 2.6.0+cu124、TorchVision 0.21.0+cu124；
- 模型：MatAnyone 2 官方 `matanyone2.pth`；
- 输入分辨率：1920×1080，未降采样；
- 官方默认 warm-up：10 次；
- 官方默认形态学参数：erosion=10、dilation=10。

源码、权重和隔离环境均放在 `Midterm` 目录中，没有修改原 RVM 环境。

### 4.2 测试样例

使用 `Results/Test1228_1_lab` 的 0160～0167 连续 8 帧，重点观察第 0166 帧。该样例包含：

- 大面积白色实验室背景；
- 高曝光皮肤和帽子；
- 高反射银白色衣物；
- 人物与背景亮度、颜色接近；
- RVM 在右臂、肩部和手部周围产生明显光晕和背景泄漏。

### 4.3 首帧 Mask

MatAnyone 2 不是 RVM 的同输入替代品，它必须获得首帧目标 Mask。本次将 RVM 在 0160 帧的 Alpha 以 127 为阈值转换为二值 Mask，用作 MatAnyone 2 的初始化提示。

因此，本实验回答的是：

> 在使用 RVM 作为自动初始化器后，MatAnyone 2 能否把后续 Alpha 做得更好？

它不能证明 MatAnyone 2 在完全无提示条件下优于 RVM，因为 MatAnyone 2 本身不提供无提示人体检测。

## 5. 测试结果

### 5.1 视觉结果

对比图：`SOTA_Test_white_sequence_0160_0167/comparison_subject_crop.png`。

观察结果：

1. **肩部和右臂光晕明显减少**：RVM 的 Alpha 在右侧形成宽而模糊的灰白区域；MatAnyone 2 将其收紧到人物边界附近。
2. **手部轮廓更明确**：RVM 的双手周围存在较宽的半透明拖尾；MatAnyone 2 的手形更清楚。
3. **主体没有明显漏掉**：MatAnyone 2 的 `alpha ≥ 0.5` 前景面积略大于 RVM，并非简单腐蚀掉问题区域。
4. **边缘偏硬**：MatAnyone 2 的头发、袖口和衣物轮廓更接近二值分割，柔和过渡少于 RVM。
5. **皮肤内部不受修复**：脸部过曝仍然存在。两种模型主要改变 Alpha，不能恢复输入中已经丢失的肤色细节。

### 5.2 描述性统计

本样例没有 Ground Truth Alpha，因此以下数字不是准确率，只用于描述两个输出的形态。

| 指标（0166 帧） | RVM | MatAnyone 2 | 解释 |
|---|---:|---:|---|
| `alpha ≥ 128` 前景像素 | 136,623 | 141,390 | MatAnyone 2 没有通过整体缩小人物来去除光晕 |
| 1～254 的非整数 Alpha 像素 | 29,895 | 9,271 | MatAnyone 2 的软边界显著减少 |
| 16～239 的软边缘像素 | 25,440 | 6,246 | 光晕更少，但细发丝/半透明细节也可能减少 |
| 两结果平均绝对 Alpha 差 | 1.109/255 | — | 大部分主体和背景一致，差异集中在边界 |

8 帧相邻帧平均绝对 Alpha 变化（未做光流配准）：

- RVM：1.159/255；
- MatAnyone 2：1.185/255。

两者非常接近。本短序列不能支持“MatAnyone 2 时序更稳定”的结论，也没有显示灾难性闪烁。

### 5.3 速度观察

在 RTX A4500、1080p 原尺寸上，官方脚本 warm-up 后的进度约为 **5 FPS**。这只是 Python 官方脚本的观察值，不是严格 benchmark，但已经表明它不能直接满足当前 30/50 FPS 实时链路。

MatAnyone 2 官方脚本还会执行 10 次首帧 warm-up，模型加载和初始化会增加启动延迟。相比之下，现有 RVM MobileNetV3 FP16 的设计目标就是实时推理。

## 6. 是否直接替换 RVM

### 6.1 画质层面：可以升级为 baseline

本次目标坏例的主要症状是白色场景中的宽光晕和背景泄漏。MatAnyone 2 对该症状有明确改善，因此后续画质优化应以 MatAnyone 2 输出作为 baseline，而不是继续只围绕 RVM 的宽光晕结果微调。

### 6.2 工程层面：暂不能直接替换

仍有四个阻碍：

1. **需要首帧 Mask**：直播或自动流水线需要额外的人体检测/分割初始化模块；
2. **速度不足**：当前 A4500 上约 5 FPS，距离 30/50 FPS 有明显差距；
3. **输出接口不同**：RVM 直接预测去背景污染的 `fgr`，MatAnyone 2 官方输出的所谓 `fgr` 实际是把原图按 Alpha 合成到绿色背景，并非独立的 clean foreground；
4. **许可证限制**：MatAnyone 2 使用 NTU S-Lab License 1.0，默认仅允许非商业用途，商业使用需要联系作者授权。

因此建议采用双 baseline：

- **质量 baseline：MatAnyone 2**；
- **实时部署 baseline：RVM MobileNetV3 FP16**。

## 7. 下一步改进路线

### 7.1 短期：RVM + MatAnyone 2 两阶段离线链路

1. 用 RVM 自动生成首帧人体 Mask；
2. 对 Mask 做连通域筛选和孔洞修复；
3. 用 MatAnyone 2 传播并细化整个镜头；
4. 在镜头切换时重新初始化；
5. 使用原始 RGB + MatAnyone 2 Alpha 合成，或者增加独立的前景去污染模块。

这条路线可立即用于离线高质量结果，并避免人工绘制首帧 Mask。

### 7.2 中期：建立可量化测试集

当前坏例没有真实 Alpha，视觉判断无法区分“去除光晕”和“误删半透明细节”。建议建立至少包含以下内容的测试集：

- 白衣/白背景；
- 高曝光皮肤；
- 发丝和透明纱；
- 快速手部动作；
- 镜头硬切；
- 多人交叉遮挡。

有 Ground Truth 时计算 MAD、MSE、Gradient、Connectivity 和 dtSSD；没有 Ground Truth 时至少进行人工 A/B 盲评，并记录端到端 FPS、显存和启动延迟。

### 7.3 中长期：跟踪 MatAnyone-E

MatAnyone-E 与当前方向最匹配：保持 MatAnyone 2 质量，同时显著降低高分辨率记忆传播成本。其官方代码和权重发布后，应优先在相同 0160～0167 序列上复测。如果 A4500 上能接近实时，它比直接优化 MatAnyone 2 更有潜力成为真正替换 RVM 的统一 baseline。

## 8. 复现实验文件

- `SOTA_Test_white_sequence_0160_0167/comparison_subject_crop.png`：主体局部对比；
- `SOTA_Test_white_sequence_0160_0167/comparison_full.png`：完整画面对比；
- `SOTA_Test_white_sequence_0160_0167/comparison_metrics.json`：描述性统计；
- `SOTA_Test_white_sequence_0160_0167/matanyone2_output/`：8 帧 Alpha、预览帧和视频；
- `make_sota_comparison.py`：生成对比图与统计的脚本；
- `run_matanyone2_test.ps1`：使用 `Midterm` 内环境和缓存一键复现实验；
- `MatAnyone2/`：官方源码及模型权重；
- `.venv-matanyone2/`：隔离运行环境。
- `torch_cache/`：实验所需的 ResNet 辅助权重缓存。

## 9. 参考资料

- [MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_MatAnyone_2_Scaling_Video_Matting_via_a_Learned_Quality_Evaluator_CVPR_2026_paper.pdf)
- [MatAnyone 2 官方代码](https://github.com/pq-yang/MatAnyone2)
- [MatAnyone: Stable Video Matting with Consistent Memory Propagation（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_MatAnyone_Stable_Video_Matting_with_Consistent_Memory_Propagation_CVPR_2025_paper.html)
- [αMatte4K & µMatting（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_alphaMatte4K__muMatting_Dataset_and_Model_for_Ultra-Micro_Precision_Alpha_CVPR_2026_paper.html)
- [µMatting 官方代码](https://github.com/kadatec/mu-Matting)
- [VideoMaMa 官方代码](https://github.com/cvlab-kaist/VideoMaMa)
- [MatAnyone-E 论文](https://arxiv.org/abs/2610.35724)
- [MatAnyone-E 官方仓库](https://github.com/Shuyun-Wang/MatAnyone-E)
