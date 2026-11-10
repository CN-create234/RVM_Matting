# Midterm 内容清单

更新时间：2026 年 11 月 10 日

## 一 建议正式交付

- [x] `01_RVM算法代码分析.md`
  - 内容：RVM 原理、当前代码数据流、适用场景、主要问题和改进建议。
- [x] `02_SOTA调研与坏例对比.md`
  - 内容：2025 至 2026 人体视频抠像方法调研、MatAnyone 2 复现、RVM 坏例对比和后续路线。
- [x] `04_人体先验与皮肤次表面散射改进调研.md`
  - 内容：人体结构先验、SMPL 和 SMPL-X、皮肤次表面散射、实验分组和工程风险。
- [x] `05_2D姿态先验V1实验记录.md`
  - 内容：Sapiens2 姿态融合实现、0160 至 0167 首轮实验、修改面积和局限性。
- [x] `06_姿态先验GroundTruth评估交付说明.md`
  - 内容：UE Actor Mask 导出规范、GT 指标、运行命令和验收标准。
- [x] `RVM深度信息融合调研与实验报告.docx`
  - 内容：RGB-D 方法调研、固定背景深度融合实现、受控合成实验、局限性和真实 RGB-D 实验方案。
  - 注意：当前深度实验为合成验证，不是现场深度相机结果。
- [x] `SOTA_Test_white_sequence_0160_0167/comparison_full.png`
  - 内容：8 帧完整画面对比。
- [x] `SOTA_Test_white_sequence_0160_0167/comparison_subject_crop.png`
  - 内容：主体局部对比。
- [x] `SOTA_Test_white_sequence_0160_0167/comparison_metrics.json`
  - 内容：RVM 与 MatAnyone 2 的描述性统计；无真实 Alpha，不能视为精度指标。
- [x] `DepthFusionSyntheticExperiment/comparison.png`
  - 内容：目标 Alpha、模拟 RGB Matte、深度证据和融合结果对比。
- [x] `DepthFusionSyntheticExperiment/metrics.json`
  - 结果：MAE 从 0.075208 降至 0.019262，相对下降 74.4%；仅代表受控合成验证。

## 二 建议随项目代码交付

- [x] `make_sota_comparison.py`
  - 用途：生成 RVM 与 MatAnyone 2 对比图及统计数据。
- [x] `run_matanyone2_test.ps1`
  - 用途：复现 MatAnyone 2 白色场景实验。
- [x] `run_depth_fusion_synthetic_experiment.py`
  - 用途：复现深度融合合成实验和指标。
- [x] `build_depth_experiment_report.py`
  - 用途：重新生成 Word 深度实验报告。
- [x] `docx_preview_html.py`
  - 用途：Word 报告在缺少 LibreOffice 时的排版预览辅助工具。
- [x] `../evaluate_pose_prior.py`
  - 用途：生成 RVM 与姿态融合版本的整帧/手部 ROI Ground Truth 指标、CSV 和对比图。

## 三 实验数据与结果目录

- [x] `SOTA_Test_white_sequence_0160_0167/`
  - 31 个文件，约 28.96 MB。
  - 包含 0160 至 0167 输入帧、首帧 Mask、RVM Alpha、MatAnyone 2 Alpha 和前景结果、对比图及统计。
  - 建议作为主要 SOTA 对比实验交付。
- [x] `SOTA_Test_white_1_0166/`
  - 12 个文件，约 5.37 MB。
  - 单帧 0166 的早期测试，包括默认形态学参数和调整参数结果。
  - 与 8 帧序列实验部分重复，可作为补充材料。
- [x] `DepthFusionSyntheticExperiment/`
  - 2 个文件，约 0.14 MB。
  - 包含深度融合对比图和指标。
  - 尚缺真实 RGB-D 实验数据与结果。

## 四 复现依赖 不建议作为普通附件发送

- [ ] `MatAnyone2/`
  - 161 个文件，约 168 MB。
  - MatAnyone 2 官方源码、配置和模型相关文件。
  - 建议通过 Git 子模块、仓库链接或单独压缩包提供。
- [ ] `.venv-matanyone2/`
  - 约 26418 个文件，约 5016.50 MB。
  - 本地 Python 虚拟环境。
  - 不建议提交 Git 或直接交付；应改为提供依赖清单和环境创建说明。
- [ ] `torch_cache/`
  - 2 个文件，约 142.41 MB。
  - PyTorch 下载缓存或辅助权重。
  - 不建议提交 Git；可通过下载脚本或模型链接恢复。

## 五 临时文件与可清理内容

- [ ] `_report_render/`
  - 约 1255 个文件，约 38.39 MB。
  - Word 报告渲染页面、PDF 预览、可访问性检查和 Edge 临时配置。
  - 最终报告确认后可以删除，不属于正式交付物。
- [ ] `__pycache__/`
  - Python 字节码缓存，可以删除。
- [ ] `SOTA_Test_white_1_0166/`
  - 若只保留完整的 0160 至 0167 序列实验，可以归档或删除该单帧早期实验。

## 六 当前结果摘要

- [x] 完成 RVM 原理与当前工程分析。
- [x] 完成视频抠像 SOTA 调研和 MatAnyone 2 对比。
- [x] 完成人体先验与皮肤次表面散射方向调研。
- [x] 完成深度信息后融合代码和受控合成验证。
- [x] 完成深度融合 Word 实验报告。
- [x] 完成 Sapiens2 2D 姿态先验融合代码和 8 帧首轮验证。
- [x] 完成 UE Ground Truth 评估器、指标定义、测试和交付规范。
- [ ] 尚未取得与 0160 至 0167 对齐的 UE Actor/Object Mask，因此尚无可报告的真实 GT 精度数字。
- [ ] 尚未完成真实 RGB-D 相机实验。
- [ ] 尚未建立带真实 Alpha 标注的统一评测集。
- [ ] 尚未验证深度相机与制作相机的标定、同步和配准误差。
- [ ] 尚未验证 LED 大屏对 ToF 或主动红外深度相机的干扰。

## 七 整理建议

- [ ] 将深度报告补为编号 `03`，解决当前 `02` 直接跳到 `04` 的编号缺口。
- [ ] 新增 `README.md`，写明环境搭建、模型下载、实验命令和目录结构。
- [ ] 从 Git 中排除 `.venv-matanyone2`、`torch_cache`、`_report_render`、模型权重和生成视频。
- [ ] 正式交付时分为三个包：报告、代码、精选实验结果。
- [ ] 真实深度数据到位后，补充原始 RVM、深度单独阈值、RVM 加深度融合三组对照。

## 八 推荐交付结构

```text
Midterm_Delivery/
├─ Reports/
│  ├─ 01_RVM算法代码分析.md
│  ├─ 02_SOTA调研与坏例对比.md
│  ├─ 03_RVM深度信息融合调研与实验报告.docx
│  ├─ 04_人体先验与皮肤次表面散射改进调研.md
│  ├─ 05_2D姿态先验V1实验记录.md
│  └─ 06_姿态先验GroundTruth评估交付说明.md
├─ Code/
│  ├─ make_sota_comparison.py
│  ├─ run_matanyone2_test.ps1
│  ├─ run_depth_fusion_synthetic_experiment.py
│  ├─ build_depth_experiment_report.py
│  └─ evaluate_pose_prior.py
└─ Results/
   ├─ SOTA_Comparison/
   │  ├─ comparison_full.png
   │  ├─ comparison_subject_crop.png
   │  └─ comparison_metrics.json
   └─ DepthFusionSynthetic/
      ├─ comparison.png
      └─ metrics.json
```
