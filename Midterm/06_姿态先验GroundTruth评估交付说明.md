# 姿态先验 Ground Truth 评估交付说明

更新时间：2026 年 11 月 10 日

## 1. 当前结论

真实 Ground Truth 精度评估的代码已经完成，但当前 `Results` 中没有人物 Actor/Object Mask。已有 EXR 的第 4 通道是常量，不能作为人物 Alpha；背景差分也会重复当前算法在肤色、半透明边缘上的失败模式，因此不能冒充 Ground Truth。

当前 Unreal MCP 没有返回可用的场景 Actor，故本轮不能直接替项目渲染 Mask。正式精度数字必须等 UE 导出与原视频逐帧对齐的二值人物 Mask 后生成。

还需特别区分两类数据：UE Object ID 只能标记 UE 中真正存在的 MetaHuman/Skeletal Mesh Actor，不能从摄像机 Video Plate 中自动分割现实演员。因此，现有 0160–0167 真人坏例若只是摄像机画面，必须人工标注或使用独立标注工具；UE Ground Truth 应用于新建的合成 MetaHuman/数字人受控序列，并可顺便系统改变皮肤 Subsurface 参数。

## 2. 已交付内容

- `../evaluate_pose_prior.py`：GT 评估器。
- `../tests/test_pose_prior_evaluation.py`：指标正确性测试。
- `05_2D姿态先验V1实验记录.md`：无 GT 的首轮结果与限制。
- 本文档：UE 导出规范、运行命令和验收标准。

评估器会生成：

```text
PosePrior_GTEvaluation/
├─ metrics.json              # 聚合指标与相对变化
├─ per_frame_metrics.csv     # 每帧、每方法、每区域指标
└─ comparison.png            # GT / RVM / 姿态融合 / 手部ROI / 误差改善图
```

报告两个范围：整帧，以及根据 Sapiens2 手/腕/前臂关键点生成并向外扩张 12 像素的手部 ROI。指标包括 SAD、MAE、MSE、IoU、Precision、Recall、误检像素和漏检像素。

## 3. UE Ground Truth 导出要求

推荐用 Movie Render Queue 的 **Object IDs (Limited)**：

1. 启用 `Movie Render Queue` 和 `Movie Render Queue Additional Render Passes` 插件并重启编辑器。
2. 在 Movie Render Queue 添加 `Object Ids (Limited)`；按 Actor Name 或 Actor 分组，确保人物可单独选出。
3. 输出多层 EXR，并从 Cryptomatte 中提取人物 Actor；或转存为单通道 PNG。
4. Mask 中人物为 255、背景为 0；文件名与 RGB/RVM 帧一致，例如 `0166.png`。
5. 输出必须与用于 RVM 和 Sapiens2 的画面保持完全相同的相机、分辨率、帧率、起止帧和时序。
6. Object IDs 不应开启会混合对象 ID 的景深、运动模糊等多采样效果。若目标是评估硬轮廓，应保留无抗锯齿二值 Mask；若另有可信软 Alpha，可作为第二套 Matting GT 单独评估。

建议建立两套评测数据：

- **真人回归集**：人工标注当前 0160–0167，直接验证真实坏例；至少精标手、腕、前臂区域。
- **UE 合成集**：MetaHuman/数字人 Actor，导出 Object ID 真值；设置低、中、高三档皮肤 SSS，并覆盖浅色/深色背景、运动模糊和不同手势，用于可控消融。

官方依据：

- Epic 的 Render Pass 文档说明 Object IDs 会给场景 Actor 分配 ID，要求启用 Additional Render Passes 插件，并以 Cryptomatte 存入多层 EXR：<https://dev.epicgames.com/documentation/unreal-engine/cinematic-render-passes-in-unreal-engine>
- Custom Depth/Stencil 也能标记指定 Actor，但需要项目开启带 Stencil 的 Custom Depth，并给人物 Primitive 设置值：<https://dev.epicgames.com/documentation/unreal-engine/post-process-materials-in-unreal-engine>

## 4. 目录约定

```text
Midterm/PosePrior_GT_Input/
├─ 0160.png
├─ 0161.png
└─ ...
```

GT 图片可以命名为 `0160.png`、`0160_mask.png` 或 `0160_gt.png`。评估器会统一按帧号配对。

## 5. 运行命令

```powershell
external\.venv-sapiens2-py310\Scripts\python.exe evaluate_pose_prior.py `
  --baseline-dir Results\Test1228_1_lab\RVMResults `
  --fused-dir Midterm\PoseFusion_Sapiens2_0160_0167 `
  --ground-truth-dir Midterm\PosePrior_GT_Input `
  --pose-json Midterm\PosePrior_Sapiens2_0160_0167\PosePrior_Sapiens2_0160_0167_predictions.json `
  --output-dir Midterm\PosePrior_GTEvaluation
```

## 6. 验收标准

第一版建议采用以下门槛，而不是只看一项指标：

- 手部 ROI Recall 提升；
- 手部 ROI SAD、MSE 和漏检像素下降；
- 整帧误检像素的增加可控；
- 逐帧可视化中没有明显“凭空长出”的手臂区域；
- 所有输入尺寸和帧号严格匹配，缺帧或尺寸不一致直接报错。

如果 Recall 提升但误检明显增加，下一版应将几何先验改为“局部高分辨率 Matting 引导”，而不是继续提高目标 Alpha。
