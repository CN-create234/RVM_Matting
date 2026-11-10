import os
import argparse
import shutil
import OpenEXR
import Imath
import numpy as np
import cv2
import onnxruntime as ort

from depth_guided_matting import (
    BackgroundDepthModel,
    DepthFusionConfig,
    find_depth_frame,
    fuse_alpha_with_depth,
    load_depth,
)

def rename_exr_sequence(
    src_dir,
    dst_subdir="renamed",
    ext=".exr",
    start_index=0,
    digits=4,
    copy=True
):
    """
    将 src_dir 中的 EXR 文件按排序后重命名为 0000.exr, 0001.exr, ...
    并放入 src_dir/dst_subdir 目录中。

    参数：
    - src_dir:        原 EXR 文件夹路径
    - dst_subdir:     新建的输出子文件夹名（默认 "renamed"）
    - ext:            文件扩展名（默认 ".exr"）
    - start_index:    起始编号（默认 0）
    - digits:         编号位数（默认 4，对应 0000）
    - copy:           True=复制文件，False=移动文件
    """

    dst_dir = os.path.join(src_dir, dst_subdir)
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(src_dir)
        if f.lower().endswith(ext)
    ])

    if not files:
        raise RuntimeError(f"目录中未找到 {ext} 文件: {src_dir}")

    for i, fname in enumerate(files):
        src_path = os.path.join(src_dir, fname)
        new_name = f"{start_index + i:0{digits}d}{ext}"
        dst_path = os.path.join(dst_dir, new_name)

        if copy:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)

    print(f"[Rename] 完成：{len(files)} 个文件 → {dst_dir}")

    return dst_dir

def transform_exr_to_png(
    src_dir,
    dst_subdir="png",
    ext=".exr",
    gamma=2.2
):
    """
    将 src_dir 中的所有 EXR 文件转换为 PNG，按文件名一一对应输出。

    参数：
    - src_dir:     输入 EXR 文件夹（通常是 rename 之后的目录）
    - dst_subdir:  输出 PNG 的子文件夹名（默认 "png"）
    - ext:         输入文件扩展名（默认 ".exr"）
    - gamma:       Gamma 校正值（EXR linear -> sRGB，默认 2.2）

    返回：
    - dst_dir:     PNG 输出目录路径
    """

    dst_dir = os.path.join(src_dir, dst_subdir)
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(src_dir)
        if f.lower().endswith(ext)
    ])

    if not files:
        raise RuntimeError(f"目录中未找到 {ext} 文件: {src_dir}")

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    for fname in files:
        name = os.path.splitext(fname)[0]
        exr_path = os.path.join(src_dir, fname)
        png_path = os.path.join(dst_dir, name + ".png")

        exr = OpenEXR.InputFile(exr_path)

        dw = exr.header()['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        R = np.frombuffer(exr.channel('R', pt), dtype=np.float32).reshape(h, w)
        G = np.frombuffer(exr.channel('G', pt), dtype=np.float32).reshape(h, w)
        B = np.frombuffer(exr.channel('B', pt), dtype=np.float32).reshape(h, w)

        img = np.stack([R, G, B], axis=-1)

        # Linear → sRGB
        img = np.clip(img, 0.0, 1.0)
        img = img ** (1.0 / gamma)

        img8 = (img * 255.0).astype(np.uint8)
        img8 = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)

        cv2.imwrite(png_path, img8)
        exr.close()

        print(f"[EXR→PNG] {fname} → {name}.png")

    print(f"[EXR→PNG] 完成，共转换 {len(files)} 个文件 → {dst_dir}")

    return dst_dir

def run_rvm_batch(
    input_dir,
    output_dir,
    model_path="rvm_mobilenetv3_fp16.onnx",
    downsample_ratio=0.25,
    providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    depth_dir=None,
    background_depth_dir=None,
    depth_scale=1.0,
    depth_fusion_config=None,
    save_depth_diagnostics=False,
):
    """
    对 input_dir 中的 PNG 序列进行 RVM 批量分割，
    输出 *_fgr.png 和 *_alpha.png 到 output_dir。

    参数：
    - input_dir:        输入 PNG 文件夹
    - output_dir:       输出目录
    - model_path:       RVM ONNX 模型路径
    - downsample_ratio: RVM downsample_ratio
    - providers:        ONNX Runtime providers
    - depth_dir:        与 RGB 同名、且已配准到 RGB 坐标系的深度序列；None=保持原 RVM
    - background_depth_dir: 无演员背景深度序列，用于建立逐像素背景模型
    - depth_scale:      深度读入后的单位缩放，阈值默认按毫米设置
    - depth_fusion_config: DepthFusionConfig；None 时使用默认参数
    - save_depth_diagnostics: 保存深度前景/背景置信度图
    """

    os.makedirs(output_dir, exist_ok=True)

    # ------------------ ONNX Runtime ------------------
    sess = ort.InferenceSession(model_path, providers=list(providers))

    input_names  = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]

    print("[RVM] ONNX Inputs :", input_names)
    print("[RVM] ONNX Outputs:", output_names)

    # ------------------ 初始化 recurrent states ------------------
    rec = [np.zeros((1, 1, 1, 1), dtype=np.float16) for _ in range(4)]

    background_depth_model = None
    if (depth_dir is None) != (background_depth_dir is None):
        raise ValueError("depth_dir 和 background_depth_dir 必须同时提供")
    if depth_dir is not None:
        background_depth_model = BackgroundDepthModel.from_directory(
            background_depth_dir,
            depth_scale=depth_scale,
        )
        if depth_fusion_config is None:
            depth_fusion_config = DepthFusionConfig()
        if save_depth_diagnostics:
            os.makedirs(os.path.join(output_dir, "depth_diagnostics"), exist_ok=True)

    # ------------------ 预处理 ------------------
    def preprocess_frame(bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float16) / 255.0
        inp = np.transpose(rgb, (2, 0, 1))[None]  # NCHW
        return inp

    def run_inference(inp, rec):
        feed = {}
        feed[input_names[0]] = inp

        for name, r in zip(input_names[1:], rec):
            feed[name] = r

        feed["downsample_ratio"] = np.array([downsample_ratio], dtype=np.float32)

        outs = sess.run(None, feed)

        fgr = outs[0]
        pha = outs[1]
        new_rec = outs[2:6]

        return fgr, pha, new_rec

    def to_bgr_image(tex):
        img = np.transpose(tex[0], (1, 2, 0))
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ------------------ 批量处理 ------------------
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    ])

    if not files:
        raise RuntimeError(f"[RVM] 输入目录中没有 PNG 文件: {input_dir}")

    for idx, fname in enumerate(files):
        path = os.path.join(input_dir, fname)
        frame = cv2.imread(path)

        if frame is None:
            print(f"[WARN] 读取失败: {fname}")
            continue

        inp = preprocess_frame(frame)
        fgr, pha, rec = run_inference(inp, rec)

        fgr_img = to_bgr_image(fgr)
        alpha = (np.transpose(pha[0], (1, 2, 0))[:, :, 0] * 255).astype(np.uint8)

        base = os.path.splitext(fname)[0]
        if background_depth_model is not None:
            depth_path = find_depth_frame(depth_dir, base)
            depth = load_depth(depth_path, depth_scale=depth_scale)
            if save_depth_diagnostics:
                alpha, diagnostics = fuse_alpha_with_depth(
                    alpha,
                    depth,
                    background_depth_model,
                    depth_fusion_config,
                    return_diagnostics=True,
                )
                diagnostic_dir = os.path.join(output_dir, "depth_diagnostics")
                for diagnostic_name in ("foreground_score", "background_score", "valid"):
                    diagnostic = np.round(
                        np.clip(diagnostics[diagnostic_name], 0, 1) * 255
                    ).astype(np.uint8)
                    cv2.imwrite(
                        os.path.join(diagnostic_dir, f"{base}_{diagnostic_name}.png"),
                        diagnostic,
                    )
            else:
                alpha = fuse_alpha_with_depth(
                    alpha,
                    depth,
                    background_depth_model,
                    depth_fusion_config,
                )

        cv2.imwrite(os.path.join(output_dir, f"{base}_fgr.png"), fgr_img)
        cv2.imwrite(os.path.join(output_dir, f"{base}_alpha.png"), alpha)

        print(f"[RVM] {fname}")

    print(f"[RVM] 批量分割完成，共 {len(files)} 帧 → {output_dir}")

    return output_dir

def composite_rvm_with_background(
    rvm_dir,
    bg_dir,
    out_dir
):
    """
    使用 RVM 分割结果（*_fgr.png + *_alpha.png）
    与对应编号的背景 PNG 进行虚实合成。

    参数：
    - rvm_dir:  包含 *_fgr.png 和 *_alpha.png 的目录
    - bg_dir:   背景 PNG 目录（如 0000.png, 0001.png, ...）
    - out_dir:  合成结果输出目录
    """

    os.makedirs(out_dir, exist_ok=True)

    fgr_files = sorted([
        f for f in os.listdir(rvm_dir)
        if f.endswith("_fgr.png")
    ])

    if not fgr_files:
        raise RuntimeError("未找到 *_fgr.png 文件")

    for fgr_name in fgr_files:
        base = fgr_name.replace("_fgr.png", "")
        alpha_name = base + "_alpha.png"
        bg_name = base + ".png"

        fgr_path = os.path.join(rvm_dir, fgr_name)
        alpha_path = os.path.join(rvm_dir, alpha_name)
        bg_path = os.path.join(bg_dir, bg_name)

        if not os.path.exists(alpha_path):
            print(f"[WARN] 缺少 alpha: {alpha_name}")
            continue

        if not os.path.exists(bg_path):
            print(f"[WARN] 缺少背景: {bg_name}")
            continue

        fgr = cv2.imread(fgr_path, cv2.IMREAD_COLOR)
        alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)

        if fgr is None or alpha is None or bg is None:
            print(f"[WARN] 读取失败: {base}")
            continue

        h, w = fgr.shape[:2]

        # 背景尺寸不一致则 resize
        if bg.shape[0] != h or bg.shape[1] != w:
            bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)

        # Alpha 归一化
        alpha_f = alpha.astype(np.float32) / 255.0
        alpha_f = alpha_f[:, :, None]

        # 合成
        comp = (
            fgr.astype(np.float32) * alpha_f +
            bg.astype(np.float32) * (1.0 - alpha_f)
        ).astype(np.uint8)

        out_path = os.path.join(out_dir, base + ".png")
        cv2.imwrite(out_path, comp)

        print(f"[Composite] {base}.png")

    print(f"[Composite] 虚实合成完成 → {out_dir}")

    return out_dir

# 视频生成
import os
import cv2

def png_sequence_to_mp4(
    input_dir=".",
    output_path="output.mp4",
    fps=50,
    pattern=".png"
):
    """
    将指定目录中的 PNG 序列合成为 MP4 视频

    参数：
        input_dir (str): 图片所在目录（默认当前目录）
        output_path (str): 输出视频路径
        fps (int): 帧率（默认 50）
        pattern (str): 文件后缀过滤（默认 .png）
    """

    # 1. 收集并排序文件
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(pattern)
    ])

    if not files:
        raise ValueError("没有找到符合条件的图片文件")

    # 2. 读取第一张图确定分辨率
    first_img_path = os.path.join(input_dir, files[0])
    frame = cv2.imread(first_img_path)

    if frame is None:
        raise ValueError(f"无法读取图片: {first_img_path}")

    height, width, _ = frame.shape

    # 3. 初始化视频写入器（mp4v兼容性最好）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 4. 写入每一帧
    for fname in files:
        img_path = os.path.join(input_dir, fname)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"跳过损坏文件: {img_path}")
            continue

        # 尺寸检查（防止崩溃）
        if frame.shape[0] != height or frame.shape[1] != width:
            raise ValueError(f"图片尺寸不一致: {fname}")

        video.write(frame)

    video.release()
    print(f"视频已生成: {output_path}")

# 不需要copy & paste，只需要在D:/CODES/RVM/Results/Test/UEComposureResults/Video和D:/CODES/RVM/Results/Test/UEComposureResults/CG路径下生成即可。完成后test文件夹改名/加识别后缀即可。
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the EXR -> RVM -> composite -> MP4 pipeline."
    )
    parser.add_argument(
        "output_video",
        help="Output MP4 file name or path, for example result.mp4.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Output video frame rate. Default: 50.",
    )
    parser.add_argument(
        "--depth-dir",
        default=None,
        help="Registered depth sequence whose stems match the RGB frame names.",
    )
    parser.add_argument(
        "--background-depth-dir",
        default=None,
        help="Actor-free depth frames used to build the fixed-background model.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Scale applied after reading depth. Defaults assume millimetres.",
    )
    parser.add_argument(
        "--depth-background-tolerance",
        type=float,
        default=30.0,
        help="Depth difference still treated as background, in scaled depth units.",
    )
    parser.add_argument(
        "--depth-foreground-margin",
        type=float,
        default=100.0,
        help="Depth difference treated as reliable foreground, in scaled depth units.",
    )
    parser.add_argument(
        "--depth-boundary-width",
        type=int,
        default=5,
        help="Uncertain depth-edge width that falls back to RVM alpha.",
    )
    parser.add_argument(
        "--save-depth-diagnostics",
        action="store_true",
        help="Save depth foreground/background confidence maps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_video = args.output_video

    if not output_video.lower().endswith(".mp4"):
        output_video += ".mp4"

    output_parent = os.path.dirname(output_video)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    exr_dir_video = r"D:/CODES/RVM/Results/Test/UEComposureResults/Video"
    renamed_dir_video = rename_exr_sequence(exr_dir_video)
    exr_dir_cg = r"D:/CODES/RVM/Results/Test/UEComposureResults/CG"
    renamed_dir_cg = rename_exr_sequence(exr_dir_cg)

    png_dir_video = transform_exr_to_png(renamed_dir_video)
    png_dir_cg = transform_exr_to_png(renamed_dir_cg)

    rvm_out_dir = r"D:/CODES/RVM/Results/Test/RVMResults"
    depth_fusion_config = None
    if args.depth_dir is not None or args.background_depth_dir is not None:
        depth_fusion_config = DepthFusionConfig(
            background_tolerance=args.depth_background_tolerance,
            foreground_margin=args.depth_foreground_margin,
            boundary_width=args.depth_boundary_width,
        )
    run_rvm_batch(
        png_dir_video,
        rvm_out_dir,
        depth_dir=args.depth_dir,
        background_depth_dir=args.background_depth_dir,
        depth_scale=args.depth_scale,
        depth_fusion_config=depth_fusion_config,
        save_depth_diagnostics=args.save_depth_diagnostics,
    )

    final_out_dir = r"D:/CODES/RVM/Results/Test/CompositingResults"
    composite_rvm_with_background(
        rvm_dir=rvm_out_dir,
        bg_dir=png_dir_cg,
        out_dir=final_out_dir
    )

    png_sequence_to_mp4(
        input_dir=final_out_dir,
        output_path=output_video,
        fps=args.fps
    )


if __name__ == "__main__":
    main()

# png_sequence_to_mp4(
#     input_dir="D:/CODES/RVM/Results/Test/CompositingResults",          # 当前目录
#     output_path="out.mp4",
#     fps=50                  # 可改，比如 24 / 30 / 60
# )




