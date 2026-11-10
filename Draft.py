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


png_sequence_to_mp4(
    input_dir="D:/CODES/RVM/Results/Test/CompositingResults",          # 当前目录
    output_path="D:/CODES/RVM/Results/Test/CompositingResults/out_50.mp4",
    fps=50                  # 可改，比如 24 / 30 / 60
)
