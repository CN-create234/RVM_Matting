import os
import time
import cv2
import NDIlib as ndi
import numpy as np

# =============================
# 配置
# =============================
IMAGE_DIR = r"D:/CODES/RVM/DebugOutputs"
FPS = 30

NDI_VIDEO_NAME = "NDI_Video"
NDI_MASK_NAME  = "NDI_Mask"
# =============================

# -----------------------------
# 收集视频帧（主索引）
# -----------------------------
video_frames = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.startswith("fgrimg_") and f.endswith(".png")
])

if not video_frames:
    raise RuntimeError("没有找到 fgrimg_*.png")

# 用第一帧确定分辨率
first_img = cv2.imread(os.path.join(IMAGE_DIR, video_frames[0]), cv2.IMREAD_COLOR)
H, W = first_img.shape[:2]

# -----------------------------
# 初始化 NDI
# -----------------------------
if not ndi.initialize():
    raise RuntimeError("NDI 初始化失败")

def create_sender(name):
    desc = ndi.SendCreate()
    desc.ndi_name = name.encode("utf-8")
    sender = ndi.send_create(desc)
    if sender is None:
        raise RuntimeError(f"创建 NDI Sender 失败: {name}")
    return sender

sender_video = create_sender(NDI_VIDEO_NAME)
sender_mask  = create_sender(NDI_MASK_NAME)

# -----------------------------
# 通用 VideoFrame
# -----------------------------
def create_frame():
    f = ndi.VideoFrameV2()
    f.xres = W
    f.yres = H
    f.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
    f.line_stride_in_bytes = W * 4
    return f

video_frame = create_frame()
mask_frame  = create_frame()

frame_interval = 1.0 / FPS

print(f"开始发送 {len(video_frames)} 帧")

# -----------------------------
# 主循环（严格同步）
# -----------------------------
for idx, video_name in enumerate(video_frames):
    index = video_name.split("_")[1].split(".")[0]
    mask_name = f"alpha_{index}.png"

    video_path = os.path.join(IMAGE_DIR, video_name)
    mask_path  = os.path.join(IMAGE_DIR, mask_name)

    if not os.path.exists(mask_path):
        print(f"⚠ 缺少 mask: {mask_name}")
        continue

    # ===== Video =====
    video_bgr = cv2.imread(video_path, cv2.IMREAD_COLOR)
    video_bgra = cv2.cvtColor(video_bgr, cv2.COLOR_BGR2BGRA)
    video_frame.data = video_bgra

    # ===== Mask =====
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bgra = np.zeros((H, W, 4), dtype=np.uint8)
    mask_bgra[:, :, 0] = mask_gray
    mask_bgra[:, :, 1] = mask_gray
    mask_bgra[:, :, 2] = mask_gray
    mask_bgra[:, :, 3] = 255
    mask_frame.data = mask_bgra

    # ===== 同一时间点发送 =====
    ndi.send_send_video_v2(sender_video, video_frame)
    ndi.send_send_video_v2(sender_mask,  mask_frame)

    print(f"[{idx+1}/{len(video_frames)}] video={video_name} mask={mask_name}")
    time.sleep(frame_interval)

# -----------------------------
# 清理
# -----------------------------
ndi.send_destroy(sender_video)
ndi.send_destroy(sender_mask)
ndi.destroy()

print("NDI 发送完成")
