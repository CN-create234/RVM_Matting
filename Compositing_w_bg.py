import os
import cv2
import numpy as np

# ================== 配置 ==================
RVM_DIR = r"D:/CODES/RVM/Test12212206"        # *_fgr.png, *_alpha.png
BG_DIR  = r"D:/Master/ComposureResults/CG/renamed/png"       # 0000.png, 0001.png, ...
OUT_DIR = r"D:/CODES/RVM/Test12212206/Compositing_bg"     # 输出目录
# =========================================

os.makedirs(OUT_DIR, exist_ok=True)

# 找到所有前景帧
fgr_files = sorted([
    f for f in os.listdir(RVM_DIR)
    if f.endswith("_fgr.png")
])

if not fgr_files:
    raise RuntimeError("未找到 *_fgr.png 文件")

for fgr_name in fgr_files:
    base = fgr_name.replace("_fgr.png", "")
    alpha_name = base + "_alpha.png"
    bg_name = base + ".png"

    fgr_path = os.path.join(RVM_DIR, fgr_name)
    alpha_path = os.path.join(RVM_DIR, alpha_name)
    bg_path = os.path.join(BG_DIR, bg_name)

    if not os.path.exists(alpha_path):
        print(f"[WARN] 缺少 alpha: {alpha_name}")
        continue

    if not os.path.exists(bg_path):
        print(f"[WARN] 缺少背景: {bg_name}")
        continue

    # 读取图像
    fgr = cv2.imread(fgr_path, cv2.IMREAD_COLOR)
    alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)

    if fgr is None or alpha is None or bg is None:
        print(f"[WARN] 读取失败: {base}")
        continue

    h, w = fgr.shape[:2]

    # 背景尺寸不一致则 resize
    if bg.shape[0] != h or bg.shape[1] != w:
        print("resized!")
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)

    # Alpha 归一化
    alpha_f = alpha.astype(np.float32) / 255.0
    alpha_f = alpha_f[:, :, None]   # HWC

    # 合成
    comp = (
        fgr.astype(np.float32) * alpha_f +
        bg.astype(np.float32) * (1.0 - alpha_f)
    ).astype(np.uint8)

    out_path = os.path.join(OUT_DIR, base + ".png")
    cv2.imwrite(out_path, comp)

    print(f"[OK] {base}.png")

print("人物 + 虚拟背景 合成完成")
