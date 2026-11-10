import os
import cv2
import numpy as np

# ================== 配置 ==================
INPUT_DIR  = r"D:/CODES/RVM/MainGraph"     # 包含 *_fgr.png 和 *_alpha.png
OUTPUT_DIR = r"D:/CODES/RVM/MainGraph/Output"
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.endswith("_fgr.png")
])

if not files:
    raise RuntimeError("未找到 *_fgr.png 文件")

for fgr_name in files:
    base = fgr_name.replace("_fgr.png", "")
    alpha_name = base + "_alpha.png"

    fgr_path   = os.path.join(INPUT_DIR, fgr_name)
    alpha_path = os.path.join(INPUT_DIR, alpha_name)

    if not os.path.exists(alpha_path):
        print(f"[WARN] 缺少 alpha: {alpha_name}")
        continue

    fgr = cv2.imread(fgr_path, cv2.IMREAD_COLOR)
    alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)

    if fgr is None or alpha is None:
        print(f"[WARN] 读取失败: {base}")
        continue

    # 组装 RGBA
    rgba = cv2.cvtColor(fgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    out_path = os.path.join(OUTPUT_DIR, base + ".png")
    cv2.imwrite(out_path, rgba)

    print(f"[OK] {base}.png")

print("RGBA 合成完成")
