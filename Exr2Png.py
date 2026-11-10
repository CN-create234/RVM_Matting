import os
import OpenEXR
import Imath
import numpy as np
import cv2

# ================= 配置 =================
# SRC_DIR = r"D:/Master/ComposureResults/CG"          # EXR 文件所在目录
# DST_DIR = r"D:/Master/ComposureResults/CG/png"          # 新的 PNG 输出目录
SRC_DIR = r"D:/Master/ComposureResults/Video5"          # EXR 文件所在目录
DST_DIR = r"D:/Master/ComposureResults/Video5/png"          # 新的 PNG 输出目录
START = 0
END = 1289                                     # 0000.exr ~ 0523.exr
DIGITS = 4                                   # 文件名位数：0000
GAMMA = 2.2
# =======================================

os.makedirs(DST_DIR, exist_ok=True)

pt = Imath.PixelType(Imath.PixelType.FLOAT)

for i in range(START, END + 1):
    name = f"{i:0{DIGITS}d}"
    exr_path = os.path.join(SRC_DIR, name + ".exr")
    png_path = os.path.join(DST_DIR, name + ".png")

    if not os.path.exists(exr_path):
        print(f"[WARN] 文件不存在，跳过: {exr_path}")
        continue

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
    img = img ** (1.0 / GAMMA)

    img8 = (img * 255.0).astype(np.uint8)
    img8 = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)

    cv2.imwrite(png_path, img8)
    exr.close()

    print(f"[OK] {name}.exr → {name}.png")

print("全部转换完成")
