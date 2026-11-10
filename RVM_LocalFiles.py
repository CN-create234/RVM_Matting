import os
import cv2
import numpy as np
import onnxruntime as ort

# ================== 配置 ==================
MODEL_PATH = "rvm_mobilenetv3_fp16.onnx"

INPUT_DIR  = r"D:/Master/ComposureResults/Video/renamed/png"     # 输入 PNG
OUTPUT_DIR = r"./Test12212206"     # 输出目录

DOWNSAMPLE_RATIO = 0.25
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ ONNX Runtime ------------------
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
sess = ort.InferenceSession(MODEL_PATH, providers=providers)

input_names  = [i.name for i in sess.get_inputs()]
output_names = [o.name for o in sess.get_outputs()]

print("ONNX Inputs :", input_names)
print("ONNX Outputs:", output_names)

# ------------------ 初始化 recurrent states ------------------
# 与 LiveFeed 完全一致（占位，真实 shape 由 ONNX 推理返回）
rec = [np.zeros((1, 1, 1, 1), dtype=np.float16) for _ in range(4)]

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

    feed["downsample_ratio"] = np.array([DOWNSAMPLE_RATIO], dtype=np.float32)

    outs = sess.run(None, feed)

    fgr = outs[0]        # [1,3,H,W]
    pha = outs[1]        # [1,1,H,W]
    new_rec = outs[2:6]  # recurrent outputs

    return fgr, pha, new_rec

def to_bgr_image(tex):
    img = np.transpose(tex[0], (1, 2, 0))
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# ------------------ 批量处理 ------------------
files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".png")
])

if not files:
    raise RuntimeError("输入目录中没有 PNG 文件")

for idx, fname in enumerate(files):
    path = os.path.join(INPUT_DIR, fname)
    frame = cv2.imread(path)

    if frame is None:
        print(f"[WARN] 读取失败: {fname}")
        continue

    inp = preprocess_frame(frame)

    fgr, pha, rec = run_inference(inp, rec)

    # foreground
    fgr_img = to_bgr_image(fgr)

    # alpha
    alpha = (np.transpose(pha[0], (1, 2, 0))[:, :, 0] * 255).astype(np.uint8)

    base = os.path.splitext(fname)[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_fgr.png"), fgr_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_alpha.png"), alpha)

    print(f"[OK] {fname}")

print("RVM 批量分割完成")
