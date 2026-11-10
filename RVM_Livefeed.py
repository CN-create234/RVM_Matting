# rvm_pipeline.py
import subprocess
import numpy as np
import cv2
import onnxruntime as ort
import time
from PIL import Image
import io
import os
from NDIlib import NDISender, VideoFrame
# Optional: NDI import if you choose NDI (see below)
# import ndi

# == config ==
FFMPEG_CMD = [
    "ffmpeg",
    "-f", "decklink",
    "-i", "DeckLink Duo (1)",   # <- 替换为你的设备名
    "-pix_fmt", "bgr24",
    "-s", "1280x720",           # 建议优先 720p 来稳帧率与推理速度
    "-f", "rawvideo",
    "-"
]
WIDTH, HEIGHT = 1280, 720
MODEL_PATH = "rvm_mobilenetv3_fp16.onnx"

# == start ffmpeg process reading rawvideo stdout ==
ffmpeg = subprocess.Popen(FFMPEG_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

# == setup ONNX runtime ==
providers = ['CUDAExecutionProvider']
sess_opts = ort.SessionOptions()
sess = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=providers)

# RVM expects recurrent states; init as zeros; shapes depend on model; use dynamic
rec = [np.zeros((1,1,1,1), dtype=np.float16) for _ in range(4)]  # placeholder, real shapes set below on first run

def preprocess_frame(bgr_frame):
    # bgr_frame: HxWx3 uint8
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float16) / 255.0
    # Model may expect NCHW
    inp = np.transpose(rgb, (2, 0, 1))[None].astype(np.float16)
    return inp

def run_inference(inp, rec):
    # input names vary by ONNX; inspect sess.get_inputs()
    input_names = [i.name for i in sess.get_inputs()]
    feed = {}
    feed[input_names[0]] = inp
    # subsequent names are recurrent states in many RVM onnx exports
    for name, r in zip(input_names[1:], rec):
        feed[name] = r
        # print("type1: ", r.dtype)

    # ValueError: Required inputs (['downsample_ratio']) are missing from input feed (['src', 'r1i', 'r2i', 'r3i', 'r4i']).
    feed["downsample_ratio"] = np.array([0.25], dtype=np.float32)
    # print("type2: ", feed["downsample_ratio"].dtype)

    outs = sess.run(None, feed)
    # Typical outputs: fgr, pha, r1o, r2o, r3o, r4o
    fgr = outs[0]   # [1,3,H,W]
    pha = outs[1]   # [1,1,H,W]
    new_rec = outs[2:6]
    return fgr, pha, new_rec

def to_bgr_image(tex):  # NCHW -> HWC BGR 0..255
    img = np.transpose(tex[0], (1,2,0))
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Optional: initialize NDI senders if using NDI (see below)
use_ndi = False
# if use_ndi:
#     ndi_sender_fgr = ndi.Sender("RVM_FGR")
#     ndi_sender_mask = ndi.Sender("RVM_MASK")

# main loop
frame_size_bytes = WIDTH * HEIGHT * 3
fps_count = 0
t0 = time.time()
try:
    frame_cnt = 0
    while True:
        # read raw frame
        raw = ffmpeg.stdout.read(frame_size_bytes)
        if not raw:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        inp = preprocess_frame(frame)  # NCHW float32
        # run inference
        # print("test:")
        # print(inp)
        # print(rec)
        fgr, pha, rec_new = run_inference(inp, rec)
        rec = rec_new
        # Convert outputs
        fgr_img = to_bgr_image(fgr)          # foreground RGB
        alpha = (np.transpose(pha[0], (1,2,0))[:, :, 0] * 255).astype(np.uint8)  # HxW
        # optionally composite fgr over transparent background (BGRA)
        bgra = cv2.cvtColor(fgr_img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha

        # ----- send out: two options: NDI or write to local RTSP/HTTP stream
        # For demo: simple imshow previews (remove in prod)
        output_dir = "DebugOutputs"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"alpha_{frame_cnt:04d}.png"), alpha)
        cv2.imwrite(os.path.join(output_dir, f"fgrimg_{frame_cnt:04d}.png"), fgr_img)
        print("output done!", frame_cnt)
        frame_cnt += 1

        # cv2.imshow("alpha", alpha)
        # cv2.imshow("foreground", fgr_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        fps_count += 1
        if fps_count % 60 == 0:
            print("FPS approx:", fps_count / (time.time() - t0))

finally:
    ffmpeg.stdout.close()
    ffmpeg.kill()
    cv2.destroyAllWindows()
