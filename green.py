from PIL import Image

w, h = 2304, 1152
img = Image.new("RGB", (w, h), (0, 255, 0))  # 纯绿
img.save("green_2304x1152.png")
print("Saved: green_2304x1152.png")
