import cv2
import os
import random

# =============================
# CONFIG (TUNED FOR REAL CHEAT)
# =============================
INPUT_DIR = r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Images_ML\TRAIN\Cheat"
OUTPUT_DIR = r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Images_ML\TRAIN\Cheat_ESP"

# ESP appearance
BOX_COLOR = (0, 255, 0)          # bright green (BGR)
THICKNESS_RANGE = (3, 5)         # thick enough to dominate
BOX_COUNT_RANGE = (1, 4)         # multiple ESP boxes

# Box size relative to image
MIN_BOX_RATIO = 0.15             # 15% of min dimension
MAX_BOX_RATIO = 0.35             # 35% of min dimension

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================
# DRAW ONE ESP BOX
# =============================
def draw_esp_box(img):
    h, w, _ = img.shape
    min_dim = min(h, w)

    box_size = random.randint(
        int(min_dim * MIN_BOX_RATIO),
        int(min_dim * MAX_BOX_RATIO)
    )

    thickness = random.randint(*THICKNESS_RANGE)

    x1 = random.randint(0, max(1, w - box_size - 1))
    y1 = random.randint(0, max(1, h - box_size - 1))
    x2 = x1 + box_size
    y2 = y1 + box_size

    cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, thickness)


# =============================
# MAIN LOOP
# =============================
count = 0

for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    esp_img = img.copy()

    num_boxes = random.randint(*BOX_COUNT_RANGE)
    for _ in range(num_boxes):
        draw_esp_box(esp_img)

    save_path = os.path.join(OUTPUT_DIR, file)
    cv2.imwrite(save_path, esp_img)
    count += 1

print(f"✅ ESP overlays added to {count} real cheat images")
