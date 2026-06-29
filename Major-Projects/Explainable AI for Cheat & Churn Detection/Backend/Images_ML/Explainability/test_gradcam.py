import os
from gradcam import run_gradcam

# No save_dir anymore – handled internally
result = run_gradcam(
    image_path=r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Images_ML\TRAIN\1_Cheat_ESP\C (8).jpg",
    model_path=r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Images_ML\Models\cheat_esp_cnn.h5"
)

print("RESULT:", result)
