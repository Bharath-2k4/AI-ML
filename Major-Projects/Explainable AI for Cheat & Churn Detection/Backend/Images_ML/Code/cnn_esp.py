import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# =========================
# CONFIG
# =========================
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

TRAIN_DIR = r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Images_ML\TRAIN"
MODEL_OUT = r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Images_ML\Models\cheat_esp_cnn.h5"

# =========================
# DATA GENERATORS
# =========================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# =========================
# VERIFY CLASS MAPPING
# =========================
print("Class mapping:", train_gen.class_indices)

# MUST be:
# {'0_Non_Cheat': 0, '1_Cheat_ESP': 1}
expected = {'0_Non_Cheat': 0, '1_Cheat_ESP': 1}
if train_gen.class_indices != expected:
    raise ValueError(
        f"❌ Class mapping incorrect: {train_gen.class_indices}\n"
        f"Expected: {expected}\n"
        f"Check folder names inside TRAIN directory."
    )

# =========================
# CLASS WEIGHTS
# =========================
# Non-Cheat = 2050 images
# Cheat_ESP = 1120 images
class_weights = {
    0: 1.0,           # Non-Cheat
    1: 2050 / 1120    # Cheat_ESP ≈ 1.83
}

# =========================
# CNN MODEL
# =========================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu",
           input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(1, activation="sigmoid")  # P(CHEAT)
])

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights
)

# =========================
# SAVE MODEL
# =========================
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
model.save(MODEL_OUT)

print("✅ Model training complete and saved.")
