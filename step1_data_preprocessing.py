# ============================================================
# STEP 1: DATA LOADING & PREPROCESSING
# Dataset: Waste Classification Data (Kaggle - techsash)
# https://www.kaggle.com/datasets/techsash/waste-classification-data
# Classes: Organic (O) vs Recyclable (R)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------------------------------------
# 1.1 CONFIGURATION
# ------------------------------------------------------------
IMG_SIZE    = (224, 224)   # Input size expected by transfer learning models
BATCH_SIZE  = 32
SEED        = 42

# After downloading from Kaggle the folder structure is:
#   DATASET/
#       TRAIN/
#           O/   (Organic)
#           R/   (Recyclable)
#       TEST/
#           O/
#           R/

TRAIN_DIR = "DATASET/TRAIN"
TEST_DIR  = "DATASET/TEST"

# ------------------------------------------------------------
# 1.2 DATA GENERATORS  (augmentation only on training set)
# ------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # Normalize pixel values to [0, 1]
    rotation_range=20,           # Random rotation ±20°
    width_shift_range=0.15,      # Horizontal shift
    height_shift_range=0.15,     # Vertical shift
    shear_range=0.10,            # Shear transformation
    zoom_range=0.15,             # Random zoom
    horizontal_flip=True,        # Mirror images
    fill_mode="nearest",         # Fill empty pixels
    validation_split=0.15        # 15 % of training data used for validation
)

# No augmentation on validation / test — only rescaling
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Training generator
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",         # Binary: Organic vs Recyclable
    subset="training",
    seed=SEED,
    shuffle=True
)

# Validation generator (carved out of TRAIN_DIR)
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=SEED,
    shuffle=False
)

# Test generator (from TEST_DIR — no augmentation, no shuffle)
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ------------------------------------------------------------
# 1.3 DATASET SUMMARY
# ------------------------------------------------------------
class_names = list(train_generator.class_indices.keys())
print("=" * 50)
print("CLASS MAPPING:", train_generator.class_indices)
print(f"Training   samples : {train_generator.samples}")
print(f"Validation samples : {val_generator.samples}")
print(f"Test       samples : {test_generator.samples}")
print(f"Classes            : {class_names}")
print("=" * 50)

# ------------------------------------------------------------
# 1.4 CLASS BALANCE VISUALISATION
# ------------------------------------------------------------
def plot_class_distribution(generator, title):
    labels = generator.classes
    counts = np.bincount(labels)
    plt.figure(figsize=(6, 4))
    plt.bar(class_names, counts, color=["#2ecc71", "#3498db"], edgecolor="black")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    for i, v in enumerate(counts):
        plt.text(i, v + 30, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()

plot_class_distribution(train_generator, "Training Set Class Distribution")
plot_class_distribution(test_generator,  "Test Set Class Distribution")

# ------------------------------------------------------------
# 1.5 SAMPLE IMAGE GRID
# ------------------------------------------------------------
def show_sample_images(generator, n=8):
    images, labels = next(generator)
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 6))
    axes = axes.flatten()
    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].set_title(class_names[int(labels[i])], fontsize=9)
        axes[i].axis("off")
    plt.suptitle("Sample Augmented Training Images", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150)
    plt.show()

show_sample_images(train_generator)

print("\n[STEP 1 COMPLETE] Generators are ready for model training.")
