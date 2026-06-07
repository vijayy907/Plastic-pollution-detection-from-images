# ============================================================
# STEP 2: CUSTOM CNN MODEL (Baseline)
# A simple CNN built from scratch — no pre-trained weights
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Re-use generators from Step 1 (run step1 first, or paste generators here)
# from step1_data_preprocessing import train_generator, val_generator, test_generator

IMG_SIZE   = (224, 224)
EPOCHS     = 30
MODEL_NAME = "custom_cnn"

# ------------------------------------------------------------
# 2.1 BUILD CUSTOM CNN
# ------------------------------------------------------------
def build_custom_cnn(input_shape=(224, 224, 3)):
    """
    3-block CNN:
      Block 1 → 32 filters
      Block 2 → 64 filters
      Block 3 → 128 filters
    Each block: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
    Head: GlobalAvgPool → Dense(128) → Dropout → Sigmoid output
    """
    model = models.Sequential([
        # ── Input ──────────────────────────────────────────
        layers.Input(shape=input_shape),

        # ── Block 1 ────────────────────────────────────────
        layers.Conv2D(32, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 2 ────────────────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 3 ────────────────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.40),

        # ── Classification Head ─────────────────────────────
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(1, activation="sigmoid")   # Binary output
    ], name=MODEL_NAME)

    return model

custom_cnn = build_custom_cnn()
custom_cnn.summary()

# ------------------------------------------------------------
# 2.2 COMPILE
# ------------------------------------------------------------
custom_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

# ------------------------------------------------------------
# 2.3 CALLBACKS
# ------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_auc", patience=7,
                  restore_best_weights=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(f"best_{MODEL_NAME}.h5", monitor="val_auc",
                    save_best_only=True, mode="max", verbose=1)
]

# ------------------------------------------------------------
# 2.4 TRAIN
# ------------------------------------------------------------
history_cnn = custom_cnn.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------------------------------------
# 2.5 PLOT TRAINING CURVES
# ------------------------------------------------------------
def plot_training_curves(history, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    metrics = [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")]
    colors  = [("#2ecc71", "#27ae60"), ("#e74c3c", "#c0392b"), ("#3498db", "#2980b9")]

    for ax, (metric, label), (c1, c2) in zip(axes, metrics, colors):
        ax.plot(history.history[metric],          color=c1, label="Train",      linewidth=2)
        ax.plot(history.history[f"val_{metric}"], color=c2, label="Validation", linewidth=2, linestyle="--")
        ax.set_title(f"{model_name} — {label}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"training_curves_{model_name}.png", dpi=150)
    plt.show()

plot_training_curves(history_cnn, MODEL_NAME)
print(f"\n[STEP 2 COMPLETE] Custom CNN trained. Best model saved as best_{MODEL_NAME}.h5")
