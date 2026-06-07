# ============================================================
# STEP 3: TRANSFER LEARNING
# Models: MobileNetV2 | ResNet50 | EfficientNetB0
# Strategy: Feature extraction first → then fine-tuning
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

IMG_SHAPE = (224, 224, 3)

# ============================================================
# 3.1  GENERIC BUILDER  (used for all three backbones)
# ============================================================
def build_transfer_model(base_model_fn, model_name, trainable_layers=0):
    """
    Parameters
    ----------
    base_model_fn   : Keras application constructor (e.g. MobileNetV2)
    model_name      : string label for saving / printing
    trainable_layers: how many top layers of base to unfreeze (0 = full freeze)
    """
    # Load backbone WITHOUT the top classifier
    base = base_model_fn(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SHAPE
    )

    # ── Phase 1: freeze entire backbone ──────────────────────
    base.trainable = False

    # ── Classification head ───────────────────────────────────
    inputs  = tf.keras.Input(shape=IMG_SHAPE)
    x       = base(inputs, training=False)          # BN layers stay in inference mode
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.50)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name=model_name)
    return model, base


# ============================================================
# 3.2  COMPILE HELPER
# ============================================================
def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )


# ============================================================
# 3.3  CALLBACKS HELPER
# ============================================================
def get_callbacks(model_name):
    return [
        EarlyStopping(monitor="val_auc", patience=6,
                      restore_best_weights=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f"best_{model_name}.h5", monitor="val_auc",
                        save_best_only=True, mode="max", verbose=1)
    ]


# ============================================================
# 3.4  TRAINING HELPER  (2-phase: freeze → fine-tune)
# ============================================================
def train_transfer_model(base_model_fn, model_name,
                         train_gen, val_gen,
                         epochs_frozen=10,
                         epochs_finetune=20,
                         unfreeze_last_n=30):
    """
    Phase 1 – Train only the head (backbone frozen, fast convergence).
    Phase 2 – Unfreeze last N layers of backbone and fine-tune with low LR.
    """
    model, base = build_transfer_model(base_model_fn, model_name)
    model.summary()

    # ── PHASE 1: Feature extraction ───────────────────────────
    print(f"\n{'='*55}")
    print(f"  {model_name}  |  Phase 1: Feature Extraction")
    print(f"{'='*55}")
    compile_model(model, lr=1e-3)
    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_frozen,
        callbacks=get_callbacks(f"{model_name}_phase1"),
        verbose=1
    )

    # ── PHASE 2: Fine-tuning ──────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {model_name}  |  Phase 2: Fine-Tuning (last {unfreeze_last_n} layers)")
    print(f"{'='*55}")
    base.trainable = True
    # Freeze all layers except the last `unfreeze_last_n`
    for layer in base.layers[:-unfreeze_last_n]:
        layer.trainable = False

    compile_model(model, lr=1e-5)           # Much lower LR for fine-tuning
    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_finetune,
        initial_epoch=h1.epoch[-1] + 1,    # Continue epoch count
        callbacks=get_callbacks(f"{model_name}_phase2"),
        verbose=1
    )

    return model, h1, h2


# ============================================================
# 3.5  PLOT HELPER
# ============================================================
def plot_two_phase_curves(h1, h2, model_name):
    """Concatenate phase-1 and phase-2 histories and plot."""
    def merge(key):
        return h1.history.get(key, []) + h2.history.get(key, [])

    metrics = [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    phase_boundary = len(h1.epoch)

    for ax, (metric, label) in zip(axes, metrics):
        train_vals = merge(metric)
        val_vals   = merge(f"val_{metric}")
        ax.plot(train_vals, label="Train",      linewidth=2)
        ax.plot(val_vals,   label="Validation", linewidth=2, linestyle="--")
        ax.axvline(phase_boundary, color="red", linestyle=":", linewidth=1.5,
                   label="Fine-tune start")
        ax.set_title(f"{model_name} — {label}", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(label)
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"curves_{model_name}.png", dpi=150)
    plt.show()


# ============================================================
# 3.6  TRAIN ALL THREE MODELS
#      (assumes train_generator, val_generator from Step 1)
# ============================================================

# MobileNetV2 ─────────────────────────────────────────────
mobilenet_model, h1_mb, h2_mb = train_transfer_model(
    MobileNetV2, "MobileNetV2",
    train_generator, val_generator,
    epochs_frozen=10, epochs_finetune=20, unfreeze_last_n=30
)
plot_two_phase_curves(h1_mb, h2_mb, "MobileNetV2")

# ResNet50 ────────────────────────────────────────────────
resnet_model, h1_rn, h2_rn = train_transfer_model(
    ResNet50, "ResNet50",
    train_generator, val_generator,
    epochs_frozen=10, epochs_finetune=20, unfreeze_last_n=30
)
plot_two_phase_curves(h1_rn, h2_rn, "ResNet50")

# EfficientNetB0 ──────────────────────────────────────────
effnet_model, h1_en, h2_en = train_transfer_model(
    EfficientNetB0, "EfficientNetB0",
    train_generator, val_generator,
    epochs_frozen=10, epochs_finetune=20, unfreeze_last_n=30
)
plot_two_phase_curves(h1_en, h2_en, "EfficientNetB0")

print("\n[STEP 3 COMPLETE] All transfer learning models trained and saved.")
