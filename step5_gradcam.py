# ============================================================
# STEP 5: GRAD-CAM EXPLAINABILITY
# Visualise what the CNN "looks at" for:
#   ✓ Correct predictions
#   ✗ False Positives  (model says Recyclable, truth is Organic)
#   ✗ False Negatives  (model says Organic,    truth is Recyclable)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2

CLASS_NAMES = ["Organic (O)", "Recyclable (R)"]

# ============================================================
# 5.1  GRAD-CAM CORE FUNCTION
# ============================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Parameters
    ----------
    img_array          : (1, H, W, 3) normalised image tensor
    model              : trained Keras model
    last_conv_layer_name: name of the last Conv2D layer in the backbone
    pred_index         : class index to explain (None → argmax)

    Returns
    -------
    heatmap : (H, W) numpy array in [0, 1]
    """
    # Sub-model: inputs → [last_conv_output, final_prediction]
    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        # For binary classification, the single output IS the class score
        if pred_index is None:
            class_channel = predictions[:, 0]
        else:
            # Flip for class 0 (Organic): score = 1 - sigmoid
            class_channel = predictions[:, 0] if pred_index == 1 else (1 - predictions[:, 0])

    grads = tape.gradient(class_channel, conv_outputs)          # Gradient w.r.t. feature maps
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # Global average pooling over spatial dims

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# ============================================================
# 5.2  OVERLAY HEATMAP ON IMAGE
# ============================================================
def overlay_gradcam(img_array, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Blend the Grad-CAM heatmap on top of the original image.
    img_array : (H, W, 3) in [0, 1]
    Returns   : (H, W, 3) uint8 blended image
    """
    img_uint8   = (img_array * 255).astype(np.uint8)
    heatmap_res = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    heatmap_col = cv2.applyColorMap(
        (heatmap_res * 255).astype(np.uint8), colormap
    )
    heatmap_rgb = cv2.cvtColor(heatmap_col, cv2.COLOR_BGR2RGB)
    superimposed = (heatmap_rgb * alpha + img_uint8 * (1 - alpha)).astype(np.uint8)
    return superimposed


# ============================================================
# 5.3  FIND LAST CONV LAYER NAME  (auto-detect)
# ============================================================
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # Handle sub-models (transfer learning backbones)
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    raise ValueError("No Conv2D layer found in model.")


# ============================================================
# 5.4  COLLECT EXAMPLE IMAGES BY ERROR TYPE
# ============================================================
def collect_examples(model, test_gen, n_each=4):
    """
    Returns three lists of (raw_img, true_label, pred_label) tuples:
      correct_preds, false_positives, false_negatives
    """
    test_gen.reset()
    correct_preds   = []
    false_positives = []
    false_negatives = []

    for batch_imgs, batch_labels in test_gen:
        probs  = model.predict(batch_imgs, verbose=0).flatten()
        preds  = (probs >= 0.5).astype(int)
        labels = batch_labels.astype(int)

        for img, true, pred in zip(batch_imgs, labels, preds):
            if true == pred and len(correct_preds) < n_each:
                correct_preds.append((img, true, pred))
            elif true == 0 and pred == 1 and len(false_positives) < n_each:
                # FP: predicted Recyclable, actually Organic
                false_positives.append((img, true, pred))
            elif true == 1 and pred == 0 and len(false_negatives) < n_each:
                # FN: predicted Organic, actually Recyclable
                false_negatives.append((img, true, pred))

        if (len(correct_preds) >= n_each and
            len(false_positives) >= n_each and
            len(false_negatives) >= n_each):
            break

    return correct_preds, false_positives, false_negatives


# ============================================================
# 5.5  VISUALISE GRAD-CAM FOR ONE ERROR TYPE
# ============================================================
def plot_gradcam_grid(examples, model, conv_layer_name,
                      title, filename, n=4):
    fig, axes = plt.subplots(2, n, figsize=(n * 3.5, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for i, (img, true, pred) in enumerate(examples[:n]):
        img_4d  = np.expand_dims(img, axis=0)
        heatmap = make_gradcam_heatmap(img_4d, model, conv_layer_name,
                                       pred_index=pred)
        overlay = overlay_gradcam(img, heatmap)

        axes[0, i].imshow(img)
        axes[0, i].set_title(
            f"True: {CLASS_NAMES[true]}\nPred: {CLASS_NAMES[pred]}",
            fontsize=8, color="green" if true == pred else "red"
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Grad-CAM", fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {filename}")


# ============================================================
# 5.6  RUN GRAD-CAM  (choose your best model here)
#      Change `target_model` to whichever performed best in Step 4
# ============================================================
target_model     = effnet_model      # ← swap to best model
target_model_name = "EfficientNetB0"

# Auto-detect last Conv2D layer
conv_layer = get_last_conv_layer_name(target_model)
print(f"Using Conv layer for Grad-CAM: {conv_layer}")

# Collect examples
correct_preds, false_positives, false_negatives = collect_examples(
    target_model, test_generator, n_each=4
)

# Plot Grad-CAM grids
plot_gradcam_grid(
    correct_preds, target_model, conv_layer,
    title=f"Grad-CAM — Correct Predictions ({target_model_name})",
    filename=f"gradcam_correct_{target_model_name}.png"
)

plot_gradcam_grid(
    false_positives, target_model, conv_layer,
    title=f"Grad-CAM — False Positives: Predicted Recyclable, True Organic ({target_model_name})",
    filename=f"gradcam_false_positive_{target_model_name}.png"
)

plot_gradcam_grid(
    false_negatives, target_model, conv_layer,
    title=f"Grad-CAM — False Negatives: Predicted Organic, True Recyclable ({target_model_name})",
    filename=f"gradcam_false_negative_{target_model_name}.png"
)

print("\n[STEP 5 COMPLETE] Grad-CAM visualisations saved.")
