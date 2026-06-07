# ============================================================
# STEP 4: EVALUATION METRICS
# Accuracy | Precision | Recall | F1 | Confusion Matrix |
# ROC-AUC  | Inference Time
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, f1_score
)

# ── Class names (from Step 1) ─────────────────────────────
CLASS_NAMES = ["Organic (O)", "Recyclable (R)"]

# ============================================================
# 4.1  EVALUATE A SINGLE MODEL
# ============================================================
def evaluate_model(model, test_gen, model_name):
    """
    Returns a dict of all metrics for the given model.
    Also saves confusion matrix and ROC curve plots.
    """
    test_gen.reset()
    true_labels = test_gen.classes

    # ── Inference time ───────────────────────────────────────
    start = time.time()
    y_prob = model.predict(test_gen, verbose=0).flatten()
    elapsed = time.time() - start
    inference_ms = (elapsed / len(true_labels)) * 1000   # ms per image

    y_pred = (y_prob >= 0.5).astype(int)

    # ── Core metrics ─────────────────────────────────────────
    report = classification_report(
        true_labels, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    macro = report["macro avg"]
    fpr, tpr, _ = roc_curve(true_labels, y_prob)
    roc_auc      = auc(fpr, tpr)

    results = {
        "model_name"    : model_name,
        "accuracy"      : report["accuracy"],
        "precision"     : macro["precision"],
        "recall"        : macro["recall"],
        "f1_score"      : macro["f1-score"],
        "roc_auc"       : roc_auc,
        "inference_ms"  : inference_ms,
        "y_prob"        : y_prob,
        "y_pred"        : y_pred,
        "fpr"           : fpr,
        "tpr"           : tpr,
        "true_labels"   : true_labels,
    }

    # ── Print summary ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*55}")
    print(classification_report(true_labels, y_pred, target_names=CLASS_NAMES))
    print(f"  ROC-AUC          : {roc_auc:.4f}")
    print(f"  Inference time   : {inference_ms:.3f} ms / image")

    # ── Confusion matrix plot ────────────────────────────────
    cm = confusion_matrix(true_labels, y_pred)
    plot_confusion_matrix(cm, model_name)

    return results


# ============================================================
# 4.2  CONFUSION MATRIX PLOT
# ============================================================
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor="gray"
    )
    plt.title(f"Confusion Matrix — {model_name}", fontweight="bold", fontsize=12)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name}.png", dpi=150)
    plt.show()


# ============================================================
# 4.3  COMBINED ROC CURVE  (all models on one plot)
# ============================================================
def plot_combined_roc(all_results):
    plt.figure(figsize=(8, 6))
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    for res, color in zip(all_results, colors):
        plt.plot(res["fpr"], res["tpr"], color=color, linewidth=2,
                 label=f'{res["model_name"]} (AUC = {res["roc_auc"]:.4f})')

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves_all_models.png", dpi=150)
    plt.show()


# ============================================================
# 4.4  COMPARISON TABLE
# ============================================================
def plot_comparison_table(all_results):
    import pandas as pd

    rows = []
    for r in all_results:
        rows.append({
            "Model"         : r["model_name"],
            "Accuracy"      : f"{r['accuracy']:.4f}",
            "Precision"     : f"{r['precision']:.4f}",
            "Recall"        : f"{r['recall']:.4f}",
            "F1-Score"      : f"{r['f1_score']:.4f}",
            "ROC-AUC"       : f"{r['roc_auc']:.4f}",
            "Infer (ms/img)": f"{r['inference_ms']:.3f}",
        })

    df = pd.DataFrame(rows)
    print("\n" + "="*65)
    print("BENCHMARKING SUMMARY TABLE")
    print("="*65)
    print(df.to_string(index=False))

    # Visual table
    fig, ax = plt.subplots(figsize=(12, len(rows) * 0.7 + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    plt.title("Model Comparison", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("model_comparison_table.png", dpi=150, bbox_inches="tight")
    plt.show()

    return df


# ============================================================
# 4.5  RUN EVALUATION ON ALL 4 MODELS
#      (assumes models and test_generator from Steps 2 & 3)
# ============================================================

all_results = []

# Custom CNN
results_cnn = evaluate_model(custom_cnn,      test_generator, "Custom CNN")
all_results.append(results_cnn)

# MobileNetV2
results_mb  = evaluate_model(mobilenet_model, test_generator, "MobileNetV2")
all_results.append(results_mb)

# ResNet50
results_rn  = evaluate_model(resnet_model,    test_generator, "ResNet50")
all_results.append(results_rn)

# EfficientNetB0
results_en  = evaluate_model(effnet_model,    test_generator, "EfficientNetB0")
all_results.append(results_en)

# Combined ROC + Summary Table
plot_combined_roc(all_results)
comparison_df = plot_comparison_table(all_results)

print("\n[STEP 4 COMPLETE] All evaluation metrics computed and saved.")
