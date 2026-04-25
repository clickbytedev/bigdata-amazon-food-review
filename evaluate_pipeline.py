"""
evaluate_pipeline.py
--------------------
Evaluates the MiniLM + DeBERTa food-safety detection pipeline against a
labelled gold dataset.

Expected inputs
---------------
df_ground_truth  : DataFrame with columns [Id, ProductId, Text, label, human_is_hazard]
df_predictions   : DataFrame with columns [Id, pred_hazard_boolean, pred_label]

Run the two sections at the bottom to produce all metrics.
"""

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 0.  LOAD / INJECT YOUR DataFrames HERE
#     Replace the placeholders with your actual DataFrames before running.
# ─────────────────────────────────────────────────────────────────────────────
# df_ground_truth = pd.read_csv('outputs/sample_with_labels.csv')
# df_predictions  = <your predictions DataFrame>


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MERGE ground truth + predictions on Id
# ─────────────────────────────────────────────────────────────────────────────
def merge_results(df_ground_truth: pd.DataFrame,
                  df_predictions:  pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join on Id so only reviews present in both DataFrames are evaluated.
    Warns if any Ids are dropped.
    """
    df = df_ground_truth.merge(df_predictions, on='Id', how='inner')

    dropped = len(df_ground_truth) - len(df)
    if dropped:
        print(f"[WARN] {dropped} rows dropped after merge "
              f"(Id not found in predictions).")
    print(f"Merged dataset : {len(df):,} rows\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BINARY EVALUATION  (human_is_hazard vs pred_hazard_boolean)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates binary hazard detection and prints all key metrics.
    Returns a DataFrame of False Negatives for further inspection.
    """
    y_true = df['human_is_hazard'].astype(int)
    y_pred = df['pred_hazard_boolean'].astype(int)

    # --- Confusion Matrix ---------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("=" * 60)
    print("BINARY CLASSIFICATION  (hazard = 1, safe = 0)")
    print("=" * 60)
    print(f"\nConfusion Matrix:\n"
          f"                  Pred Safe  Pred Hazard\n"
          f"  Actual Safe       {tn:>6}      {fp:>6}   (TN / FP)\n"
          f"  Actual Hazard     {fn:>6}      {tp:>6}   (FN / TP)\n")

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Safe (0)', 'Hazard (1)'])
    disp.plot(cmap='Blues', colorbar=False)
    plt.title('Binary Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix_binary.png', dpi=150)
    plt.show()
    print("Plot saved → outputs/confusion_matrix_binary.png\n")

    # --- Precision / Recall / F1 --------------------------------------------
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score   (y_true, y_pred, zero_division=0)
    f1        = f1_score       (y_true, y_pred, zero_division=0)

    print(f"Precision : {precision:.4f}   (of all predicted hazards, how many are real)")
    print(f"Recall    : {recall:.4f}   (of all real hazards, how many were caught)")
    print(f"F1-Score  : {f1:.4f}   (harmonic mean of precision and recall)")
    print()

    # --- False Negatives (critical failure) ---------------------------------
    #
    # A False Negative means the model said "Safe" but the review was actually
    # a hazard.  In a food-safety context this is the worst error type:
    #   • A dangerous product stays undetected.
    #   • The higher the FN count, the lower the Recall.
    #
    # Inspect FNs to understand *why* the model missed them:
    #   - Were they subtly written?
    #   - Did they belong to a rare category (allergen, contamination)?
    #   - Did late-weighting over-suppress a valid 4-star warning?
    #
    fn_mask = (y_true == 1) & (y_pred == 0)
    df_fn   = df[fn_mask].copy()

    print(f"False Negatives : {fn:,}  "
          f"({fn / max(int(y_true.sum()), 1):.1%} of all actual hazards missed)")
    print(f"\nFalse Negative breakdown by true label:")
    if 'label' in df_fn.columns:
        print(df_fn['label'].value_counts().to_string())
    print()

    # Save FNs for manual inspection
    fn_path = 'outputs/false_negatives.csv'
    df_fn[['Id', 'label', 'Text']].to_csv(fn_path, index=False)
    print(f"False Negatives saved → {fn_path}\n")

    return df_fn


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MULTI-CLASS EVALUATION  (label vs pred_label)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_multiclass(df: pd.DataFrame) -> None:
    """
    Prints a full per-class classification report.

    zero_division=0 prevents warnings for classes with no predicted samples
    (e.g. 'allergen' has only 10 gold examples and may never be predicted).
    The affected classes will show 0.00 for precision/recall/F1.
    """
    y_true = df['label']
    y_pred = df['pred_label']

    # Consistent label ordering for readability
    label_order = ['safe', 'quality_defect', 'illness', 'spoilage',
                   'contamination', 'allergen']
    # Keep only labels that actually appear in the gold set
    present_labels = [l for l in label_order if l in y_true.values]

    print("=" * 60)
    print("MULTI-CLASS CLASSIFICATION  (label vs pred_label)")
    print("=" * 60)
    print(classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        zero_division=0,        # silences UndefinedMetricWarning for rare classes
    ))

    # ---- Interpretation note ------------------------------------------------
    print("Notes on rare classes:")
    print("  • 'allergen'     (10 samples) — very few examples; low support means")
    print("    metrics are unreliable. A single missed sample heavily impacts recall.")
    print("  • 'contamination'(12 samples) — similar caveat.")
    print("  • Focus on 'macro avg' to treat every class equally, or")
    print("    'weighted avg' to reflect the class-imbalance of the dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)

    # ── Replace these two lines with your actual DataFrames ───────────────
    df_ground_truth = pd.read_csv('outputs/sample_with_labels.csv')
    # df_predictions  = pd.DataFrame(...)   # supply your predictions here
    raise SystemExit(
        "\n[ACTION REQUIRED]  Set df_predictions before running.\n"
        "  df_predictions must have columns: Id, pred_hazard_boolean, pred_label"
    )
    # ──────────────────────────────────────────────────────────────────────

    df_merged = merge_results(df_ground_truth, df_predictions)
    df_fn     = evaluate_binary(df_merged)
    evaluate_multiclass(df_merged)
