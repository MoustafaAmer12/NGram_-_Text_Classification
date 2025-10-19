"""Evaluation metrics for text classification.

This module provides functions to compute various evaluation metrics
such as accuracy, precision, recall, and F1 score for text classification tasks.
"""

import numpy as np
import sklearn.metrics as sk
import pandas as pd


def confusion_matrix(y_true, y_pred, labels=None):
    """Compute the confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true (np.ndarray): True labels of the data.
        y_pred (np.ndarray): Predicted labels of the data.
        labels (list, optional): List of labels to include in the confusion matrix. Defaults to None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for true, pred in zip(y_true, y_pred):
        x = np.where(labels == true)[0][0]
        y = np.where(labels == pred)[0][0]
        cm[x, y] += 1

    return cm, labels


def accuracy_score(y_true, y_pred):
    """Compute the accuracy score.

    Args:
        y_true (np.ndarray): True labels of the data.
        y_pred (np.ndarray): Predicted labels of the data.

    Returns:
        float: Accuracy score.
    """
    cm, _ = confusion_matrix(y_true, y_pred)
    return np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0


def precision_score(y_true, y_pred):
    """Compute the precision score.

    Args:
        y_true (np.ndarray): True labels of the data.
        y_pred (np.ndarray): Predicted labels of the data.

    Returns:
        float: macro_avg Precision score.
        list: per_class Precision scores.
    """
    cm, _ = confusion_matrix(y_true, y_pred)
    per_class_precision = np.empty(cm.shape[0])

    for i in range(cm.shape[0]):
        class_i_precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
        per_class_precision[i] = class_i_precision

    macro_avg_precision = np.mean(per_class_precision)
    return macro_avg_precision, per_class_precision


def recall_score(y_true, y_pred):
    """Compute the recall score.

    Args:
        y_true (np.ndarray): True labels of the data.
        y_pred (np.ndarray): Predicted labels of the data.

    Returns:
        float: macro_avg Recall score.
        list: per_class Recall scores.
    """
    cm, _ = confusion_matrix(y_true, y_pred)
    per_class_recall = np.empty(cm.shape[0])

    for i in range(cm.shape[0]):
        class_i_recall = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        per_class_recall[i] = class_i_recall

    macro_avg_recall = np.mean(per_class_recall)
    return macro_avg_recall, per_class_recall


def f1_score(y_true, y_pred):
    """Compute the F1 score.

    Args:
        y_true (np.ndarray): True labels of the data.
        y_pred (np.ndarray): Predicted labels of the data.

    Returns:
        float: macro_avg F1 score.
        list: per_class F1 scores.
    """
    _, per_class_precision = precision_score(y_true, y_pred)
    _, per_class_recall = recall_score(y_true, y_pred)

    per_class_f1 = np.divide(
        2 * per_class_precision * per_class_recall,
        per_class_precision + per_class_recall,
        out=np.zeros_like(per_class_precision),
        where=(per_class_precision + per_class_recall) > 0,
    )

    macro_avg_f1 = np.mean(per_class_f1)

    return macro_avg_f1, per_class_f1


if __name__ == "__main__":
    # Example usage
    y_true = np.array([0, 1, 2, 2, 0, 1])
    y_pred = np.array([0, 2, 1, 2, 0, 1])

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision (macro avg):", precision_score(y_true, y_pred)[0])
    print("Recall (macro avg):", recall_score(y_true, y_pred)[0])
    print("F1 Score (macro avg):", f1_score(y_true, y_pred)[0])


def compare_metrics(y_true, y_pred, task):
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    cm, _ = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precison, _ = precision_score(y_true, y_pred)
    recall, _ = recall_score(y_true, y_pred)
    f1, _ = f1_score(y_true, y_pred)

    cm_2 = sk.confusion_matrix(y_true, y_pred)
    acc_2 = sk.accuracy_score(y_true, y_pred)
    precison_2 = sk.precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_2 = sk.recall_score(y_true, y_pred, average="macro")
    f1_2 = sk.f1_score(y_true, y_pred, average="macro")

    scalar_results = {
        "Metric": [
            "Accuracy",
            "Precision (Macro)",
            "Recall (Macro)",
            "F1-Score (Macro)",
        ],
        "Custom Implementation": [acc, precison, recall, f1],
        "Sklearn (Reference)": [acc_2, precison_2, recall_2, f1_2],
    }

    df_scalars = pd.DataFrame(scalar_results)
    df_scalars = df_scalars.set_index("Metric")

    # --------------------
    # 2. Print Results
    # --------------------

    print("\n" + "=" * 70)
    print(f"            {task} EVALUATION METRICS COMPARISON (SCALAR)")
    print("=" * 70)
    print(df_scalars.to_string(float_format="%.4f"))
    print("=" * 70 + "\n")

    # --------------------
    # 3. Print Confusion Matrices Separately
    # --------------------

    # Note: We use np.array_str for clean printing if they are numpy arrays.

    print("--- Confusion Matrix (Custom Implementation) ---")
    # Use pd.DataFrame for a nicer, labeled matrix display
    cm_df_custom = pd.DataFrame(
        cm,
        index=[f"True_{i}" for i in labels],
        columns=[f"Pred_{i}" for i in labels],
    )
    print(cm_df_custom)
    print("\n")

    print("--- Confusion Matrix (Sklearn Reference) ---")
    cm_df_sklearn = pd.DataFrame(
        cm_2,
        index=[f"True_{i}" for i in labels],
        columns=[f"Pred_{i}" for i in labels],
    )
    print(cm_df_sklearn)
    print("--------------------------------------------")
