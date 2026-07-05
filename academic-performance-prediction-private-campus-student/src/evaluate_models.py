"""
Evaluation helper functions.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def show_metrics(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision macro:", precision_score(y_true, y_pred, average="macro", zero_division=0))
    print("Recall macro:", recall_score(y_true, y_pred, average="macro", zero_division=0))
    print("F1 macro:", f1_score(y_true, y_pred, average="macro", zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
