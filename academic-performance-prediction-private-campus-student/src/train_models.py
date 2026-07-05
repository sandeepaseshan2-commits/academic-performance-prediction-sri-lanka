"""
Train the three selected models.

This is a simple first version for the Milestone 2 repository.
"""

from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = Path("data/processed/cleaned_survey_data.csv")
TARGET_COLUMN = "academic_performance_level"


def load_data():
    df = pd.read_csv(DATA_FILE)

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column not found. Check the column name in the dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def main():
    if not DATA_FILE.exists():
        print("Cleaned dataset not found. Run preprocessing.py first.")
        return

    X, y = load_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro"
    }

    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

        print("\nModel:", name)
        print("Accuracy:", round(scores["test_accuracy"].mean(), 4))
        print("Precision:", round(scores["test_precision_macro"].mean(), 4))
        print("Recall:", round(scores["test_recall_macro"].mean(), 4))
        print("F1-score:", round(scores["test_f1_macro"].mean(), 4))


if __name__ == "__main__":
    main()
