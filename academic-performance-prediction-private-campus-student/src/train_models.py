"""
Train and compare Logistic Regression, Decision Tree and Random Forest.

Run this file from the project root folder:

    python src/train_models.py

Before running, complete preprocessing:

    python src/preprocessing.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier


# ------------------------------------------------------------
# File locations
# ------------------------------------------------------------

DATA_FILE = Path(
    "data/processed/cleaned_survey_data.csv"
)

RESULTS_FOLDER = Path("results")

FOLD_RESULTS_FILE = RESULTS_FOLDER / "cross_validation_scores.csv"
SUMMARY_FILE = RESULTS_FOLDER / "model_summary.csv"
PARAMETERS_FILE = RESULTS_FOLDER / "best_parameters.json"

TARGET_COLUMN = "academic_performance_level"

RANDOM_STATE = 42


# ------------------------------------------------------------
# Predictor groups
# ------------------------------------------------------------

NOMINAL_FEATURES = [
    "degree_area",
    "study_mode",
    "part_time_work",
]

ORDINAL_FEATURES = [
    "year_of_study",
    "attendance",
    "study_hours",
    "lms_usage",
    "assignment_habit",
    "participation",
    "sleep_hours",
    "motivation",
    "time_management",
    "internet_quality",
    "academic_stress",
    "travel_time",
    "study_resources",
]

ALL_FEATURES = NOMINAL_FEATURES + ORDINAL_FEATURES


# ------------------------------------------------------------
# Ordered categories
# The order must match the Google Form answer choices.
# ------------------------------------------------------------

ORDINAL_CATEGORIES = [
    # year_of_study
    [
        "Year 1",
        "Year 2",
        "Year 3",
        "Year 4",
    ],

    # attendance
    [
        "Less than 50%",
        "50%–69%",
        "70%–84%",
        "85% and above",
    ],

    # study_hours
    [
        "Less than 1 hour",
        "1–2 hours",
        "3–4 hours",
        "More than 4 hours",
    ],

    # lms_usage
    [
        "Rarely",
        "Sometimes",
        "Often",
        "Very often",
    ],

    # assignment_habit
    [
        "Usually late",
        "Sometimes late",
        "Usually on time",
        "Always on time",
    ],

    # participation
    [
        "Low",
        "Medium",
        "High",
    ],

    # sleep_hours
    [
        "Less than 5 hours",
        "5–6 hours",
        "7–8 hours",
        "More than 8 hours",
    ],

    # motivation
    [
        "Low",
        "Medium",
        "High",
    ],

    # time_management
    [
        "Poor",
        "Average",
        "Good",
    ],

    # internet_quality
    [
        "Poor",
        "Average",
        "Good",
        "Very good",
    ],

    # academic_stress
    [
        "Low",
        "Medium",
        "High",
    ],

    # travel_time
    [
        "Less than 30 minutes",
        "30 minutes–1 hour",
        "1–2 hours",
        "More than 2 hours",
    ],

    # study_resources
    [
        "Poor",
        "Average",
        "Good",
        "Very good",
    ],
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the cleaned survey dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. "
            "Run python src/preprocessing.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = ALL_FEATURES + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The cleaned dataset is missing these columns: "
            + ", ".join(missing_columns)
        )

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    if y.isna().any():
        raise ValueError(
            "The target column contains missing values."
        )

    y = y.astype(int)

    valid_labels = {0, 1, 2}

    if not set(y.unique()).issubset(valid_labels):
        raise ValueError(
            "The target must contain only 0, 1 and 2."
        )

    return X, y


def create_preprocessor() -> ColumnTransformer:
    """Create preprocessing steps for the predictor variables."""

    nominal_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "nominal",
                nominal_pipeline,
                NOMINAL_FEATURES,
            ),
            (
                "ordinal",
                ordinal_pipeline,
                ORDINAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def create_models() -> dict[str, dict[str, Any]]:
    """Create model pipelines and small parameter grids."""

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "scaler",
                StandardScaler(
                    with_mean=False
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    decision_tree_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": {
            "pipeline": logistic_pipeline,
            "parameters": {
                "model__C": [
                    0.1,
                    1,
                    10,
                ],
            },
        },

        "Decision Tree": {
            "pipeline": decision_tree_pipeline,
            "parameters": {
                "model__max_depth": [
                    3,
                    5,
                    8,
                    None,
                ],
                "model__min_samples_leaf": [
                    1,
                    3,
                    5,
                ],
            },
        },

        "Random Forest": {
            "pipeline": random_forest_pipeline,
            "parameters": {
                "model__n_estimators": [
                    100,
                    200,
                ],
                "model__max_depth": [
                    5,
                    10,
                    None,
                ],
                "model__min_samples_leaf": [
                    1,
                    2,
                    4,
                ],
            },
        },
    }


def create_scoring() -> dict[str, Any]:
    """Create evaluation metrics."""

    return {
        "accuracy": "accuracy",

        "precision_macro": make_scorer(
            precision_score,
            average="macro",
            zero_division=0,
        ),

        "recall_macro": make_scorer(
            recall_score,
            average="macro",
            zero_division=0,
        ),

        "f1_macro": make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
    }


def convert_json_value(value: Any) -> Any:
    """Convert NumPy values into JSON-compatible values."""

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def main() -> None:
    """Run nested cross-validation for all three models."""

    try:
        X, y = load_data()

        class_distribution = (
            y.value_counts()
            .sort_index()
        )

        print("\nDataset loaded successfully")
        print("Rows:", len(X))
        print("Predictors:", len(X.columns))

        print(
            "\nTarget distribution "
            "(0=Low, 1=Average, 2=High):"
        )
        print(class_distribution)

        smallest_class = int(
            class_distribution.min()
        )

        if smallest_class < 5:
            raise ValueError(
                "The smallest class has fewer than five records. "
                "Stratified 5-Fold Cross-Validation cannot be used."
            )

        outer_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        scoring = create_scoring()
        model_definitions = create_models()

        all_fold_results: list[dict[str, Any]] = []
        all_summaries: list[dict[str, Any]] = []
        all_best_parameters: dict[str, list[dict[str, Any]]] = {}

        RESULTS_FOLDER.mkdir(
            parents=True,
            exist_ok=True,
        )

        for model_name, definition in model_definitions.items():

            print("\n" + "=" * 60)
            print("Training:", model_name)
            print("=" * 60)

            grid_search = GridSearchCV(
                estimator=definition["pipeline"],
                param_grid=definition["parameters"],
                scoring="f1_macro",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )

            cv_results = cross_validate(
                estimator=grid_search,
                X=X,
                y=y,
                cv=outer_cv,
                scoring=scoring,
                return_estimator=True,
                n_jobs=1,
                error_score="raise",
            )

            fold_best_parameters = []

            for fold_number in range(5):

                fold_result = {
                    "Model": model_name,
                    "Fold": fold_number + 1,
                    "Accuracy": cv_results[
                        "test_accuracy"
                    ][fold_number],
                    "Macro Precision": cv_results[
                        "test_precision_macro"
                    ][fold_number],
                    "Macro Recall": cv_results[
                        "test_recall_macro"
                    ][fold_number],
                    "Macro F1": cv_results[
                        "test_f1_macro"
                    ][fold_number],
                }

                all_fold_results.append(fold_result)

                fitted_search = cv_results[
                    "estimator"
                ][fold_number]

                best_params = {
                    key: convert_json_value(value)
                    for key, value
                    in fitted_search.best_params_.items()
                }

                fold_best_parameters.append(
                    {
                        "fold": fold_number + 1,
                        "parameters": best_params,
                    }
                )

            all_best_parameters[
                model_name
            ] = fold_best_parameters

            metric_columns = {
                "Accuracy": "test_accuracy",
                "Macro Precision": "test_precision_macro",
                "Macro Recall": "test_recall_macro",
                "Macro F1": "test_f1_macro",
            }

            summary = {
                "Model": model_name,
            }

            for display_name, result_key in metric_columns.items():

                values = cv_results[result_key]

                summary[
                    f"{display_name} Mean"
                ] = float(np.mean(values))

                summary[
                    f"{display_name} SD"
                ] = float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )

            all_summaries.append(summary)

            print(
                "Accuracy:",
                f"{summary['Accuracy Mean']:.4f}",
                "±",
                f"{summary['Accuracy SD']:.4f}",
            )

            print(
                "Macro Precision:",
                f"{summary['Macro Precision Mean']:.4f}",
                "±",
                f"{summary['Macro Precision SD']:.4f}",
            )

            print(
                "Macro Recall:",
                f"{summary['Macro Recall Mean']:.4f}",
                "±",
                f"{summary['Macro Recall SD']:.4f}",
            )

            print(
                "Macro F1:",
                f"{summary['Macro F1 Mean']:.4f}",
                "±",
                f"{summary['Macro F1 SD']:.4f}",
            )

        fold_results_df = pd.DataFrame(
            all_fold_results
        )

        summary_df = pd.DataFrame(
            all_summaries
        )

        fold_results_df.to_csv(
            FOLD_RESULTS_FILE,
            index=False,
        )

        summary_df.to_csv(
            SUMMARY_FILE,
            index=False,
        )

        PARAMETERS_FILE.write_text(
            json.dumps(
                all_best_parameters,
                indent=4,
            ),
            encoding="utf-8",
        )

        best_model_row = summary_df.loc[
            summary_df["Macro F1 Mean"].idxmax()
        ]

        print("\n" + "=" * 60)
        print("Training completed successfully")
        print("=" * 60)

        print(
            "\nBest model based on mean Macro F1:",
            best_model_row["Model"],
        )

        print(
            "Mean Macro F1:",
            round(
                float(
                    best_model_row[
                        "Macro F1 Mean"
                    ]
                ),
                4,
            ),
        )

        print("\nSaved fold scores to:")
        print(FOLD_RESULTS_FILE)

        print("\nSaved model summary to:")
        print(SUMMARY_FILE)

        print("\nSaved selected parameters to:")
        print(PARAMETERS_FILE)

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        OSError,
        pd.errors.ParserError,
    ) as error:

        print("\nModel training failed.")
        print("Reason:", error)


if __name__ == "__main__":
    main()