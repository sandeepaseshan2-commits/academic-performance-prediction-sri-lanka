"""
Evaluate the three selected machine-learning models.

This script:

1. Produces out-of-fold predictions
2. Calculates overall evaluation metrics
3. Calculates class-wise recall
4. Creates confusion matrices
5. Runs the Friedman test
6. Runs Wilcoxon tests when required
7. Applies Holm correction

Run from the project root:

    python src/evaluate_models.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from statsmodels.stats.multitest import multipletests

from train_models import (
    RANDOM_STATE,
    create_models,
    load_data,
)


# ------------------------------------------------------------
# File locations
# ------------------------------------------------------------

RESULTS_FOLDER = Path("results")

FOLD_SCORES_FILE = (
    RESULTS_FOLDER / "cross_validation_scores.csv"
)

EVALUATION_FILE = (
    RESULTS_FOLDER / "evaluation_metrics.csv"
)

CLASS_RECALL_FILE = (
    RESULTS_FOLDER / "class_wise_recall.csv"
)

STATISTICAL_FILE = (
    RESULTS_FOLDER / "statistical_comparison.csv"
)

PREDICTIONS_FILE = (
    RESULTS_FOLDER / "out_of_fold_predictions.csv"
)


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
]

CLASS_LABELS = [0, 1, 2]

CLASS_NAMES = [
    "Low",
    "Average",
    "High",
]

ALPHA = 0.05


def safe_filename(model_name: str) -> str:
    """Convert a model name into a safe filename."""

    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def save_confusion_matrix(
    model_name: str,
    matrix,
) -> None:
    """Save a confusion matrix as CSV and PNG."""

    safe_name = safe_filename(model_name)

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual {name}"
            for name in CLASS_NAMES
        ],
        columns=[
            f"Predicted {name}"
            for name in CLASS_NAMES
        ],
    )

    csv_file = (
        RESULTS_FOLDER
        / f"confusion_matrix_{safe_name}.csv"
    )

    matrix_df.to_csv(csv_file)

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    image = axis.imshow(matrix)

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set_title(
        f"{model_name} Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted Class"
    )

    axis.set_ylabel(
        "Actual Class"
    )

    axis.set_xticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
    )

    axis.set_yticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.tight_layout()

    image_file = (
        RESULTS_FOLDER
        / f"confusion_matrix_{safe_name}.png"
    )

    figure.savefig(
        image_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def evaluate_predictions() -> None:
    """Generate predictions and calculate metrics."""

    X, y = load_data()

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

    model_definitions = create_models()

    evaluation_rows = []
    class_recall_rows = []
    prediction_rows = []

    print("\nGenerating out-of-fold predictions...")

    for model_name in MODEL_ORDER:

        print("\nEvaluating:", model_name)

        definition = model_definitions[
            model_name
        ]

        grid_search = GridSearchCV(
            estimator=definition["pipeline"],
            param_grid=definition["parameters"],
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        predictions = cross_val_predict(
            estimator=grid_search,
            X=X,
            y=y,
            cv=outer_cv,
            method="predict",
            n_jobs=1,
        )

        accuracy = accuracy_score(
            y,
            predictions,
        )

        macro_precision = precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_recall = recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_f1 = f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        )

        evaluation_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Macro Precision": macro_precision,
                "Macro Recall": macro_recall,
                "Macro F1": macro_f1,
            }
        )

        recalls = recall_score(
            y,
            predictions,
            labels=CLASS_LABELS,
            average=None,
            zero_division=0,
        )

        for class_name, class_recall in zip(
            CLASS_NAMES,
            recalls,
        ):
            class_recall_rows.append(
                {
                    "Model": model_name,
                    "Class": class_name,
                    "Recall": class_recall,
                }
            )

        matrix = confusion_matrix(
            y,
            predictions,
            labels=CLASS_LABELS,
        )

        save_confusion_matrix(
            model_name,
            matrix,
        )

        for record_number, (
            actual,
            predicted,
        ) in enumerate(
            zip(y, predictions),
            start=1,
        ):
            prediction_rows.append(
                {
                    "Model": model_name,
                    "Record": record_number,
                    "Actual": actual,
                    "Predicted": predicted,
                }
            )

        print(
            "Accuracy:",
            round(accuracy, 4),
        )

        print(
            "Macro Precision:",
            round(macro_precision, 4),
        )

        print(
            "Macro Recall:",
            round(macro_recall, 4),
        )

        print(
            "Macro F1:",
            round(macro_f1, 4),
        )

        print(
            "Class-wise recall:",
            {
                class_name: round(
                    float(class_recall),
                    4,
                )
                for class_name, class_recall
                in zip(CLASS_NAMES, recalls)
            },
        )

    pd.DataFrame(
        evaluation_rows
    ).to_csv(
        EVALUATION_FILE,
        index=False,
    )

    pd.DataFrame(
        class_recall_rows
    ).to_csv(
        CLASS_RECALL_FILE,
        index=False,
    )

    pd.DataFrame(
        prediction_rows
    ).to_csv(
        PREDICTIONS_FILE,
        index=False,
    )


def safe_wilcoxon(
    first_scores,
    second_scores,
):
    """Run Wilcoxon safely when scores are identical."""

    try:
        statistic, p_value = wilcoxon(
            first_scores,
            second_scores,
            alternative="two-sided",
        )

    except ValueError:
        statistic = 0.0
        p_value = 1.0

    return float(statistic), float(p_value)


def run_statistical_tests() -> None:
    """Run Friedman, Wilcoxon and Holm tests."""

    if not FOLD_SCORES_FILE.exists():
        raise FileNotFoundError(
            "cross_validation_scores.csv was not found. "
            "Run train_models.py first."
        )

    scores_df = pd.read_csv(
        FOLD_SCORES_FILE
    )

    required_columns = {
        "Model",
        "Fold",
        "Macro F1",
    }

    missing_columns = (
        required_columns
        - set(scores_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "The fold-score file is missing: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    score_table = scores_df.pivot(
        index="Fold",
        columns="Model",
        values="Macro F1",
    )

    missing_models = [
        model
        for model in MODEL_ORDER
        if model not in score_table.columns
    ]

    if missing_models:
        raise ValueError(
            "Missing model scores: "
            + ", ".join(missing_models)
        )

    score_table = score_table[
        MODEL_ORDER
    ].dropna()

    friedman_statistic, friedman_p = (
        friedmanchisquare(
            score_table[
                "Logistic Regression"
            ],
            score_table[
                "Decision Tree"
            ],
            score_table[
                "Random Forest"
            ],
        )
    )

    statistical_rows = [
        {
            "Comparison": "All three models",
            "Statistical Test": "Friedman test",
            "Statistic": friedman_statistic,
            "Raw p-value": friedman_p,
            "Adjusted p-value": friedman_p,
            "Decision": (
                "Reject the null hypothesis"
                if friedman_p < ALPHA
                else "Fail to reject the null hypothesis"
            ),
        }
    ]

    print("\nStatistical comparison")
    print(
        "Friedman statistic:",
        round(
            float(friedman_statistic),
            4,
        ),
    )

    print(
        "Friedman p-value:",
        round(
            float(friedman_p),
            4,
        ),
    )

    pairwise_comparisons = [
        (
            "Logistic Regression",
            "Decision Tree",
        ),
        (
            "Logistic Regression",
            "Random Forest",
        ),
        (
            "Decision Tree",
            "Random Forest",
        ),
    ]

    if friedman_p < ALPHA:

        raw_p_values = []
        pairwise_results = []

        for first_model, second_model in (
            pairwise_comparisons
        ):

            statistic, raw_p_value = (
                safe_wilcoxon(
                    score_table[first_model],
                    score_table[second_model],
                )
            )

            raw_p_values.append(
                raw_p_value
            )

            pairwise_results.append(
                {
                    "first_model": first_model,
                    "second_model": second_model,
                    "statistic": statistic,
                    "raw_p_value": raw_p_value,
                }
            )

        rejected, adjusted_p_values, _, _ = (
            multipletests(
                raw_p_values,
                alpha=ALPHA,
                method="holm",
            )
        )

        for result, adjusted_p, reject in zip(
            pairwise_results,
            adjusted_p_values,
            rejected,
        ):

            statistical_rows.append(
                {
                    "Comparison": (
                        f"{result['first_model']} vs "
                        f"{result['second_model']}"
                    ),
                    "Statistical Test": (
                        "Wilcoxon signed-rank test "
                        "with Holm correction"
                    ),
                    "Statistic": result[
                        "statistic"
                    ],
                    "Raw p-value": result[
                        "raw_p_value"
                    ],
                    "Adjusted p-value": adjusted_p,
                    "Decision": (
                        "Statistically significant"
                        if reject
                        else "Not statistically significant"
                    ),
                }
            )

    else:

        for first_model, second_model in (
            pairwise_comparisons
        ):

            statistical_rows.append(
                {
                    "Comparison": (
                        f"{first_model} vs "
                        f"{second_model}"
                    ),
                    "Statistical Test": (
                        "Wilcoxon test not performed"
                    ),
                    "Statistic": None,
                    "Raw p-value": None,
                    "Adjusted p-value": None,
                    "Decision": (
                        "Friedman test was not significant"
                    ),
                }
            )

    statistical_df = pd.DataFrame(
        statistical_rows
    )

    statistical_df.to_csv(
        STATISTICAL_FILE,
        index=False,
    )


def main() -> None:
    """Run the complete evaluation process."""

    try:
        RESULTS_FOLDER.mkdir(
            parents=True,
            exist_ok=True,
        )

        evaluate_predictions()
        run_statistical_tests()

        print("\nEvaluation completed successfully.")

        print("\nSaved overall metrics to:")
        print(EVALUATION_FILE)

        print("\nSaved class-wise recall to:")
        print(CLASS_RECALL_FILE)

        print("\nSaved statistical results to:")
        print(STATISTICAL_FILE)

        print("\nSaved out-of-fold predictions to:")
        print(PREDICTIONS_FILE)

        print(
            "\nConfusion matrix CSV and PNG files "
            "were saved inside the results folder."
        )

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        OSError,
        pd.errors.ParserError,
    ) as error:

        print("\nEvaluation failed.")
        print("Reason:", error)


if __name__ == "__main__":
    main()