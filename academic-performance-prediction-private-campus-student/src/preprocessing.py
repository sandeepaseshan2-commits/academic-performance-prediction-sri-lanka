"""
Preprocess the Google Form survey dataset.

Before running:
1. Put the CSV file inside data/raw/
2. Rename it to raw_survey_responses.csv
3. Run from the project root:
   python src/preprocessing.py
"""

from pathlib import Path
import re

import pandas as pd


RAW_FILE = Path("data/raw/raw_survey_responses.csv")
OUTPUT_FILE = Path("data/processed/cleaned_survey_data.csv")
REPORT_FILE = Path("data/processed/preprocessing_report.txt")

TARGET = "academic_performance_level"

# Short names for the 23 Google Form questions
QUESTION_NAMES = {
    1: "consent",
    2: "currently_studying",
    3: "institution_type",
    4: "year_of_study",
    5: "degree_area",
    6: "study_mode",
    7: "gender",
    8: "attendance",
    9: "study_hours",
    10: "lms_usage",
    11: "assignment_habit",
    12: "participation",
    13: "ca_marks_range",
    14: "sleep_hours",
    15: "motivation",
    16: "time_management",
    17: "internet_quality",
    18: "part_time_work",
    19: "academic_stress",
    20: "travel_time",
    21: "study_resources",
    22: TARGET,
    23: "academic_support",
}

# The 16 approved predictor variables
PREDICTORS = [
    "year_of_study",
    "degree_area",
    "study_mode",
    "attendance",
    "study_hours",
    "lms_usage",
    "assignment_habit",
    "participation",
    "sleep_hours",
    "motivation",
    "time_management",
    "internet_quality",
    "part_time_work",
    "academic_stress",
    "travel_time",
    "study_resources",
]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename long Google Form headings to short names."""

    rename_map = {}

    for column in df.columns:
        column_name = str(column).strip()

        if column_name.lower() == "timestamp":
            rename_map[column] = "timestamp"
            continue

        question_match = re.match(r"^(\d+)\.", column_name)

        if question_match:
            question_number = int(question_match.group(1))

            rename_map[column] = QUESTION_NAMES.get(
                question_number,
                f"question_{question_number}",
            )
        else:
            safe_name = re.sub(
                r"[^a-z0-9]+",
                "_",
                column_name.lower(),
            ).strip("_")

            rename_map[column] = safe_name

    return df.rename(columns=rename_map).copy()


def clean_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unnecessary spaces from text values."""

    cleaned_df = df.copy()

    for column in cleaned_df.select_dtypes(
        include=["object", "string"]
    ).columns:
        cleaned_df[column] = cleaned_df[column].apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    return cleaned_df


def encode_target(value):
    """Convert Low, Average and High into 0, 1 and 2."""

    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()

    if text.startswith("low"):
        return 0

    if text.startswith("average"):
        return 1

    if text.startswith("high"):
        return 2

    return pd.NA


def expected_class_from_ca(value):
    """Find the expected class using Question 13."""

    if pd.isna(value):
        return None

    text = str(value).strip().lower().replace("–", "-")

    if text == "below 40":
        return 0

    if text in {"40-54", "55-69"}:
        return 1

    if text == "70 and above":
        return 2

    return None


def check_q13_q22_consistency(df: pd.DataFrame) -> dict:
    """Compare Question 13 and Question 22."""

    result = {
        "consistent": 0,
        "inconsistent": 0,
        "not_sure": 0,
    }

    for _, row in df.iterrows():
        expected = expected_class_from_ca(
            row["ca_marks_range"]
        )

        actual = encode_target(
            row[TARGET]
        )

        if expected is None or pd.isna(actual):
            result["not_sure"] += 1

        elif expected == actual:
            result["consistent"] += 1

        else:
            result["inconsistent"] += 1

    return result


def main() -> None:

    if not RAW_FILE.exists():
        print("CSV file not found.")
        print(f"Please add: {RAW_FILE}")
        return

    try:
        # Load the original survey dataset
        df = pd.read_csv(RAW_FILE)

        # Rename columns and clean text
        df = rename_columns(df)
        df = clean_text_values(df)

        raw_rows = len(df)
        raw_columns = len(df.columns)
        raw_missing = int(df.isna().sum().sum())

        print("\nBefore cleaning")
        print("Rows:", raw_rows)
        print("Columns:", raw_columns)
        print("Missing values:", raw_missing)

        # Check duplicates while ignoring timestamp
        duplicate_columns = [
            column
            for column in df.columns
            if column != "timestamp"
        ]

        duplicate_mask = df.duplicated(
            subset=duplicate_columns,
            keep="first",
        )

        duplicate_count = int(
            duplicate_mask.sum()
        )

        df = df.loc[
            ~duplicate_mask
        ].copy()

        # Check Q13 and Q22 consistency
        consistency = check_q13_q22_consistency(df)

        # Apply consent and eligibility rules
        consent_mask = (
            df["consent"]
            .fillna("")
            .astype(str)
            .str.contains(
                "agree",
                case=False,
            )
        )

        studying_mask = (
            df["currently_studying"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq("yes")
        )

        private_mask = (
            df["institution_type"]
            .fillna("")
            .astype(str)
            .str.contains(
                "private",
                case=False,
            )
        )

        consent_count = int(
            consent_mask.sum()
        )

        studying_count = int(
            studying_mask.sum()
        )

        private_count = int(
            private_mask.sum()
        )

        # Keep only eligible private-campus students
        df = df.loc[
            consent_mask
            & studying_mask
            & private_mask
        ].copy()

        # Encode Question 22 target
        df[TARGET] = df[TARGET].apply(
            encode_target
        )

        invalid_target_count = int(
            df[TARGET].isna().sum()
        )

        # Remove records with missing or invalid targets
        df = df.dropna(
            subset=[TARGET]
        ).copy()

        df[TARGET] = df[TARGET].astype(int)

        # Check that all approved predictor columns exist
        missing_columns = [
            column
            for column in PREDICTORS + [TARGET]
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        # Keep only the approved predictors and target
        cleaned_df = df[
            PREDICTORS + [TARGET]
        ].copy()

        # Create processed-data folder
        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save cleaned dataset
        cleaned_df.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        target_distribution = (
            cleaned_df[TARGET]
            .value_counts()
            .sort_index()
        )

        # Create preprocessing report
        report_lines = [
            "PREPROCESSING REPORT",
            "=" * 40,
            "",
            f"Raw responses: {raw_rows}",
            f"Raw columns: {raw_columns}",
            f"Raw missing values: {raw_missing}",
            f"Exact duplicate responses: {duplicate_count}",
            "",
            f"Consent provided: {consent_count}",
            f"Currently studying - Yes: {studying_count}",
            f"Private institution: {private_count}",
            f"Final eligible records: {len(cleaned_df)}",
            f"Invalid target records removed: {invalid_target_count}",
            "",
            "Q13 and Q22 consistency:",
            f"Consistent: {consistency['consistent']}",
            f"Inconsistent: {consistency['inconsistent']}",
            f"Not sure or uncheckable: {consistency['not_sure']}",
            "",
            "Final target distribution:",
            "0 = Low",
            "1 = Average",
            "2 = High",
            target_distribution.to_string(),
            "",
            f"Final columns: {len(cleaned_df.columns)}",
            (
                "Remaining missing predictor values: "
                f"{int(cleaned_df[PREDICTORS].isna().sum().sum())}"
            ),
            "",
            (
                "Encoding, imputation and scaling will be "
                "performed inside the training pipeline."
            ),
        ]

        REPORT_FILE.write_text(
            "\n".join(report_lines),
            encoding="utf-8",
        )

        print("\nAfter preprocessing")
        print("Eligible modelling rows:", len(cleaned_df))
        print("Predictor columns:", len(PREDICTORS))
        print("Target column:", TARGET)

        print(
            "Remaining missing predictor values:",
            int(
                cleaned_df[PREDICTORS]
                .isna()
                .sum()
                .sum()
            ),
        )

        print(
            "\nTarget distribution "
            "(0=Low, 1=Average, 2=High):"
        )

        print(target_distribution)

        print(
            "\nSaved cleaned dataset to:",
            OUTPUT_FILE,
        )

        print(
            "Saved preprocessing report to:",
            REPORT_FILE,
        )

    except (
        OSError,
        ValueError,
        KeyError,
        pd.errors.ParserError,
    ) as error:

        print("\nPreprocessing failed.")
        print("Reason:", error)


if __name__ == "__main__":
    main()