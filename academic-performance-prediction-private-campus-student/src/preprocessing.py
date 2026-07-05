"""
Basic preprocessing file for the survey dataset.

Before running:
1. Put the Google Form CSV file inside data/raw/
2. Rename it as raw_survey_responses.csv
3. Run this file from the project root folder
"""

from pathlib import Path
import pandas as pd

RAW_FILE = Path("data/raw/raw_survey_responses.csv")
OUTPUT_FILE = Path("data/processed/cleaned_survey_data.csv")


def clean_column_names(df):
    # Make column names easier to use in Python
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("-", "_")
    )
    return df


def remove_unwanted_columns(df):
    # These fields are not useful for prediction
    remove_words = [
        "timestamp",
        "consent",
        "agree",
        "currently_studying",
        "institution_type"
    ]

    columns_to_remove = []

    for col in df.columns:
        for word in remove_words:
            if word in col:
                columns_to_remove.append(col)

    return df.drop(columns=list(set(columns_to_remove)), errors="ignore")


def show_basic_info(df, title):
    print("\n" + title)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())


def encode_categorical_data(df):
    # Change this target column name if needed after checking the real CSV columns
    target_col = "academic_performance_level"

    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        X = pd.get_dummies(X)
        X[target_col] = y
        return X

    # If target name is different, this still saves an encoded dataset
    print("Target column name not found. Please check the column names.")
    return pd.get_dummies(df)


def main():
    if not RAW_FILE.exists():
        print("CSV file not found.")
        print("Please add data/raw/raw_survey_responses.csv")
        return

    df = pd.read_csv(RAW_FILE)
    df = clean_column_names(df)

    show_basic_info(df, "Before cleaning")

    df = df.drop_duplicates()
    df = remove_unwanted_columns(df)
    df_encoded = encode_categorical_data(df)

    show_basic_info(df_encoded, "After preprocessing")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_encoded.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved cleaned file to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
