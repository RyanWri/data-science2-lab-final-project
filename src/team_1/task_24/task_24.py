import json
from typing import List

import pandas as pd


def normalize_date_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    return pd.to_datetime(df[column_name]).dt.date


def rename_columns_from_file(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    # Reading columns from the JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        column_names = json.load(f)

    return df.rename(columns=column_names)


def merge_dataframes_left_join(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left_table_columns: List[str],
    right_table_columns: List[str],
) -> pd.DataFrame:
    for column in left_table_columns:
        assert (
            column in df1.columns
        ), f"Column '{column}' not found in the left dataframe."
    for column in right_table_columns:
        assert (
            column in df2.columns
        ), f"Column '{column}' not found in the right dataframe."

    return pd.merge(
        df1, df2, how="left", left_on=left_table_columns, right_on=right_table_columns
    )


def append_rehospitalized_status_to_patients(df, column_name="Patient"):
    """
    append to each patient 2 columns: 1) if they were rehospitalized 2) how many times they were rehospitalized
    Parameters:
    df (pd.DataFrame): DataFrame containing the column "Patient"
    Returns:
    pd.DataFrame: A DataFrame with the added columns
    """
    # Count the number of hospitalizations per patient in each department
    df["rehospitalization_count"] = df.groupby(column_name)[column_name].transform(
        "count"
    )
    # Mark patients with more than one hospitalization as repeated
    df["is_rehospitalization"] = df["rehospitalization_count"] > 1

    return df
