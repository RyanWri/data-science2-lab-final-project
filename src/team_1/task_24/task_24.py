import pandas as pd
import json


def normalize_date_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    return pd.to_datetime(df[column_name]).dt.date


def rename_columns_from_file(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    # Reading columns from the JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        column_names = json.load(f)

    return df.rename(columns=column_names)


def merge_dataframes_on_column(
    df1: pd.DataFrame, df2: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    assert (
        column_name in df1.columns and column_name in df2.columns
    ), f"Column '{column_name}' not found in at least one of the dataframes."

    return pd.merge(df1, df2, on=column_name)
