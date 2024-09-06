import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_excel_sheet(file_path, sheet_name):
    """
    Reads a specific sheet from an Excel file.

    Parameters:
    file_path (str): The path to the Excel file.
    sheet_name (str): The name of the sheet to be read.

    Returns:
    pd.DataFrame: The data from the specified sheet as a pandas DataFrame.
    """
    try:
        # Read the specified sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None


def label_encode(df, columns):
    """
    Label encode specified columns in the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing the data.
    columns : list
        List of column names that need to be label encoded.

    Returns:
    --------
    pd.DataFrame
        DataFrame with label encoded columns.
    """
    label_encoder = LabelEncoder()

    # Loop through the columns and apply label encoding
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])

    return df
