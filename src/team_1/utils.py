import pandas as pd


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
