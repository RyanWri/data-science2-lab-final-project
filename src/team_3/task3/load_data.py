import os
import pandas as pd


# Function to load all sheets from an Excel file as separate DataFrames
def load_excel_sheets(sheet_name):
    # Construct the file path (modify based on your actual file path)
    file_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'rehospitalization.xlsx')

    # Load the Excel file
    excel_file = pd.ExcelFile(file_path)

    # Create a dataframe for a specific sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Loaded sheet: {sheet_name}")

    return df
