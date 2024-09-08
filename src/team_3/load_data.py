import os
import pandas as pd

# Construct the file path
file_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'rehospitalization.xlsx')


# Function to load all sheets from an Excel file as separate DataFrames
def load_excel_sheets(file_path):
    # Load the Excel file
    excel_file = pd.ExcelFile(file_path)

    # Create a dictionary to store DataFrames, with sheet names as keys
    dataframes = {}

    # Loop through each sheet and load it as a DataFrame
    for sheet_name in excel_file.sheet_names:
        dataframes[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Loaded sheet: {sheet_name}")

    return dataframes


# Example usage:
dataframes = load_excel_sheets(file_path)

# Access individual dataframes by sheet name, for example:
# df_general_data = dataframes["GeneralData"]
