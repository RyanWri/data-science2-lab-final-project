# use_sheet1.py
from load_data import get_sheet1_df

# Specify the path to your Excel file
excel_file_path = 'your_excel_file.xlsx'

# Get the DataFrame for 'Sheet1'
sheet1_df = get_sheet1_df(excel_file_path)

# Now you can use sheet1_df in your code
if sheet1_df is not None:
    print(sheet1_df.head())  # For example, print the first few rows
else:
    print("Sheet1 not found in the Excel file.")