"""
This script is an example of how to use the load_data.py code and load a DataFrame for your own task.
In this file, you need to set the sheet_name argument from the options below as a string to fit your purpose.
"""

from load_data import load_excel_sheets

# Option for sheet_name, parameters to run the code with from the xlsx file
sheet_name = "hospitalization2"
"""
GeneralData
Drugs
hospitalization1
unitsAdmissions
unitsOccupancyRate
erAdmission
erDoctor
hDoctor
erBeforeHospitalization2
hospitalization2
ICD9
רופאים מאשפזים מהמלרד
רופאים משחררים מהאשפוז
טבלאות
"""

# Load the dataframes from the first script
dataframe = load_excel_sheets(sheet_name)
# Example operation on the DataFrame
print(dataframe.head())  # Show the first few rows
