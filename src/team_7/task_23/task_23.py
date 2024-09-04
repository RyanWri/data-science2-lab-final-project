# %% 
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout

##########################################
def normalize_and_sort_by_admission_date(df1, df2):
    """
    Normalize the date column and then sort the provided DataFrames by the given date column.
    
    Parameters:
    df1 (pd.DataFrame): The first DataFrame to normalize and sort.
    df2 (pd.DataFrame): The second DataFrame to normalize and sort.
    
    Returns:
    pd.DataFrame, pd.DataFrame: The normalized and sorted DataFrames.
    """
    # Normalize the date columns in both DataFrames
    df1["Admission_Entry_Date"] = pd.to_datetime(df1["Admission_Entry_Date"]).dt.date
    df2["Admission_Entry_Date"] = pd.to_datetime(df2["Admission_Entry_Date"]).dt.date
    
    # Sort both DataFrames by the normalized date column
    sorted_df1 = df1.sort_values(by="Admission_Entry_Date")
    sorted_df2 = df2.sort_values(by="Admission_Entry_Date")
    
    return sorted_df1, sorted_df2


##########################################
# Load the specific sheets to understand their content
file_path = "F:\\לימודים\\תואר שני\\סמסטר ב\\Data Science 2\\DS2-Final Project\\rehospitalization.xlsx"
units_occupancy_rate = pd.read_excel(file_path, sheet_name='unitsOccupancyRate')
hospitalization1 = pd.read_excel(file_path, sheet_name='hospitalization1')
hospitalization2 = pd.read_excel(file_path, sheet_name='hospitalization2')


# %% 

# Translate column names to English
column_translation = {
    "תאריך": "Date",
    "מחלקה": "Department",
    "כמות שוהים": "Occupancy",
    "שיעור תפוסה": "Occupancy Rate",
    # Add all other columns that need to be translated
}

units_occupancy_rate.rename(columns=column_translation, inplace=True)
units_occupancy_rate["Date"] = pd.to_datetime(units_occupancy_rate["Date"]).dt.date
units_occupancy_rate.head()
# %%
############################
print("hospitalization1")
display(hospitalization1.head(5))
print("hospitalization2")
display(hospitalization2.head(5))
print("units_occupancy_rate")
display(units_occupancy_rate.head(5))
############################

# %%
# Normalize the dates to be only the date without time
hospitalization2['Admission_Entry_Date'] = pd.to_datetime(hospitalization2['Admission_Entry_Date']).dt.date
hospitalization2['Release_Date'] = pd.to_datetime(hospitalization2['Release_Date']).dt.date
hospitalization2['Admission_Entry_Date2'] = pd.to_datetime(hospitalization2['Admission_Entry_Date2']).dt.date
hospitalization2['Release_Date2'] = pd.to_datetime(hospitalization2['Release_Date2']).dt.date

# Calculate the days between Release_Date and Admission_Entry_Date2
hospitalization2['Days_Between'] = (pd.to_datetime(hospitalization2['Admission_Entry_Date2']) - pd.to_datetime(hospitalization2['Release_Date'])).dt.days

# Categorize the Days_Between into 'Short', 'Medium', 'Long'
hospitalization2['Duration_Category'] = pd.cut(hospitalization2['Days_Between'], bins=[-1, 10, 20, 30], labels=['Short', 'Medium', 'Long'])

# %%
# Display the updated dataframe with the new category column
hospitalization2.head()

# %% 
plt.figure(figsize=(12, 8))
sns.histplot(hospitalization2["Days_Between"], kde=True, bins=30, alpha=0.6
    )
plt.title("Occupancy Rate Distribution by Department")
plt.xlabel("Occupancy Rate")
plt.ylabel("Frequency")
plt.legend(title="Department")
plt.show()


# %%
def get_occupancy_rate(date, department, units_occupancy_rate):
    """
    Retrieve the Occupancy Rate for a given date and department.

    Parameters:
    date (str or pd.Timestamp): The date for which to retrieve the Occupancy Rate.
    department (str or int): The department for which to retrieve the Occupancy Rate.
    units_occupancy_rate (pd.DataFrame): The DataFrame containing occupancy rate data.

    Returns:
    float: The Occupancy Rate for the given date and department, or None if not found.
    """
    # Convert date to datetime if it's not already
    date = pd.to_datetime(date).date()
    
    # Convert department to string and strip any leading/trailing spaces
    department = str(department).strip()

    # Filter the DataFrame for the matching date and department
    result = units_occupancy_rate[(units_occupancy_rate['Date'] == date) & 
                                  (units_occupancy_rate['Department'].astype(str).str.strip() == department)]

    # Return the Occupancy Rate if found, otherwise return None
    if not result.empty:
        return result['Occupancy Rate'].values[0]
    else:
        return None


    
# %% 
rate = get_occupancy_rate("2022-01-25", 4 ,units_occupancy_rate)
display(rate)

# %%
# Add a new column 'Unit_Occupancy_Rate_ReleaseDate' by applying the get_occupancy_rate function
hospitalization2['Unit_Occupancy_Rate_ReleaseDate'] = hospitalization2.apply(
    lambda row: get_occupancy_rate(row['Release_Date'], row['unitName1'], units_occupancy_rate), axis=1
)
# %% 
# display the updated DataFrame with the new column
display(hospitalization2[['Release_Date', 'unitName1', 'Unit_Occupancy_Rate_ReleaseDate']].head())

# %%
# Check for Null Values
hospitalization2.Unit_Occupancy_Rate_ReleaseDate.isnull().sum()

# %% 
# Identify all rows (indexes) that contain any NaN values
nan_indexes = hospitalization2[hospitalization2['Unit_Occupancy_Rate_ReleaseDate'].isna()].index

# Display the indexes with NaN values
print("Indexes with NaN values:", nan_indexes.tolist())

# Drop the rows with NaN values from the hospitalization2 DataFrame
hospitalization2_cleaned = hospitalization2.drop(index=nan_indexes)

# Display the cleaned DataFrame to verify that the rows were removed
display(hospitalization2_cleaned.head())




# %%
