import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# Step 1: Load the "GeneralData" and hospitalization data
# File path for the original data
file_path = '/Users/matanoz/Documents/לימודים תואר שני/סמסטר ב׳/data-science2-lab-final-project/src/data/rehospitalization.xlsx'
generaldata_df = pd.read_excel(file_path, sheet_name='GeneralData', engine='openpyxl')

# Load the 'hospitalization2_cleaned.xlsx' file uploaded earlier
hospitalization2_cleaned = pd.read_excel('/Users/matanoz/Documents/לימודים תואר שני/סמסטר ב׳/data-science2-lab-final-project/src/team_7/task_23/hospitalization2_cleaned.xlsx')

# Step 2: Define the column translation dictionary for 'GeneralData'
column_translation = {
    "גורם משלם": "Payer",
    "משקל": "Weight",
    "גובה": "Height",
    "מחלות כרוניות": "Chronic Diseases",
    "השכלה": "Education",
    "מספר ילדים": "Number of Children",
    "מצב משפחתי": "Marital Status",
    "תרופות קבועות": "Regular Medications"
}

# Apply the translation to the column names for 'GeneralData'
generaldata_df.rename(columns=column_translation, inplace=True)

# Display the translated DataFrame
print("Translated General Data:")
print(generaldata_df.head())

# Step 3: Check for null values in the 'GeneralData'
null_counts_generaldata = generaldata_df.isnull().sum()
print("\nNull values in 'GeneralData':")
print(null_counts_generaldata)

# Dropping rows with null values in 'Regular Medications'
generaldata_df_cleaned = generaldata_df.dropna(subset=['Regular Medications'])
generaldata_df_cleaned = generaldata_df_cleaned.dropna(axis=1)  # Remove columns with remaining null values

# Step 4: Ensure we are only processing strings in the 'Regular Medications' column
medications_series_cleaned = generaldata_df_cleaned['Regular Medications'].dropna()
medications_series_cleaned = medications_series_cleaned[medications_series_cleaned.apply(lambda x: isinstance(x, str))].str.split(',')

# Step 5: Flatten the list of medications
all_medications_cleaned = [med.strip() for sublist in medications_series_cleaned for med in sublist]

# Create a DataFrame to count the occurrences of each medication
medications_cleaned_df = pd.DataFrame(all_medications_cleaned, columns=['Regular Medications'])

# Step 6: Find the top 20 most common medications
top_20_medications_cleaned = medications_cleaned_df['Regular Medications'].value_counts().head(20)

# Step 7: Plot the 20 most common medications
plt.figure(figsize=(12, 8))
sns.barplot(x=top_20_medications_cleaned.values, y=top_20_medications_cleaned.index, palette='viridis')
plt.title("Top 20 Most Common Regular Medications", fontsize=16)
plt.xlabel("Count", fontsize=14)
plt.ylabel("Medication", fontsize=14)
plt.tight_layout()
plt.show()

def merge_medications_with_hospitalization(generaldata_df, hospitalization2_cleaned):
    """
    Merges the 'Regular Medications' from generaldata_df into hospitalization2_cleaned
    based on the 'Patient' column.

    Parameters:
    generaldata_df (pd.DataFrame): The dataframe containing patient general data, including 'Regular Medications'.
    hospitalization2_cleaned (pd.DataFrame): The dataframe containing hospitalization data.

    Returns:
    pd.DataFrame: The merged dataframe with 'Regular Medications' added.
    """
    # Step 1: Check if 'Patient' column exists in both dataframes
    if 'Patient' in generaldata_df.columns and 'Patient' in hospitalization2_cleaned.columns:
        # Merge both datasets on 'Patient' column
        merged_df = pd.merge(hospitalization2_cleaned, generaldata_df[['Patient', 'Regular Medications']],
                             on='Patient', how='left')
        print("Merging completed successfully.")
    else:
        print("Patient column not found in one or both dataframes. Please adjust the merge key.")
        return None

    # Step 2: Display a few rows of the merged dataframe to verify
    print("Merged Data (first 5 rows):")
    print(merged_df[['Patient', 'Regular Medications']].head())

    # Step 3: Optionally save the merged dataframe to an Excel file
    merged_df.to_excel('merged_hospitalization_with_medication.xlsx', index=False)
    print("Merged dataset saved to 'merged_hospitalization_with_medication.csv'")

    return merged_df

# Assuming 'generaldata_df' and 'hospitalization2_cleaned' are already loaded
merged_df = merge_medications_with_hospitalization(generaldata_df, hospitalization2_cleaned)


def add_top_20_medications(merged_data):
    """
    Adds binary columns for the top 20 most common medications to the dataset and saves it as a CSV file.

    Parameters:
    merged_data (pd.DataFrame): The merged dataframe containing 'Regular Medications' and patient data.

    Returns:
    pd.DataFrame: The updated dataframe with top 20 medications as binary columns.
    """
    # Step 1: Ensure we are only processing strings in the 'Regular Medications' column
    medications_series = merged_data['Regular Medications'].dropna()
    medications_series = medications_series[medications_series.apply(lambda x: isinstance(x, str))].str.split(',')

    # Step 2: Flatten the list of medications
    all_medications = [med.strip() for sublist in medications_series for med in sublist]

    # Step 3: Create a DataFrame to count the occurrences of each medication
    medications_df = pd.DataFrame(all_medications, columns=['Regular Medications'])

    # Step 4: Find the top 20 most common medications
    top_20_medications = medications_df['Regular Medications'].value_counts().head(20).index

    # Step 5: Create binary columns for each of the top 20 medications
    for med in top_20_medications:
        merged_data[med] = merged_data['Regular Medications'].apply(lambda x: 1 if med in str(x) else 0)

    # Step 6: Save the updated DataFrame to CSV
    merged_data.to_excel('task_29_dataset.xlsx', index=False)
    print("Final dataset with top 20 medications saved to 'merged_data_with_top_20_medications.xlsx'")

    return merged_data

# Assuming 'merged_data_final' is the merged dataset you provided
merged_df_updated = add_top_20_medications(merged_df)

# %%
# Function to rename columns
def rename_columns(dataframe, rename_dict):
    """
    Renames columns in a dataframe based on the provided dictionary.

    Parameters:
    dataframe (pd.DataFrame): The dataframe whose columns are to be renamed.
    rename_dict (dict): A dictionary mapping old column names to new column names.

    Returns:
    pd.DataFrame: The dataframe with renamed columns.
    """
    dataframe = dataframe.rename(columns=rename_dict)
    print(f"Columns renamed successfully: {rename_dict}")
    return dataframe

# Function to remove specified columns
def remove_columns(dataframe, columns_to_remove):
    """
    Removes specified columns from a dataframe.

    Parameters:
    dataframe (pd.DataFrame): The dataframe from which columns are to be removed.
    columns_to_remove (list): A list of column names to be removed.

    Returns:
    pd.DataFrame: The dataframe with specified columns removed.
    """
    dataframe = dataframe.drop(columns=columns_to_remove, errors='ignore')
    print(f"Columns removed successfully: {columns_to_remove}")
    return dataframe

#usage
# Load the dataset
file_path = 'task_29_dataset.xlsx'
df = pd.read_excel(file_path)

# Define column renaming and columns to remove
rename_dict = {
    'Days_Between': 'Days_Between_16',
    'Duration_Category': 'Duration_Category_16',
    # Add other column mappings as needed
}

columns_to_remove = ['ct', 'מחלקות מייעצות','אבחנות בשחרור','אבחנות בקבלה','ימי אשפוז','רופא משחרר','סוג קבלה']

# Rename columns
df_renamed = rename_columns(df, rename_dict)

# Remove columns
df_cleaned = remove_columns(df_renamed, columns_to_remove)

# Save the updated DataFrame
df_cleaned.to_excel('task_29_dataset.xlsx', index=False)
print("Updated dataset saved as 'updated_task_29_dataset.xlsx'")
# %%
import pandas as pd

# Load the dataset
file_path = 'task_29_dataset.xlsx'
df = pd.read_excel(file_path)

# Function to fix inconsistencies in the binary medication columns
def fix_medication_inconsistencies_exact(dataframe, medication_columns):
    """
    Ensures that binary medication columns are marked TRUE (1) only if the medication
    is found exactly as a distinct number in the 'Regular Medications' column for each patient.

    Parameters:
    dataframe (pd.DataFrame): The dataset containing patient information.
    medication_columns (list): A list of binary column names for medications (e.g., ['37', '38']).

    Returns:
    pd.DataFrame: The dataframe with inconsistencies fixed.
    """
    # Loop through each binary medication column
    for med_column in medication_columns:
        # Loop through each row and fix inconsistencies
        for index, row in dataframe.iterrows():
            # If the binary column is TRUE (1), check if the medication is exactly in 'Regular Medications'
            if row[med_column] == 1:
                # Split the 'Regular Medications' into distinct medications
                regular_meds = str(row['Regular Medications']).split(',')
                # Strip spaces and check for exact matches
                regular_meds = [med.strip() for med in regular_meds]
                if med_column not in regular_meds:
                    # If the medication is not present as a distinct number, set the value to 0 (FALSE)
                    dataframe.at[index, med_column] = 0

    return dataframe

# Get the list of columns corresponding to the medications (assuming they are binary columns)
medication_columns = [col for col in df.columns if col.isdigit()]  # Assuming binary columns are named as numbers

# Fix inconsistencies
df_fixed = fix_medication_inconsistencies_exact(df, medication_columns)

# Save the corrected DataFrame to Excel
df_fixed.to_excel('task_29_dataset.xlsx', index=False)
print("Fixed dataset saved to 'task_29_dataset.xlsx'.")


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the fixed dataset
file_path = 'task_29_dataset.xlsx'
df_fixed = pd.read_excel(file_path)

# Display the first few rows of the fixed dataset to ensure it loaded correctly
print(df_fixed.head())
# %%
# Function to add the 'Days_Between_17' and 'Days_Between_18' columns
def add_days_between_column(data):
    # Convert date columns to datetime
    data['Admission_Entry_Date'] = pd.to_datetime(data['Admission_Entry_Date'])
    data['Release_Date'] = pd.to_datetime(data['Release_Date'])

    data['Admission_Entry_Date2'] = pd.to_datetime(data['Admission_Entry_Date2'])
    data['Release_Date2'] = pd.to_datetime(data['Release_Date2'])

    # Create 'Days_Between_17' column
    data['Days_Between_17'] = (data['Release_Date'] - data['Admission_Entry_Date']).dt.days
    # Create 'Days_Between_18' column
    data['Days_Between_18'] = (data['Release_Date2'] - data['Admission_Entry_Date2']).dt.days

    return data

# Apply the function to the loaded dataset
df_fixed = add_days_between_column(df_fixed)

# Display the first few rows of the updated dataset to ensure the new columns are added
print(df_fixed.head())

# Save the updated dataset back to Excel
df_fixed.to_excel("task_29_dataset.xlsx", index=False)

# Function to categorize 'Days_Between_17' into 'Duration_Category_17'
def categorize_days_between_17(data):
    # Define the categorization logic for 'Duration_Category_17'
    def categorize_duration(days):
        if days <= 2:
            return 'Short'
        elif 3 <= days <= 4:
            return 'Medium'
        else:
            return 'Long'

    # Apply the categorization function to 'Days_Between_17' and create 'Duration_Category_17'
    data['Duration_Category_17'] = data['Days_Between_17'].apply(categorize_duration)

    return data
df_fixed = categorize_days_between_17(df_fixed)
df_fixed.head()

def categorize_days_between_18(data):
    # Define the categorization logic for 'Duration_Category_18'
    def categorize_duration(days):
        if days <= 2:
            return 'Short'
        elif 3 <= days <= 4:
            return 'Medium'
        else:
            return 'Long'

    # Apply the categorization function to 'Days_Between_18' and create 'Duration_Category_18'
    data['Duration_Category_18'] = data['Days_Between_18'].apply(categorize_duration)

    return data

df_fixed = categorize_days_between_18(df_fixed)
df_fixed.head()
df_fixed.to_excel("task_29_dataset.xlsx", index=False)

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the fixed dataset
file_path = 'task_29_dataset.xlsx'
df_fixed = pd.read_excel(file_path)

# Display the first few rows of the fixed dataset to ensure it loaded correctly
print(df_fixed.head())
# %%
# Function to add the 'Days_Between_17' and 'Days_Between_18' columns
def add_days_between_column(data):
    # Convert date columns to datetime
    data['Admission_Entry_Date'] = pd.to_datetime(data['Admission_Entry_Date'])
    data['Release_Date'] = pd.to_datetime(data['Release_Date'])

    data['Admission_Entry_Date2'] = pd.to_datetime(data['Admission_Entry_Date2'])
    data['Release_Date2'] = pd.to_datetime(data['Release_Date2'])

    # Create 'Days_Between_17' column
    data['Days_Between_17'] = (data['Release_Date'] - data['Admission_Entry_Date']).dt.days
    # Create 'Days_Between_18' column
    data['Days_Between_18'] = (data['Release_Date2'] - data['Admission_Entry_Date2']).dt.days

    return data

# Apply the function to the loaded dataset
df_fixed = add_days_between_column(df_fixed)

# Display the first few rows of the updated dataset to ensure the new columns are added
print(df_fixed.head())

# Save the updated dataset back to Excel
df_fixed.to_excel("task_29_dataset.xlsx", index=False)

# Function to categorize 'Days_Between_17' into 'Duration_Category_17'
def categorize_days_between_17(data):
    # Define the categorization logic for 'Duration_Category_17'
    def categorize_duration(days):
        if days <= 2:
            return 'Short'
        elif 3 <= days <= 4:
            return 'Medium'
        else:
            return 'Long'

    # Apply the categorization function to 'Days_Between_17' and create 'Duration_Category_17'
    data['Duration_Category_17'] = data['Days_Between_17'].apply(categorize_duration)

    return data
df_fixed = categorize_days_between_17(df_fixed)
df_fixed.head()

def categorize_days_between_18(data):
    # Define the categorization logic for 'Duration_Category_18'
    def categorize_duration(days):
        if days <= 2:
            return 'Short'
        elif 3 <= days <= 4:
            return 'Medium'
        else:
            return 'Long'

    # Apply the categorization function to 'Days_Between_18' and create 'Duration_Category_18'
    data['Duration_Category_18'] = data['Days_Between_18'].apply(categorize_duration)

    return data

df_fixed = categorize_days_between_18(df_fixed)
# %%
df_fixed.head()
# %%
df_fixed.to_excel("task_29_dataset_final.xlsx", index=False)
# %%
