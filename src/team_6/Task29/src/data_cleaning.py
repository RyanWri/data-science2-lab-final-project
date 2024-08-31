import pandas as pd
import os

def load_and_clean_data(file_path):
    try:
        # Specify the engine to use when reading the Excel file
        drugs_df = pd.read_excel(file_path, sheet_name='Drugs', engine='openpyxl')
        hospitalization1_df = pd.read_excel(file_path, sheet_name='hospitalization1', engine='openpyxl')
        general_data_df = pd.read_excel(file_path, sheet_name='GeneralData', engine='openpyxl')
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None, None, None
    except FileNotFoundError:
        print(f"FileNotFoundError: The file at {file_path} was not found.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

    # Clean 'Drugs' DataFrame by dropping rows with missing 'Drug' names
    cleaned_drugs_df = drugs_df.dropna(subset=['Drug']).reset_index(drop=True)

    # Fill missing values in 'hospitalization1' DataFrame
    cleaned_hospitalization1_df = hospitalization1_df.copy()
    cleaned_hospitalization1_df['אבחנות בקבלה'] = cleaned_hospitalization1_df['אבחנות בקבלה'].fillna('Unknown')
    cleaned_hospitalization1_df['אבחנות בשחרור'] = cleaned_hospitalization1_df['אבחנות בשחרור'].fillna('Unknown')
    cleaned_hospitalization1_df = cleaned_hospitalization1_df.dropna(subset=['Patient', 'Admission_Medical_Record'])

    return cleaned_drugs_df, cleaned_hospitalization1_df, general_data_df

if __name__ == "__main__":
    # Create the medications_data directory if it does not exist
    os.makedirs('medications_data', exist_ok=True)

    file_path = r'/Users/liav/Desktop/GIT/DL_Final/task25/data/rehospitalization.xlsx'
    cleaned_drugs_df, cleaned_hospitalization1_df, general_data_df = load_and_clean_data(file_path)
    
    if cleaned_drugs_df is not None:
        # Save cleaned data to CSV files in the medications_data folder
        cleaned_drugs_df.to_csv(r'/Users/liav/Desktop/GIT/DL_Final/task25/medications_data/drugs_cleaned.csv', index=False)
        cleaned_hospitalization1_df.to_csv(r'/Users/liav/Desktop/GIT/DL_Final/task25/medications_data/hospitalization1_cleaned.csv', index=False)
        general_data_df.to_csv(r'/Users/liav/Desktop/GIT/DL_Final/task25/medications_data/general_data_cleaned.csv', index=False)
        print("Data cleaning ended")
    else:
        print("Data cleaning failed due to an error in loading the Excel file.")
