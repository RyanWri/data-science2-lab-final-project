import pandas as pd
import os

def process_medication_data(general_data_df, cleaned_drugs_df, cleaned_hospitalization1_df):
    # Split the 'תרופות קבועות' column into separate medication codes
    general_data_df['תרופות קבועות'] = general_data_df['תרופות קבועות'].str.split(', ')

    # Explode the column to have one medication code per row
    exploded_general_data_df = general_data_df.explode('תרופות קבועות')

    # Rename columns for clarity
    exploded_general_data_df.rename(columns={'תרופות קבועות': 'Medication_Code'}, inplace=True)

    # Convert Medication_Code to numeric
    exploded_general_data_df['Medication_Code'] = pd.to_numeric(exploded_general_data_df['Medication_Code'], errors='coerce')

    # Merge exploded general data with Drugs data to get medication names
    merged_medications = pd.merge(exploded_general_data_df, cleaned_drugs_df, left_on='Medication_Code', right_on='Code', how='inner')

    # Print available columns for debugging
    print("Available columns in cleaned_hospitalization1_df:", cleaned_hospitalization1_df.columns)

    # Merge with hospitalization data to get 'Admission_Entry_Date' and other necessary information
    necessary_columns = ['Patient', 'Admission_Entry_Date', 'Days_Between_Hospitalizations', 'Days_First_Hospitalization', 'Days_Second_Hospitalization']
    available_columns = [col for col in necessary_columns if col in cleaned_hospitalization1_df.columns]

    if available_columns:
        merged_medications = pd.merge(merged_medications, cleaned_hospitalization1_df[available_columns], on='Patient', how='left')
    else:
        print("Warning: Necessary columns for merging are not found in hospitalization data.")

    # Further clean merged_medications if necessary
    merged_medications.dropna(subset=['Patient', 'Medication_Code'], inplace=True)

    return merged_medications

if __name__ == "__main__":
    # Create the medications_data directory if it does not exist
    os.makedirs('medications_data', exist_ok=True)

    # Load cleaned data from the previous step
    general_data_df = pd.read_csv(r'/Users/liav/Desktop/GIT/DL_Final/task25/medications_data/general_data_cleaned.csv')
    cleaned_drugs_df = pd.read_csv(r'/Users/liav/Desktop/GIT/DL_Final/task25/medications_data/drugs_cleaned.csv')
    cleaned_hospitalization1_df = pd.read_csv(r'/Users/liav/Desktop/GIT/DL_Final/task25/medications_data/hospitalization1_cleaned.csv')

    # Process the medication data
    merged_medications = process_medication_data(general_data_df, cleaned_drugs_df, cleaned_hospitalization1_df)
    
    # Save processed data for further use in the medications_data folder
    merged_medications.to_csv(r'/Users/liav/Desktop/GIT/DL_Final/task25/medications_data/merged_medications.csv', index=False)
    print("Data processing completed successfully.")