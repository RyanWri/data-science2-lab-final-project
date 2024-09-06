#!/usr/bin/env python
# coding: utf-8

# # Data cleaning and completion for table: general data
import pandas as pd

GeneralData = r'src\data\GeneralData.xlsx'
Drugs = r'src\data\Drugs.xlsx'

df_GeneralData = pd.read_excel(GeneralData, engine='openpyxl')
df_Drugs = pd.read_excel(Drugs, engine='openpyxl')

mean_weight = df_GeneralData['משקל'].mean()
df_GeneralData['משקל'].fillna(mean_weight, inplace=True)

mean_height = df_GeneralData['גובה'].mean()
df_GeneralData['גובה'].fillna(mean_weight, inplace=True)

# Calculate BMI Column
df_GeneralData.loc[df_GeneralData['BMI'].isna(), 'BMI'] = df_GeneralData.apply(
    lambda row: row['משקל'] / ((row['גובה'] /100 )**2) if pd.notnull(row['משקל']) and pd.notnull(row['גובה']) else None,
    axis=1
)

# Filling empty values in the column in the value "לא ידוע"
df_GeneralData['השכלה'] = df_GeneralData['השכלה'].apply(lambda x: None if isinstance(x, (int, float)) else x)
df_GeneralData['השכלה'].fillna('לא ידוע', inplace=True)


# Convert the column to numeric, coercing errors to NaN
pd.to_numeric(df_GeneralData['מספר ילדים'], errors='coerce').notnull().all()
# Calculate the mean of the numeric values, ignoring NaN values
median_value = df_GeneralData['מספר ילדים'].median()
# Replace non-numeric (NaN) values with the mean
df_GeneralData['מספר ילדים'].fillna(median_value, inplace=True)

marital_status_mapping = {
    'לא ידוע': 1,
    'רווק': 2,
    'נשוי': 3,
    'פרוד': 4,
    'גרוש': 5,
    'אלמן': 6
}

# Apply the mapping to the 'מצב משפחתי' column
df_GeneralData['מצב משפחתי'] = df_GeneralData['מצב משפחתי'].map(marital_status_mapping)



# Save the updated DataFrame to a new Excel file
output_file_path = 'src\data\GeneralData.csv'
df_GeneralData.to_csv(output_file_path, index = False, encoding='utf-8-sig')

