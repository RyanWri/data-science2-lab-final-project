#!/usr/bin/env python
# coding: utf-8

# # Data cleaning and completion for table: general data

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

df_GeneralData['מספר ילדים'].fillna('לא ידוע', inplace=True)

df_GeneralData['מצב משפחתי'].fillna('לא ידוע', inplace=True)

# Create a mapping dictionary from the mapping file
# Assuming the mapping file has columns 'Code' and 'Drug'
mapping_dict = dict(zip(df_Drugs['Code'].astype(str), df_Drugs['Drug'].astype(str)))

# Function to replace codes with names
def replace_codes_with_names(codes_str):
    if isinstance(codes_str, str):  # Ensure the value is a string
        codes = [code.strip() for code in codes_str.split(',')]  # Split and strip whitespace
        names = [mapping_dict.get(code, value) for code in codes]  # Get names from dictionary
        # Ensure all items in names are strings
        names = [str(name) for name in names]
        return ', '.join(names)  # Join names back into a string
    return codes_str  # Return the original value if it's not a string

# Apply the function to each row in the column 'תרופות קבועות'
# Create a new column 'DrugNames' to store the results
df_GeneralData['DrugNames'] = df_GeneralData['תרופות קבועות'].apply(replace_codes_with_names)



