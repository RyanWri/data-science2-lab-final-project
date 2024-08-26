import pandas as pd

# Load the dataset
data = pd.read_csv('erBeforeHospitalization2.csv', encoding='utf-8')
print(data)

# Identify missing data
missing_data = data.isnull().sum()
print(missing_data)

# Define the conditions for rows where all the data of the second part is missing
condition = (
    data['Medical_Record'].isnull() & 
    data['ev_Admission_Date'].isnull() & 
    data['ev_Release_Time'].isnull() & 
    data['דרך הגעה למיון'].isnull() & 
    data['מיון'].isnull() & 
    data['urgencyLevelTime'].isnull() & 
    data['אבחנות במיון'].isnull() &
    data['codeDoctor'].isnull()
)

# Fill the columns based on the condition
data.loc[condition, 'Medical_Record'] = 1000000
data.loc[condition, 'ev_Admission_Date'] = '1900-01-01'
data.loc[condition, 'ev_Release_Time'] = '1900-01-01'
data.loc[condition, 'דרך הגעה למיון'] = 'No Emergency Visit'
data.loc[condition, 'מיון'] = 'No Emergency Visit'
data.loc[condition, 'urgencyLevelTime'] = 0
data.loc[condition, 'אבחנות במיון'] = 0
data.loc[condition, 'codeDoctor'] = 0

# 2. Fill missing data with specific values
data['דרך הגעה למיון'].fillna('Not provided', inplace=True)

# Condition to identify rows where 'מיון' is 'המחלקה לרפואה דחופה'
condition = data['מיון'] == 'המחלקה לרפואה דחופה'

# Fill null values in 'אבחנות במיון' with 1 where the condition is met
data.loc[condition & data['אבחנות במיון'].isnull(), 'אבחנות במיון'] = 1

# Fill null values in 'codeDoctor' with 1 where the condition is met
data.loc[condition & data['codeDoctor'].isnull(), 'codeDoctor'] = 1

# Fill all remaining null values in the entire DataFrame with 0
data.fillna(0, inplace=True)
print(data)

# Identify missing data
missing_data = data.isnull().sum()
print(missing_data)

# Save the modified dataset (optional)
data.to_csv('modified_erBeforeHospitalization2.csv', index=False, encoding='utf-8-sig')
