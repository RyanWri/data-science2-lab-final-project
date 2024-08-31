# %% 
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import random
random.seed(42)
from IPython.display import display, HTML


# Function to fill empty cells with the mode value
def fill_empty_cells(df):
    df_copy = df.copy()
    df_copy.replace('', np.nan, inplace=True)
    
    summary_df = pd.DataFrame(columns=['Column', 'Number Of Missing Cells', 'Percentage Of Null Values',
                                       'Missing Indices', 'Fill Summary'])
    
    total_rows = len(df_copy)
    filled_columns = []
    
    for col in df_copy.columns:
        null_indices = df_copy[df_copy[col].isnull()].index
        if null_indices.size > 0:
            num_missing = len(null_indices)
            percentage_missing = (num_missing / total_rows) * 100
            
            fill_value = df_copy[col].mode()[0]
            fill_summary = f"filled with mode: {fill_value}"
            df_copy[col].fillna(fill_value, inplace=True)
            
            filled_columns.append(col)
            
            temp_df = pd.DataFrame({
                'Column': [col],
                'Number Of Missing Cells': [num_missing],
                'Percentage Of Null Values': [f'{percentage_missing:.2f}%'],
                'Missing Indices': [null_indices.tolist()],
                'Fill Summary': [fill_summary]
            })
            
            summary_df = pd.concat([summary_df, temp_df], ignore_index=True)
    
    return df_copy, summary_df

def empty_cells(df, dataset_name="Dataset"):
    if df.isnull().values.any() or (df == '').any().any():
        display(HTML(f'<h2>{dataset_name} Data Set Contains Empty Cells</h2>'))
        df_filled, summary_df = fill_empty_cells(df)
    else:
        display(HTML(f'<h2>{dataset_name} Data Set Does Not Contain Empty Cells</h2>'))
        df_filled, summary_df = df.copy(), pd.DataFrame()

    return df_filled, summary_df

def translate_hospitalization1_dataset(df):
    # Rename columns
    df.rename(columns={
        "סוג קבלה": "Receipt_Type",
        "מהיכן המטופל הגיע": "Patient_Origin",
        "אבחנות בקבלה": "Admission_Diagnoses",
        "ימי אשפוז": "Hospitalization_Days",
        "אבחנות בשחרור": "Release_Diagnoses",
        "רופא משחרר-קוד": "Release_Doctor_Code"
    }, inplace=True)

    # Replace values in the "Receipt_Type" column
    df["Receipt_Type"].replace({
        "דחוף": "Urgent",
        "מוזמן": "Invited",
        "אשפוז יום": "Day_Hospitalization"
    }, inplace=True)

    # Replace values in the "Patient_Origin" column
    df["Patient_Origin"].replace({
        "מביתו": "Home",
        "ממוסד": "Institution",
        "אחר": "Other",
        "ממרפאה": "Clinic",
        "מבית חולים אחר": "Different_Hospital"
    }, inplace=True)

    # Replace values in the "Release_Type" column
    df["Release_Type"].replace({
        "שוחרר לביתו": "Home_Released",
        "שוחרר למוסד": "Institution_Released"
    }, inplace=True)

    return df

# %% 
# Load Dataset
path = "F:\\לימודים\\תואר שני\\סמסטר ב\\Data Science 2\\DS2-Final Project\\hospitalization1.xlsx"
hptl1 = pd.read_excel(path)

display(HTML('<h2>Hospitalization1 Dataset</h2>'))
display(hptl1.head(5))
# %% 
display(HTML('<h2>Translated Hospitalization1 Dataset</h2>'))

hptl1 = translate_hospitalization1_dataset(hptl1)

display(hptl1.head(5))
# %% 
# Apply the empty_cells function to the hospitalization dataset
# Example usage
filled_hptl1, summary = empty_cells(hptl1, dataset_name="Hospitalization1")
display(HTML('<h3>Missing Values Completion Summary </h3>'))
display(summary)
# %% 

temp = empty_cells(filled_hptl1, dataset_name=" Filled Hospitalization1")


