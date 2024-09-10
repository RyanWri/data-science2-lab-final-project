import pandas as pd
from task_26.rehospital_model import encode_target

# Function to convert date columns to numeric values
def convert_date_to_numeric(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    df[column_name] = (df[column_name] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return df

def encode_columns(df, categorical_features):
    df_encoded = df.copy()  # Make a copy of the original DataFrame
    
    for feature in categorical_features:
        df_encoded[feature] = encode_target(df_encoded, feature)  # Directly encode and replace in the copy

    return df_encoded  # Return the modified DataFrame
