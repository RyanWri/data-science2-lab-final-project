import pandas as pd
from task_26.rehospital_model import encode_target

# Function to convert date columns to numeric values
def convert_date_to_numeric(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    df[column_name] = (df[column_name] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return df

def encode_columns(df, categorical_features):
    df_encoded_combined = pd.DataFrame()  # Initialize an empty DataFrame
    for feature in categorical_features:
        df_encoded = pd.Series(encode_target(df,feature), name=feature)
        df_encoded = pd.DataFrame(df_encoded, columns=[feature])

    # Concatenate the new encoded column(s) to the combined DataFrame
        df_encoded_combined = pd.concat([df_encoded_combined, df_encoded], axis=1)

    # Drop original categorical columns before concatenating the encoded ones
    df_without_categorical = df.drop(categorical_features, axis=1)

    return pd.concat([df_without_categorical,df_encoded_combined], axis=1)
