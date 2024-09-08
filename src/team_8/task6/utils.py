import random
import pandas as pd




def check_null_values_in_df(df):
    # Check for nulls in the data after null filling.
    print(df.isna().sum())
    #print(df[df.isna().any(axis=1)])



def choose_random_string(string1, string2,string3):
    return random.choice([string1, string2,string3])


def drop_rows_with_many_nans(df, columns, max_nans=2):
    # Count the number of NaN values in the specified columns for each row
    nan_counts = df[columns].isna().sum(axis=1)
    
    # Drop rows where the number of NaNs exceeds the max_nans threshold
    df_filtered = df[nan_counts <= max_nans]
    
    return df_filtered



