import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_excel_sheet(file_path, sheet_name):
    """
    Reads a specific sheet from an Excel file.

    Parameters:
    file_path (str): The path to the Excel file.
    sheet_name (str): The name of the sheet to be read.

    Returns:
    pd.DataFrame: The data from the specified sheet as a pandas DataFrame.
    """
    try:
        # Read the specified sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None


class EDA:
    def __init__(self, df):
        self.df = df

    def stats(self):
        # Display the first few rows of the DataFrame
        print(self.df.head())

        # Get a summary of the DataFrame
        print(self.df.info())

        # Display summary statistics for numerical columns
        print(self.df.describe())

        # Display the number of missing values in each column
        print(self.df.isnull().sum())

    def histogram(self):
        # For numerical columns
        print(self.df.hist(figsize=(12, 12)))

        # For categorical columns
        for col in self.df.columns:
            print(self.df[col].value_counts())

    def correlation(self):
        # Display the correlation matrix
        print(self.df.corr())

        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.show()
