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

        # Display types of each column
        print(self.df.dtypes)

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
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.show()

    def categorical_frequency(self):
        # Count plots for categorical columns
        for col in self.df.select_dtypes(include=["object", "category"]).columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(y=col, data=self.df, order=self.df[col].value_counts().index)
            plt.title(f"Distribution of {col}")
            plt.show()

    def outliers(self):
        # Boxplots for numerical columns to identify outliers
        for col in self.df.select_dtypes(include=["float64", "int64"]).columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=col, data=self.df)
            plt.title(f"Boxplot of {col}")
            plt.show()

    def kde_numerical(self):
        for col in self.df.select_dtypes(include=["float64", "int64"]).columns:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(self.df[col], fill=True)
            plt.title(f"Distribution of Numerical Column {col}")
            plt.show()

    def numeric_vs_categorical(self, numeric_col, categorical_col):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=categorical_col, y=numeric_col, data=self.df)
        plt.title(
            f"Boxplot of Numerical Column {numeric_col} by Categorical Column {categorical_col}"
        )
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.violinplot(x=categorical_col, y=numeric_col, data=self.df)
        plt.title(
            f"Violin Plot of Numerical Column {numeric_col} by Categorical Column {categorical_col}"
        )
        plt.show()

    def numeric_over_time(self, numeric_col, time_col):
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=time_col, y=numeric_col, data=self.df)
        plt.title(
            f"Time Series of Numerical Column {numeric_col} over Time Column {time_col}"
        )
        plt.show()

    def rehospitalized_patients_data(self):
        # Group by 'user_id' and count the number of records for each user
        user_counts = self.df["Patient"].value_counts()

        # Filter for users with at least 2 records
        users_with_multiple_records = user_counts[user_counts >= 2].index

        # Filter the original DataFrame to include only these users
        df_rehospitalized_patients = self.df[
            self.df["Patient"].isin(users_with_multiple_records)
        ]

        return df_rehospitalized_patients
