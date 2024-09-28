import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
    
def rename_columns_from_json(dataframe, directory_path, json_file_name):
    """
    Rename columns in the DataFrame using a mapping dictionary from a JSON file.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame with Hebrew column names.
    directory_path (str): The directory path where the JSON file is located.
    json_file_name (str): The name of the JSON file containing the column mapping (Hebrew to English).

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.
    """
    # Construct the full path to the JSON file
    json_file_path = os.path.join(directory_path, json_file_name)

    # Load the column mapping from the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        column_mapping = json.load(f)
    
    # Rename columns using the mapping
    dataframe_renamed = dataframe.rename(columns=column_mapping)
    
    # Return the DataFrame with renamed columns
    return dataframe_renamed

def read_csv_file(file_path, delimiter=',', encoding='utf-8', header='infer'):
    """
    Reads a CSV file and returns a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.
    delimiter (str): The delimiter used in the CSV file (default is ',').
    encoding (str): The encoding used to read the CSV file (default is 'utf-8').
    header (int, list of int, or None): Row number(s) to use as the column names, or None if no header (default is 'infer').

    Returns:
    pd.DataFrame: The DataFrame containing the CSV data.
    """
    try:
        # Read the CSV file
        dataframe = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, header=header)
        return dataframe
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.ParserError:
        print("Error: There was a problem parsing the CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

class EDA:
    def __init__(self, df):
        self.df = df

    def stats(self):
        # Display the first few rows of the DataFrame
        print("First 5 indexes:")
        print(self.df.head())

        print("Last 5 indexes:")
        print(self.df.tail())

        # Display types of each column
        print("Column types:")
        print(self.df.dtypes)

        # Get a summary of the DataFrame
        print("Dataframe summary:")
        print(self.df.info())

        # Display summary statistics for numerical columns
        print(self.df.describe())

        # Display the number of missing values in each column
        print("Missing values:")
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

    def categorical_frequency(self, filter_columns=None):
        """
        Plot the frequency distribution of all categorical columns, with specific filtering
        for selected columns where categories with counts less than 10 are combined into 'Other'.

        Parameters:
        filter_columns (list of str): List of column names to apply filtering. If None, no filtering is applied.
        """
        # Get all categorical columns
        all_categorical_columns = self.df.select_dtypes(include=["object", "category"]).columns
        
        # Count plots for all categorical columns
        for col in all_categorical_columns:
            if filter_columns and col in filter_columns:
                # Replace categories with count less than 30 with 'Other' for specified columns
                counts = self.df[col].value_counts()
                small_counts = counts[counts < 30].index
                self.df[col] = self.df[col].replace(small_counts, 'Other')
                
                # Filter out the 'Other' category for the specified columns
                filtered_df = self.df[self.df[col] != 'Other']
            else:
                # No filtering applied, use original DataFrame
                filtered_df = self.df
            
            # Plot the frequency distribution
            plt.figure(figsize=(8, 6))
            sns.countplot(y=col, data=filtered_df, order=filtered_df[col].value_counts().index)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.show()

    def pie_charts_for_categorical(self):
        """
        Create pie charts for categorical columns in the DataFrame that have fewer than 10 unique values.

        Parameters:
        df (pd.DataFrame): The DataFrame with categorical columns.
        """
        # Get categorical columns with fewer than 10 unique values
        categorical_columns = self.df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_columns:
            if self.df[col].nunique() < 10:
                # Count occurrences of each category
                counts = self.df[col].value_counts()
                
                # Plot pie chart
                plt.figure(figsize=(8, 8))
                plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
                plt.title(f"Pie Chart of {col}")
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
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
    
    def clean_nan(self, column, fill_value=None):
        """
        Clean a specific column in the dataset by:
        - Removing rows where more than 50% of values in the specified column are missing.
        - Filling missing values in the remaining rows of the specified column with the given fill value.

        Parameters:
        -----------
        column : str
            The column to clean for NaN values.
        fill_value : Any
            The value used to fill missing data in rows with less than 50% missing.

        Conclusion:
        -----------
        This method allows column-specific cleaning of NaN values,
        helping tailor the cleaning process to each column's characteristics.
        """
        # Calculate the percentage of missing values in the specified column
        missing_percent_column = self.df[column].isnull().mean()

        if missing_percent_column > 0.5:
            # Remove rows where more than 50% of values in the specified column are missing
            self.df = self.df[self.df[column].notna()]
            print(f"Rows with more than 50% missing in '{column}' have been removed.")
        else:
            # Fill remaining NaN values in the specified column with the given fill_value
            fill_value = self.df[column].mode()[0] if fill_value is None else fill_value
            self.df[column] = self.df[column].fillna(fill_value)
            print(f"Missing values in '{column}' have been filled with {fill_value}.")