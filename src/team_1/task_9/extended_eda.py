import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from datetime import timedelta
from IPython.display import display
import re
from statsmodels.tsa.seasonal import seasonal_decompose


class ExtendedEDA:
    """
    A class to perform basic Exploratory Data Analysis (EDA) on any dataset.

    Attributes:
    -----------
    df : pd.DataFrame
        The dataset to analyze.
    """

    def __init__(self, df: pd.DataFrame, translation_file=None, dtypes=None):
        """
        Initialize with the DataFrame, rename columns using a translation file,
        and set the correct data types (if provided).

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to perform EDA on.
        translation_file : str, optional
            Path to the JSON file containing column name translations.
        dtypes : dict, optional
            A dictionary specifying the desired data types for the columns.
        """
        self.df = df.copy()  # Use a copy to avoid modifying the original data

        # Rename columns using translation file if provided
        if translation_file:
            with open(translation_file, "r", encoding="utf-8") as file:
                translations = json.load(file)
            self.df = self.df.rename(columns=translations)

        # Set column data types if specified
        if dtypes:
            self.df = self.df.astype(dtypes)

    def head(self, n=10):
        return self.df.head(n)

    def copy_df(self):
        return self.df.copy()
    def show_info(self):
        """Display basic information about the dataset."""
        print("Dataset Information:")
        display(self.df.info())

    def find_value_in_column(self, column, value):
        if column not in self.df.columns:
            print("Column not found")
        elif value not in self.df[column].values:
            print("Value not found")
        else:
            return self.df[self.df[column] == value]

    def creat_admission_number_column(self,patient_id='Patient',admission_date_column='Admission_Entry_Date', release_date_column='Release_Date'):
        self.df[f'{admission_date_column}_only'] = pd.to_datetime(self.df[admission_date_column]).dt.date
        self.df[f'{release_date_column}_only'] = pd.to_datetime(self.df[release_date_column]).dt.date
        self.df = self.df.sort_values(by=[patient_id, f'{admission_date_column}_only'])
        temp_df = pd.DataFrame()
        temp_df['admission_number'] = self.df.groupby(patient_id)[f'{admission_date_column}_only'].rank(
            method='dense').astype(int)
        self.df = pd.concat([self.df, temp_df], axis=1)

    def print_value_in_column(self, column, value):
        if column in self.df.columns:
            filtered_df = self.df[self.df[column] == value]
            display(filtered_df)
        else:
            print(
                f"Column '{column}' does not exist in the DataFrame. Available columns are: {self.df.columns.tolist()}")


    def replace_strings(self, replacement_dict):
        self.df.replace(replacement_dict, inplace=True)


    def show_summary_statistics(self):
        """Display summary statistics for numerical and categorical columns."""
        print("Summary Statistics (Numerical):")
        display(self.df.describe())
        print("\n")
        print("Summary Statistics (Categorical):")
        display(self.df.describe(include=["object", "category"]))
        print("\n")

    def fix_negative_duration(self):
        # Find rows with negative hospitalization duration
        negative_duration_indices = self.df[self.df['hospitalization_duration'] < 0].index
        for i in negative_duration_indices:
            # Swap Admission_Entry_Date and Release_Date
            self.df.at[i, 'Admission_Entry_Date'], self.df.at[i, 'Release_Date'] = self.df.at[i, 'Release_Date'], self.df.at[i, 'Admission_Entry_Date']

            # Change hospitalization_duration to positive
            self.df.at[i, 'hospitalization_duration'] = abs(self.df.at[i, 'hospitalization_duration'])

    def convert_comma_separated_values_to_list(self, column):
        self.df[column] = self.df[column].astype(str)
        self.df[f'{column}_list'] = self.df[column].apply(lambda x: [item.strip() for item in x.split(',')])

    def change_data_type(self, column, data_type):
        dtype_before = self.df[column].dtype
        print(f'The initial data type of column {column} is: {dtype_before}')
        self.df[column] = self.df[column].astype(data_type)
        dtype_after = self.df[column].dtype
        if dtype_before == dtype_after:
            print('The data type did not change.')
            print("\n")
        else:
            print(f'The data type of column {column} has been changed to: {dtype_after}')
            print("\n")


    def check_missing_values(self):
        """Display the number of missing values in each column."""
        missing_values = self.df.isnull().sum()
        if missing_values.sum() == 0:
            print(f"No missing values have been found.")
        else:
            print("Missing Values in Each Column:")
            display(missing_values[missing_values > 0])
            print("\n")

    def show_unique_values(self, column=None):
        """Display the unique values and their counts for each column."""
        pd.set_option('display.max_rows', None)
        if column is None:
            for col in self.df.columns:
                print(f"Unique values in '{col}':")
                display(self.df[col].value_counts().to_frame())
                print("\n" + "=" * 40 + "\n")
        else:
            print(f"Unique values in '{column}':")
            print(self.df[column].value_counts().to_frame())
            print("\n" + "=" * 40 + "\n")

    def total_hospitalization_duration_and_department_count(self):
        print('Total hospitalization duration per department id:')
        display(self.df.groupby('department_id')['hospitalization_duration'].sum())
        print('\n')
        total_duration_per_department = self.df.groupby('department_id')['hospitalization_duration'].sum()
        count_per_department = self.df['department_id'].value_counts().sort_index()

        # Use .codes to convert categorical index to integer codes for plotting
        codes = total_duration_per_department.index.codes
        print(f'Department id counts:')
        display(count_per_department)
        print('\n')
        # Create the plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot total hospitalization duration
        ax1.bar(codes - 0.2, total_duration_per_department.values, width=0.4,
                color='skyblue', label='Total Duration', edgecolor='black')
        ax1.set_xlabel('Department ID')
        ax1.set_ylabel('Total Hospitalization Duration (days)', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax1.set_title('Total Hospitalization Duration and Department Count')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Create a second y-axis to plot the department count as bars
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.bar(codes + 0.2, count_per_department.values, width=0.4, color='orange',
                label='Department Count', edgecolor='black')
        ax2.set_ylabel('Department Count', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Adding a legend with manual positioning
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)

        plt.tight_layout()
        plt.show()

    def check_repeated_values(self, column):
        value_counts = self.df[column].value_counts()
        # Filter for values that are repeated (appear more than once)
        repeated_values = value_counts[value_counts > 1]
        # Print the repeated values and their counts
        if len(repeated_values) > 0:
            print(f"Repeated values and their counts in column {column}")
            display(repeated_values)
        else:
            print("No repeated values in column '{}'".format(column))

    def column_list(self):
        return self.df.columns

    def find_hebrew_in_column(self, column):
        def contains_hebrew(text):
            # Compile regex for Hebrew character range
            hebrew_pattern = re.compile('[\u0590-\u05FF]')
            # Search text for Hebrew characters
            if pd.isnull(text):  # Handle potential NaN values
                return False
            return bool(hebrew_pattern.search(text))

        df = pd.DataFrame()
        # Apply the function to the DataFrame's column
        df['contains_hebrew'] = self.df[column].apply(contains_hebrew)

        # Filter rows that contain Hebrew text
        hebrew_rows = self.df[df['contains_hebrew']]
        if len(hebrew_rows) == 0:
            print(f"No hebrew data found in column {column}.")
        else:
            print("Rows containing Hebrew text:")
            display(hebrew_rows)

    def plot_distribution(self, column):
        """
        Plot the distribution of a single column, whether numerical or categorical.

        Parameters:
        -----------
        column : str
            The column name to visualize.
        """
        if self.df[column].dtype in ["float64", "int64"]:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=self.df[column], discrete=True)
            plt.xlabel(column, )
            plt.title(f"Distribution of {column}")
            plt.show()
        else:
            plt.figure(figsize=(8, 6))
            sns.countplot(y=column, data=self.df)
            plt.title(f"Count of Categories in {column}")
            plt.show()

    def plot_pairwise_distributions(self):
        """Plot pairwise distributions of numerical columns using a pairplot."""
        sns.pairplot(self.df)
        plt.show()

    def plot_correlation_matrix(self):
        """Plot a heatmap of the correlation matrix for numerical columns."""
        corr_matrix = self.df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def detect_outliers(self, column):
        """
        Display a boxplot to visualize outliers in a numerical column.

        Parameters:
        -----------
        column : str
            The column name to check for outliers.
        """
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=self.df[column])
        plt.title(f"Boxplot of {column} (Outliers Detection)")
        plt.show()

    def feature_correlation_with_target(self, target_column):
        """
        Show the correlation of each feature with the target variable.

        Parameters:
        -----------
        target_column : str
            The target variable to correlate with.
        """
        correlations = self.df.corr()[target_column].sort_values(ascending=False)
        print(f"Correlations with {target_column}:")
        print(correlations)
        print("\n")

    # New Methods

    def plot_histogram(self, column):
        """
        Plot a histogram for a numerical column.

        Parameters:
        -----------
        column : str
            The column to plot the histogram for.
        """
        plt.figure(figsize=(8, 6))
        self.df[column].hist(bins=20, edgecolor="black")
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def plot_categorical_frequency(self, column):
        """
        Plot a bar chart showing the frequency of each category in a categorical column.

        Parameters:
        -----------
        column : str
            The categorical column to plot.
        """
        plt.figure(figsize=(8, 6))
        sns.countplot(x=column, data=self.df)
        plt.title(f"Frequency of Categories in {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.show()

    def plot_kde_numerical(self, column):
        """
        Plot a Kernel Density Estimate (KDE) for a numerical column.

        Parameters:
        -----------
        column : str
            The column for which to plot the KDE.
        """
        plt.figure(figsize=(8, 6))
        sns.kdeplot(self.df[column], fill=True)
        plt.title(f"Kernel Density Estimate (KDE) of {column}")
        plt.xlabel(column)
        plt.show()

    def plot_numeric_vs_categorical(self, num_column, cat_column):
        """
        Plot the relationship between a numerical and a categorical column using a boxplot.

        Parameters:
        -----------
        num_column : str
            The numerical column to plot.
        cat_column : str
            The categorical column to group by.
        """
        categories_num = self.df[cat_column].nunique()
        if categories_num > 10:
            plt.figure(figsize=(18, 6))
        else:
            plt.figure(figsize=(8, 6))
        sns.boxplot(x=cat_column, y=num_column, data=self.df)
        plt.title(f"{num_column} vs {cat_column}")
        plt.xlabel(cat_column)
        plt.ylabel(num_column)
        plt.xticks(rotation=45)
        plt.show()


    def fill_release_doctor_code(self):
        """
        Replaces the nulls in the release doctor code with the most common
        release doctor code in the department at the same day of release.
        If no other release doctor code exist at the same day it takes from
        the most common in the days before and after, increasing the number
        of days until all null are filled.
        """
        # Convert Release_Date to datetime for consistency and correct processing
        self.df['Release_Date_only'] = pd.to_datetime(self.df['Release_Date']).dt.date
        i = 0
        while self.df['release_doctor_code'].isnull().sum() > 0:
            def fill_nan_codes(df, department, current_date):
                # Collect potential doctor codes from the current, previous, and next day
                dates_to_check = [current_date, current_date - timedelta(days=i), current_date + timedelta(days=i)]
                codes = self.df[(self.df['Release_Date_only'].isin(dates_to_check)) & (self.df['department_id'] == department)]['release_doctor_code']

                # Filter out NaN values
                valid_codes = codes.dropna()

                if not valid_codes.empty:
                    # Return the most common code among these days
                    return Counter(valid_codes).most_common(1)[0][0]

                return None  # Default to None or any other placeholder you prefer if no code is available

            # Iterate over the DataFrame to fill NaN values
            for index, row in self.df.iterrows():
                if pd.isna(row['release_doctor_code']):
                    most_common_code = fill_nan_codes(self.df, row['department_id'], row['Release_Date_only'])
                    if most_common_code is not None:
                       self.df.at[index, 'release_doctor_code'] = most_common_code
            i = i + 1
        del self.df['Release_Date_only']

    def fix_height_weight(self):
        def fill_nan_weight(df, age, Gender):
            weight = df[(df['age'] == age) & (df['Gender'] == Gender)]['weight']
            # Filter out NaN values
            valid_weight = weight.dropna()
            if not valid_weight.empty:
                return valid_weight.mean()
            else:
                j = 1
                while valid_weight.empty:
                    weight = df[(df['age'] == age - j) & (df['Gender'] == Gender)]['weight']
                    valid_weight = weight.dropna()
                    j = j + 1
                if not valid_weight.empty:
                    return valid_weight.mean()
                return None
        def fill_nan_height(df, age, Gender):
            height = df[(df['age'] == age) & (df['Gender'] == Gender)]['height']
            # Filter out NaN values
            valid_height = height.dropna()
            if not valid_height.empty:
                return valid_height.mean()
            else:
                i = 1
                while valid_height.empty:
                    height = df[(df['age'] == age - i) & (df['Gender'] == Gender)]['height']
                    valid_height = height.dropna()
                    i = i + 1
                if not valid_height.empty:
                    return valid_height.mean()
                return None
        for index, row in self.df.iterrows():
            if pd.notnull(row['height']) and (1 < self.df.at[index, 'height'] < 2.5):
                print(f'Due to height between 1 to 2.5, changed the height from {row["height"]}')
                self.df.at[index, 'height'] = row['height'] * 100
                print(f' to {self.df.at[index, "height"]}')
            if pd.notnull(row['height']) and pd.notnull(row['weight']):
                if (40 < self.df.at[index, 'height'] < 95) and (30 < self.df.at[index, 'weight'] < 120):
                    print(f'Due to height between 40 to 95 and weight between 30 and 120, changed the height from {row["height"]}')
                    self.df.at[index, 'height'] = row['height'] + 100
                    print(f' to {self.df.at[index, "height"]}')
                if (self.df.at[index, 'height'] < self.df.at[index, 'weight']) and (120 < self.df.at[index, 'weight'] < 190):
                    temp_weight = row['weight']
                    temp_height = row['height']
                    print(f'Due to height larger from weight and weight larger from 120, Changed the weight from {row["weight"]}')
                    self.df.at[index, 'weight'] = temp_height
                    print(f' to {self.df.at[index, "weight"]}')
                    print(f'Due to height larger from weight and weight larger from 120, Changed the height from {row["height"]}')
                    self.df.at[index, 'height'] = temp_weight
                    print(f' to {self.df.at[index, "height"]}')
            if pd.isna(row['weight']):
                mean_weight = fill_nan_weight(self.df, row['age'], row['Gender'])
                self.df.at[index, 'weight'] = round(mean_weight, 1)
                print(f'Fill weight null to mean of same age:{row["age"]} and gender: {row["Gender"]}: {self.df.at[index, "weight"]}')
            if pd.isna(row['height']):
                mean_height = fill_nan_height(self.df, row['age'], row['Gender'])
                self.df.at[index, 'height'] = round(mean_height, 1)
                print(f'Fill height null to mean of same age:{row["age"]} and gender: {row["Gender"]}: {self.df.at[index, "height"]}')





    def fill_bmi(self):
        def fill_nan_bmi(df, age, Gender):
            bmi = df[(df['age'] == age) & (df['Gender'] == Gender)]['BMI']
            # Filter out NaN values
            valid_bmi = bmi.dropna()
            if not valid_bmi.empty:
                return valid_bmi.mean()
            else:
                i = 1
                while valid_bmi.empty:
                    bmi = df[(df['age'] == age - i) & (df['Gender'] == Gender)]['height']
                    valid_bmi = bmi.dropna()
                    i = i + 1
                if not valid_bmi.empty:
                    return valid_bmi.mean()
                return None
        for index, row in self.df.iterrows():
            if pd.notnull(row['BMI']) and (self.df.at[index, 'BMI'] > 55):
                print(f'{row["Patient"]}BMI changed from a over 55 value of: {row["BMI"]}')
                if pd.notnull(row['height']) and pd.notnull(row['weight']):
                    self.df.at[index, 'BMI'] = round(self.df.at[index, 'weight'] / (((self.df.at[index, 'height']/100))*((self.df.at[index, 'height'])/100)), 1)
                    print(f'NaN BMI calculated according to weight and height values to: {self.df.at[index, "BMI"]}')
                if pd.isna(row['height']) or pd.isna(row['weight']):
                    mean_bmi = fill_nan_bmi(self.df, row['age'], row['Gender'])
                    self.df.at[index, 'BMI'] = round(mean_bmi, 1)
                    print(f'NaN BMI replaced by mean BMI of same age:{row["age"]} and same gender:{row["Gender"]} to: {self.df.at[index, "BMI"]}')
                print(f' to {self.df.at[index, "BMI"]}')
            if pd.isna(row['BMI']):
                if pd.notnull(row['height']) and pd.notnull(row['weight']):
                    self.df.at[index, 'BMI'] = round(self.df.at[index, 'weight'] / (((self.df.at[index, 'height']/100))*((self.df.at[index, 'height'])/100)), 1)
                    print(f'NaN BMI calculated according to weight and height values to: {self.df.at[index, "BMI"]}')
                if pd.isna(row['height']) or pd.isna(row['weight']):
                    mean_bmi = fill_nan_bmi(self.df, row['age'], row['Gender'])
                    self.df.at[index, 'BMI'] = round(mean_bmi, 1)
                    print(f'NaN BMI replaced by mean BMI of same age:{row["age"]} and same gender:{row["Gender"]} to: {self.df.at[index, "BMI"]}')
    def fix_nans_by_common_age_and_gender(self, column):
        i = 0
        while self.df[column].isnull().sum() > 0:
            def fill_nan_values(df, age, gender):
                ages_to_check = [age, age - i, age + i]
                values = df[(df['age'].isin(ages_to_check)) & (df['Gender'] == gender)][column]
                # Filter out NaN values
                valid_values = values.dropna()
                if not valid_values.empty:
                    # Return the most common code among these days
                    return Counter(valid_values).most_common(1)[0][0]

                return None  # Default to None or any other placeholder you prefer if no code is available

            # Iterate over the DataFrame to fill NaN values
            for index, row in self.df.iterrows():
                if pd.isna(row[column]):
                    print(f'{column} null replaced with:')
                    most_common_values = fill_nan_values(self.df, row['age'], row['Gender'])
                    if most_common_values is not None:
                        self.df.at[index, column] = most_common_values
                        print(self.df.at[index, column])
            i = i + 1

    def fix_medications_nans(self):
        # Convert each entry to a list of integers, handling non-string entries
        def convert_to_list(entry):
            if entry is None or (isinstance(entry, float) and pd.isna(entry)):  # Check if the entry itself is NaN
                return []  # Return an empty list
            elif isinstance(entry, str):  # Check if the entry is a string
                # Split the string by commas and convert to integer
                return list(map(int, entry.split(',')))
            elif isinstance(entry, (int, float)):  # Handle numeric single entries
                return [int(entry)]  # Create a list with the single number
            elif isinstance(entry, list):  # Handle list entries directly
                return entry  # Assume it's already in the correct format
            else:
                # If an unexpected type is encountered, raise an error
                raise ValueError(f"Unexpected entry type: {type(entry)}")
        i = 1
        while self.df['medications'].isnull().sum() > 0:
            def fill_nan_values(df, age, gender):
                ages_to_check = [age, age - i, age + i]
                values = df[(df['age'].isin(ages_to_check)) & (df['Gender'] == gender)]['medications'].dropna().apply(convert_to_list)
                flattened_values = [item for sublist in values for item in sublist]
                if flattened_values:
                    # Return the most common code among these days
                    common_values = Counter(flattened_values).most_common(3)
                    return [value for value, count in common_values]  # Extract only the values
                return None
            # Iterate over the DataFrame to fill NaN values
            for index, row in self.df.iterrows():
                if pd.isna(row['medications']):
                    print(f'{"medications"} null replaced with:')
                    most_common_values = fill_nan_values(self.df, row['age'], row['Gender'])
                    if most_common_values is not None:
                        self.df.at[index, 'medications'] = most_common_values
                        print(f'{self.df.at[index, "medications"]} from ages: {row["age"]-i}, {row["age"]}, {row["age"]+i} and from gender: {row["Gender"]}')
            i = i + 1

    def plot_numeric_over_time(self, num_column, time_column):
        """
        Plot a line chart to observe how a numerical column changes over time.

        Parameters:
        -----------
        num_column : str
            The numerical column to plot.
        time_column : str
            The time column to plot along the x-axis.
        """
        plt.figure(figsize=(18, 6))
        sns.lineplot(x=self.df[time_column], y=self.df[num_column])
        plt.title(f"{num_column} Over Time")
        plt.xlabel(time_column)
        plt.ylabel(num_column)
        plt.xticks(rotation=45)
        plt.show()

    def check_seasonality_and_cyclical_patterns(self, datetime_column='Admission_Entry_Date_only', numeric_data='hospitalization_duration'):
        """Display seasonality and cyclical patterns for the specified numeric data against a datetime column."""
        seasonality_df = self.df[[datetime_column, numeric_data]].copy()

        seasonality_df[datetime_column] = pd.to_datetime(seasonality_df[datetime_column])
        seasonality_df.sort_values(by=datetime_column, inplace=True)
        seasonality_df.set_index(datetime_column, inplace=True)
        # Decompose the time series
        decomposition = seasonal_decompose(
            seasonality_df[numeric_data].dropna(), model="additive", period=365
        )

        # Plot decomposition results
        plt.figure(figsize=(12, 8))
        plt.subplot(411)
        plt.plot(decomposition.observed, label="Observed")
        plt.legend(loc="upper right")
        plt.subplot(412)
        plt.plot(decomposition.trend, label="Trend")
        plt.legend(loc="upper right")
        plt.subplot(413)
        plt.plot(decomposition.seasonal, label="Seasonal")
        plt.legend(loc="upper right")
        plt.subplot(414)
        plt.plot(decomposition.resid, label="Residual")
        plt.legend(loc="upper right")
        plt.show()

    def drop_column(self, column_name):
        print(f'Dataframe shape is: {self.df.shape}')
        self.df = self.df.drop(column_name, axis=1)
        print(f'The column {column_name} was dropped from the dataframe')
        print(f'New dataframe shape is: {self.df.shape}')
        print("\n" + "=" * 40 + "\n")


    def remove_rows_with_nan(self, column_list, percentage_threshold):
        # Calculate the minimum number of non-NaN values required in the specified columns
        num_columns = len(column_list)
        thresh = int((100 - percentage_threshold) / 100 * num_columns)

        # Drop rows if the number of non-NaN values in specified columns is less than thresh
        self.df = self.df.dropna(subset=column_list, thresh=thresh)


    def print_nan_in_column(self, column, ):
        if self.df[column].isnull().sum() == 0:
            print('No NaNs found')
        else:
            return self.df[self.df[column].isna()]

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

        if missing_percent_column > 0.2:
            # Remove rows where more than 50% of values in the specified column are missing
            self.df = self.df[self.df[column].notna()]
            print(f"Rows with more than 20% missing in '{column}' have been removed.")
        else:
            # Fill remaining NaN values in the specified column with the given fill_value
            fill_value = self.df[column].mode()[0] if fill_value is None else fill_value
            self.df[column] = self.df[column].fillna(fill_value)
            print(f"Missing values in '{column}' have been filled with {fill_value}.")

    def export_to_csv(self, file_path):
        """
        Export the cleaned DataFrame to a CSV file.

        Parameters:
        -----------
        file_path : str
            The path where the CSV file will be saved.

        Conclusion:
        -----------
        This method allows you to easily export the cleaned dataset to a CSV file
        so that it can be shared or used outside the class.
        """
        self.df.to_csv(file_path, index=False)
        print(f"Data exported successfully to {file_path}")
