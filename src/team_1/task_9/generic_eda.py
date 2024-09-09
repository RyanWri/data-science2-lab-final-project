import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class GenericEDA:
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

    def show_info(self):
        """Display basic information about the dataset."""
        print("Dataset Information:")
        print(self.df.info())
        print("\n")

    def show_summary_statistics(self):
        """Display summary statistics for numerical and categorical columns."""
        print("Summary Statistics (Numerical):")
        print(self.df.describe())
        print("\n")
        print("Summary Statistics (Categorical):")
        print(self.df.describe(include=["object", "category"]))
        print("\n")

    def check_missing_values(self):
        """Display the number of missing values in each column."""
        missing_values = self.df.isnull().sum()
        print("Missing Values in Each Column:")
        print(missing_values[missing_values > 0])
        print("\n")

    def show_unique_values(self):
        """Display the unique values and their counts for each column."""
        for col in self.df.columns:
            print(f"Unique values in '{col}':")
            print(self.df[col].value_counts())
            print("\n" + "=" * 40 + "\n")

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
            sns.histplot(self.df[column], kde=True)
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
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=cat_column, y=num_column, data=self.df)
        plt.title(f"{num_column} vs {cat_column}")
        plt.xlabel(cat_column)
        plt.ylabel(num_column)
        plt.xticks(rotation=45)
        plt.show()

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
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=self.df[time_column], y=self.df[num_column])
        plt.title(f"{num_column} Over Time")
        plt.xlabel(time_column)
        plt.ylabel(num_column)
        plt.xticks(rotation=45)
        plt.show()

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
