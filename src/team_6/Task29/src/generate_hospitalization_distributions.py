import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def calculate_optimal_bins(data):
    """
    Calculate the optimal number of bins for a histogram using the Freedman-Diaconis rule.

    Args:
        data (pd.Series): Data series to calculate the bins for.

    Returns:
        int: Optimal number of bins for the histogram.
    """
    if len(data) > 1:
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25  # Interquartile range
        bin_width = 2 * iqr * len(data) ** (-1/3)  # Freedman-Diaconis rule
        bin_count = (data.max() - data.min()) / bin_width
        return max(int(bin_count), 10)  # Ensure at least 10 bins
    else:
        return 10  # Default to 10 bins if insufficient data

def plot_histogram(data, column, title, filename):
    """
    Plot and save a histogram for the given data column.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Column name for which the histogram is plotted.
        title (str): Title for the histogram plot.
        filename (str): Filename to save the histogram image.
    """
    bins = calculate_optimal_bins(data[column])
    plt.figure(figsize=(8, 6))
    data[column].plot(kind='hist', bins=bins, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"{title} (Optimal bins: {bins})")
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"{title}: Optimal distribution calculated with {bins} bins.")

def analyze_hospitalizations(file_path, output_folder='pictures'):
    """
    Analyze the number of days between hospitalizations, in the first hospitalization, and in the second hospitalization, then generate histograms.

    Args:
        file_path (str): Path to the Excel file containing the hospitalization data.
        output_folder (str): Folder where to save the histogram images.
    """
    # Load the data
    hospitalization_data = pd.read_excel(file_path, sheet_name='hospitalization1', engine='openpyxl')
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define columns to analyze
    categories = {
        'Days_Between_Hospitalizations': ('Optimal Distribution for Days Between First and Second Hospitalization', 'days_between_hospitalizations.png'),
        'Days_First_Hospitalization': ('Optimal Distribution for Days in First Hospitalization', 'days_in_first_hospitalization.png'),
        'Days_Second_Hospitalization': ('Optimal Distribution for Days in Second Hospitalization', 'days_in_second_hospitalization.png')
    }

    for column, (title, filename) in categories.items():
        if column in hospitalization_data.columns:
            plot_filename = os.path.join(output_folder, filename)
            plot_histogram(hospitalization_data, column, title, plot_filename)
        else:
            print(f"Column '{column}' not found in the data.")

if __name__ == "__main__":
    # Use the file path of the uploaded Excel file
    file_path = r'/Users/liav/Desktop/GIT/DL_Final/task25/data/rehospitalization.xlsx'
    analyze_hospitalizations(file_path)