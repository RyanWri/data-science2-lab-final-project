import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('task_29_dataset_final.xlsx')

# Select the most common medications (binary columns)
medication_columns = ['1183', '2188', '630', '2791', '6737', '2624', '5913', 
                      '2606', '1443', '4437', '6718', '4328', '2043', '6720', 
                      '37', '3381', '643', '3459', '577', '4677']

# Target categories
duration_categories = ['Duration_Category_16', 'Duration_Category_17', 'Duration_Category_18']

# Step 1: Summary Statistics for the Most Common Medications (printed separately)
def summary_statistics(df, medication_columns):
    print("\nSummary Statistics for Medications:\n")
    stats = df[medication_columns].describe().T
    print(stats)
    return stats

# Step 2: Medication Usage Distribution (Bar Plots) in Subplots
def plot_medication_distribution_subplots(df, medication_columns):
    n_cols = 5  # Number of columns in subplots
    n_rows = len(medication_columns) // n_cols + (1 if len(medication_columns) % n_cols > 0 else 0)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))
    fig.tight_layout(pad=3.0)
    axes = axes.flatten()

    for i, med in enumerate(medication_columns):
        sns.countplot(x=med, data=df, palette="Set2", ax=axes[i])
        axes[i].set_title(f'Medication {med}')
        axes[i].set_ylabel('Count')
        axes[i].set_xlabel(f'Usage (0 = No, 1 = Yes)')
    
    # Remove empty subplots if there are any
    for ax in axes[len(medication_columns):]:
        fig.delaxes(ax)

    # Set title with enough space to avoid overlap
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Medication Usage Distribution', fontsize=16, y=1.02)
    plt.show()

# Step 3: Stacked Bar Plots in Subplots for Duration Categories
def plot_stacked_distribution_subplots(df, medication_columns, target_category):
    n_cols = 5  # Number of columns in subplots
    n_rows = len(medication_columns) // n_cols + (1 if len(medication_columns) % n_cols > 0 else 0)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))
    fig.tight_layout(pad=3.0)
    axes = axes.flatten()

    for i, med in enumerate(medication_columns):
        medication_df = pd.crosstab(df[med], df[target_category], normalize='index')
        medication_df.plot(kind='bar', stacked=True, ax=axes[i], color=['skyblue', 'orange', 'green'], legend=False)
        axes[i].set_title(f'Medication {med}')
        axes[i].set_ylabel('Proportion')
        axes[i].set_xlabel(f'Usage (0 = No, 1 = Yes)')
    
    # Remove empty subplots if there are any
    for ax in axes[len(medication_columns):]:
        fig.delaxes(ax)

    # Set title with enough space to avoid overlap
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f'Stacked Distribution for {target_category}', fontsize=16, y=1.02)
    plt.show()

# Step 4: Correlation Heatmap for Medications and Duration Categories
def plot_correlation_matrix(df, medication_columns, duration_categories):
    plt.figure(figsize=(12, 8))
    corr_df = df[medication_columns + duration_categories].corr()
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix: Medications and Duration Categories", pad=20)
    plt.show()

# Running EDA

# 1. Summary Statistics
medication_stats = summary_statistics(df, medication_columns)

# 2. Distribution of Medications (Bar Plots in Subplots)
plot_medication_distribution_subplots(df, medication_columns)

# 3. Stacked Bar Plots for Medications by Each Duration Category
for target_category in duration_categories:
    plot_stacked_distribution_subplots(df, medication_columns, target_category)

# 4. Correlation Matrix for Medications and Duration Categories
plot_correlation_matrix(df, medication_columns, duration_categories)
