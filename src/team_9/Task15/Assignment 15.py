import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.dates as mdates


# Load data
file_path = 'hospitalization2_translated_clean.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Handling Missing Data
nan_counts = df.isna().sum()
print(nan_counts)
df.dropna(inplace=True)  # Adjust according to how you want to handle missing data

# Define column types dynamically
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe(include='all'))

# Histograms for all numerical data
fig, axes = plt.subplots(nrows=(len(num_cols) + 3) // 4, ncols=4, figsize=(20, 5 * ((len(num_cols) + 3) // 4)))
fig.suptitle('Histograms of Numerical Features')
for ax, col in zip(axes.flat, num_cols):
    ax.hist(df[col], bins=15, color='blue')
    ax.set_title(f'Histogram of {col}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Box plots for numeric columns
fig, axes = plt.subplots(nrows=(len(num_cols) + 3) // 4, ncols=4, figsize=(20, 5 * ((len(num_cols) + 3) // 4)))
fig.suptitle('Box Plots of Numerical Features')
for ax, col in zip(axes.flat, num_cols):
    sns.boxplot(y=df[col], ax=ax, color='lightblue')
    ax.set_title(f'Box Plot of {col}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Bar charts for categorical data
fig, axes = plt.subplots(nrows=(len(cat_cols) + 3) // 4, ncols=4, figsize=(24, 6 * ((len(cat_cols) + 3) // 4)))  # Adjusted figure size
fig.suptitle('Bar Charts of Categorical Features', fontsize=16)

for ax, col in zip(axes.flat, cat_cols):
    series = df[col].value_counts()
    top_categories = series.head(10)  # Only show top 10 categories for clarity
    top_categories.plot(kind='bar', ax=ax)
    ax.set_title(f'Frequency of {col}', fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=45)  # Rotate labels for better fit

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make sure titles and labels don't overlap
plt.show()


# Correlation heatmap for numerical data
if len(num_cols) > 1:
    corr_matrix = df[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

# Select numeric columns for individual clustering
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Normalizing the data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Set up the matplotlib figure
n_cols = 4  # You can adjust this based on your preference
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols  # Calculate required number of rows
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 5, n_rows * 4))
fig.suptitle('Clustering of Each Variable')

# Perform clustering for each column and plot
for ax, col in zip(axes.flat, numeric_cols):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)  # You can adjust the number of clusters
    labels = kmeans.fit_predict(df_scaled[[col]])

    # Create scatter plot
    ax.scatter(df.index, df_scaled[col], c=labels, cmap='viridis', alpha=0.6)
    ax.set_title(f'Clustering of {col}')
    ax.set_xlabel('Index')
    ax.set_ylabel(col)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
