import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('hospitalization2_translated_clean.csv')

# Handling Missing Data: Drop rows with any missing values.
# Consider using other imputation methods depending on your data analysis needs.
df.dropna(inplace=True)

# Encoding categorical data: Convert categorical columns to numeric codes
for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = pd.Categorical(df[col]).codes

# Standardize the data: Essential for PCA, t-SNE, and LDA to perform correctly
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df)

# Apply PCA: Reducing dimensions to 2 for easy visualization and further analysis
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)
print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)

# Visualizing PCA results: Scatter plot of the first two principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', data=principal_df)
plt.title('PCA Results: Scatter Plot of Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Apply t-SNE: Another technique for dimensionality reduction focusing on maintaining local structures
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features_scaled)
tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE 1', 'TSNE 2'])
plt.figure(figsize=(10, 8))
sns.scatterplot(x='TSNE 1', y='TSNE 2', data=tsne_df)
plt.title('t-SNE Results: Scatter Plot')
plt.xlabel('TSNE 1')
plt.ylabel('TSNE 2')
plt.show()

# Apply LDA: Assumes a labeled 'target' column is present for supervised dimensionality reduction
if 'target' in df.columns:
    X = df.drop('target', axis=1)
    y = df['target']
    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X, y)
    lda_df = pd.DataFrame(data=X_lda, columns=['LDA 1'])
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='LDA 1', y=[0] * len(lda_df), hue=y, palette='viridis')
    plt.title('LDA Results: Projected Data onto the First Linear Discriminant')
    plt.xlabel('LDA 1')
    plt.yticks([])  # Hide y-ticks as they have no meaning in this context
    plt.show()
