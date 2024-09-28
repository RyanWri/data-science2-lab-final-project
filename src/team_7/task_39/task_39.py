# %%
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import random
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA , TruncatedSVD
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

random.seed(42)
#####################################################################
# %%
# ### Function for Transition Plot (Before-and-After Reduction) ###
# This function creates "Before and After" transition plots for better visualization of dimensionality reduction.
def plot_before_after(original_data, reduced_data, title, sample_size=100):
    # Randomly sample a subset of the data points to avoid clutter
    sample_indices = np.random.choice(len(original_data), size=sample_size, replace=False)
    original_sample = original_data[sample_indices]
    reduced_sample = reduced_data[sample_indices]

    plt.figure(figsize=(10, 6))

    # Plot the original high-dimensional data (projected to 2D with PCA)
    plt.scatter(original_sample[:, 0], original_sample[:, 1], alpha=0.5, label="Original High-Dimensional Data", color='blue')

    # Plot the reduced 2D data
    plt.scatter(reduced_sample[:, 0], reduced_sample[:, 1], alpha=0.7, label="Reduced 2D Data", color='red')

    # Draw arrows from the original point to the reduced point
    for i in range(len(original_sample)):
        plt.arrow(original_sample[i, 0], original_sample[i, 1], 
                  reduced_sample[i, 0] - original_sample[i, 0], 
                  reduced_sample[i, 1] - original_sample[i, 1], 
                  alpha=0.4, color='grey', head_width=0.3, head_length=0.5, lw=0.5)
    
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()

def plot_neighbors_before_after(original_data, reduced_data, title, sample_size=100, n_neighbors=5):
    """
    This function visualizes the changes in the neighborhood structure before and after dimensionality reduction.
    For t-SNE and UMAP, local structure and neighbor relationships are more important than global linear structure.

    Parameters:
    - original_data: High-dimensional data
    - reduced_data: 2D data from t-SNE or UMAP
    - title: Title for the plot
    - sample_size: Number of points to sample from the dataset
    - n_neighbors: Number of neighbors to display for each point
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Randomly sample a subset of the data points
    sample_indices = np.random.choice(len(original_data), size=sample_size, replace=False)
    original_sample = original_data[sample_indices]
    reduced_sample = reduced_data[sample_indices]

    # Find nearest neighbors in the original high-dimensional data
    nn_original = NearestNeighbors(n_neighbors=n_neighbors).fit(original_sample)
    distances_orig, indices_orig = nn_original.kneighbors(original_sample)

    # Plot the original data's neighborhood relationships
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(original_sample[:, 0], original_sample[:, 1], alpha=0.5, label="Original High-Dimensional Data", color='blue')

    for i in range(sample_size):
        for neighbor in indices_orig[i]:
            plt.plot([original_sample[i, 0], original_sample[neighbor, 0]], [original_sample[i, 1], original_sample[neighbor, 1]], 
                     'grey', alpha=0.2)
    plt.title(f"{title}: Original High-Dimensional Neighbors")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Find nearest neighbors in the reduced low-dimensional data
    nn_reduced = NearestNeighbors(n_neighbors=n_neighbors).fit(reduced_sample)
    distances_red, indices_red = nn_reduced.kneighbors(reduced_sample)

    # Plot the reduced data's neighborhood relationships
    plt.subplot(1, 2, 2)
    plt.scatter(reduced_sample[:, 0], reduced_sample[:, 1], alpha=0.7, label="Reduced Data", color='red')

    for i in range(sample_size):
        for neighbor in indices_red[i]:
            plt.plot([reduced_sample[i, 0], reduced_sample[neighbor, 0]], [reduced_sample[i, 1], reduced_sample[neighbor, 1]], 
                     'grey', alpha=0.2)
    plt.title(f"{title}: Reduced Low-Dimensional Neighbors")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    plt.tight_layout()
    plt.show()
#####################################################################


# %% 
# ### Loading Dataset ###
# Load the dataset and preprocess date columns to only keep the date part (remove time).
df_path = "F:\\לימודים\\תואר שני\\סמסטר ב\\Data Science 2\\DS2-Final Project\\data-science2-lab-final-project\\src\\team_9\\assets\\erBeforeHospitalization.csv"
er_before_hospitalization2 = pd.read_csv(df_path)

# %%
# Preprocessing
er_before_hospitalization2['Admission_Entry_Date'] = pd.to_datetime(er_before_hospitalization2['Admission_Entry_Date']).dt.date
er_before_hospitalization2['Release_Date'] = pd.to_datetime(er_before_hospitalization2['Release_Date']).dt.date
er_before_hospitalization2['Admission_Entry_Date2'] = pd.to_datetime(er_before_hospitalization2['Admission_Entry_Date2']).dt.date
er_before_hospitalization2['Release_Date2'] = pd.to_datetime(er_before_hospitalization2['Release_Date2']).dt.date

# %%
# Handle missing values
er_before_hospitalization2_cleaned = er_before_hospitalization2.dropna()

# %% 
# ### Ordinal Encoding of Categorical Variables ###
# Identify non-numeric columns (categorical variables) and apply Ordinal Encoding to transform them into numerical form.
non_numeric_columns = er_before_hospitalization2_cleaned.select_dtypes(include=['object']).columns

# Apply Ordinal Encoding for all non-numeric columns
ordinal_encoder = OrdinalEncoder()

# Fit and transform the non-numeric columns using OrdinalEncoder
er_before_hospitalization2_cleaned[non_numeric_columns] = ordinal_encoder.fit_transform(er_before_hospitalization2_cleaned[non_numeric_columns])

# %%
# Extract numeric columns, including the encoded ones
numeric_columns = er_before_hospitalization2_cleaned.select_dtypes(include=['int64', 'float64']).columns

# %%
# ### Standardizing the Data ###
# Standardize numeric columns to make sure each feature contributes equally to the dimensionality reduction process.
scaler = StandardScaler()
er_before_hospitalization2_scaled = pd.DataFrame(scaler.fit_transform(er_before_hospitalization2_cleaned[numeric_columns]), columns=numeric_columns)

# %%
# ### Plotting Original Data ###
# This pairplot visualizes the original high-dimensional data using a subset of the first few numeric features.
sns.pairplot(er_before_hospitalization2_scaled[numeric_columns[:5]], diag_kind="kde")
plt.suptitle("Original High-Dimensional Data (Pairplot of Selected Features)", y=1.02)
plt.show()

# %%
# Apply PCA to the data
pca = PCA(n_components=2)
er_pca = pca.fit_transform(er_before_hospitalization2_scaled)

# %%
# Explained Variance: Higher variance means better information retention.
explained_variance = pca.explained_variance_ratio_.sum()
# Display the explained variance as HTML
display(HTML(f"<h3>Explained Variance by PCA: {explained_variance * 100:.2f}%</h3>"))

# %%
# Plot PCA Results
plt.figure(figsize=(8,6))
plt.scatter(er_pca[:, 0], er_pca[:, 1], alpha=0.6)
plt.title('PCA Result (2D Projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# %% 
# Apply PCA
pca = PCA(n_components=2)
er_pca = pca.fit_transform(er_before_hospitalization2_scaled)
# Visualize neighborhood relationships for PCA
plot_neighbors_before_after(er_before_hospitalization2_scaled.values[:, :2], er_pca, "PCA", sample_size=150, n_neighbors=5)
# %%
# Visualize Before and After for PCA
plot_before_after(er_before_hospitalization2_scaled.values[:, :2], er_pca, "PCA: Before and After Dimensionality Reduction", sample_size=150)

# %%
# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
er_tsne = tsne.fit_transform(er_before_hospitalization2_scaled)

# %%
# Plot t-SNE Results
plt.figure(figsize=(8,6))
plt.scatter(er_tsne[:, 0], er_tsne[:, 1], alpha=0.6)
plt.title('t-SNE Result (2D Projection)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
# %% 
# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
er_tsne = tsne.fit_transform(er_before_hospitalization2_scaled)

# Visualize neighborhood relationships for t-SNE
plot_neighbors_before_after(er_before_hospitalization2_scaled.values[:, :2], er_tsne, "t-SNE", sample_size=150, n_neighbors=5)
# %%
# Visualize Before and After for t-SNE
plot_before_after(er_before_hospitalization2_scaled.values[:, :2], er_tsne, "t-SNE: Before and After Dimensionality Reduction", sample_size=150)

# %%

# Apply UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
er_umap = umap_model.fit_transform(er_before_hospitalization2_scaled)

# %%
# Plot UMAP Results
plt.figure(figsize=(8,6))
plt.scatter(er_umap[:, 0], er_umap[:, 1], alpha=0.6)
plt.title('UMAP Result (2D Projection)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
# %% 
# Apply UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
er_umap = umap_model.fit_transform(er_before_hospitalization2_scaled)

# Visualize neighborhood relationships for UMAP
plot_neighbors_before_after(er_before_hospitalization2_scaled.values[:, :2], er_umap, "UMAP", sample_size=150, n_neighbors=5)
# %%
# Visualize Before and After for UMAP
plot_before_after(er_before_hospitalization2_scaled.values[:, :2], er_umap, "UMAP: Before and After Dimensionality Reduction", sample_size=150)

# %%
# Apply SVD
# Apply SVD to the data
svd_model = TruncatedSVD(n_components=2, random_state=42)
er_svd = svd_model.fit_transform(er_before_hospitalization2_scaled)

# %%
# Plot SVD Results
plt.figure(figsize=(8,6))
plt.scatter(er_svd[:, 0], er_svd[:, 1], alpha=0.6)
plt.title('SVD Result (2D Projection)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# %% 
# Visualize neighborhood relationships for SVD
plot_neighbors_before_after(er_before_hospitalization2_scaled.values[:, :2], er_svd, "SVD", sample_size=150, n_neighbors=5)

# %%
# Visualize Before and After for SVD
plot_before_after(er_before_hospitalization2_scaled.values[:, :2], er_svd, "SVD: Before and After Dimensionality Reduction", sample_size=150)

# %% 

# Apply KMeans for clustering and evaluate using Silhouette Score
kmeans = KMeans(n_clusters=3, random_state=42)

# %%
# Silhouette Score for PCA
labels_pca = kmeans.fit_predict(er_pca)
silhouette_pca = silhouette_score(er_pca, labels_pca)
print(f"Silhouette Score for PCA: {silhouette_pca:.5f}")

# %%
# Silhouette Score for t-SNE
labels_tsne = kmeans.fit_predict(er_tsne)
silhouette_tsne = silhouette_score(er_tsne, labels_tsne)
print(f"Silhouette Score for t-SNE: {silhouette_tsne:.5f}")

# %%
# Silhouette Score for UMAP
labels_umap = kmeans.fit_predict(er_umap)
silhouette_umap = silhouette_score(er_umap, labels_umap)
print(f"Silhouette Score for UMAP: {silhouette_umap:.5f}")

# %%
# Silhouette Score for SVD
labels_svd = kmeans.fit_predict(er_svd)
silhouette_svd = silhouette_score(er_svd, labels_svd)
print(f"Silhouette Score for SVD: {silhouette_svd:.5f}")

# %%
