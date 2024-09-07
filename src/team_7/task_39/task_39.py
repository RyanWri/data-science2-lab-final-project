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
# ### PCA Analysis ###
# PCA is a linear dimensionality reduction method. It reduces the data based on variance, aiming to preserve global structures and linear relationships.
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
# ### Conclusion about PCA ###
# PCA performed well for this data, as indicated by the explained variance and clear cluster separation in the plots. PCA preserves the linear structure and is highly interpretable.

# %%
# ### t-SNE Analysis ###
# t-SNE is a non-linear dimensionality reduction method primarily used for visualization. It preserves local structures and relationships but might not maintain global separation as well.
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

# %%
# ### Conclusion about t-SNE ###
# t-SNE captured local structures well but had a lower Silhouette Score (0.404). It may not have preserved global separation between clusters as well as PCA, making it ideal for detailed visualizations of smaller clusters.

# %%
# ### UMAP Analysis ###
# UMAP is a non-linear method designed to balance local and global structure preservation, making it scalable and useful for visualization and cluster separation.
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
# SVD (Singular Value Decomposition) is a linear dimensionality reduction method similar to PCA.
# It reduces the data based on variance and is often used in large matrix factorization problems.

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
# ### Conclusion about UMAP ###
# UMAP produced a moderate Silhouette Score (0.511), indicating a good balance between local and global structure preservation.
# UMAP is scalable and works well on larger datasets, and its ability to capture both local and global structures makes it useful for clustering tasks.

# %%
# ### Clustering Evaluation using Silhouette Score ###
# The Silhouette Score is used to measure how well the data points are separated into clusters after dimensionality reduction.
# A higher Silhouette Score indicates better clustering and separation between the clusters.

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
# ### Conclusion Based on Silhouette Scores ###
# PCA: Silhouette Score of 0.589 - PCA provided the best clustering performance in this case, suggesting that the data may have a strong linear structure.
# t-SNE: Silhouette Score of 0.404 - t-SNE, while useful for visualizing local structures, struggled with global cluster separation in this case.
# UMAP: Silhouette Score of 0.511 - UMAP provided a balance between local and global structure preservation, with clustering performance better than t-SNE but slightly lower than PCA.

# %%
# ### Final Recommendation on Dimensionality Reduction Method ###
# - **PCA**: Best for preserving global structure and works well for linear relationships. It is computationally efficient and provides interpretable results, making it the top choice when cluster separation is important.
# - **t-SNE**: Best for detailed visualization of local structures and small clusters, though it may not capture global structure as well. Use for visualizing fine details in smaller datasets.
# - **UMAP**: A strong option when you need a balance between local and global structure preservation. UMAP is faster than t-SNE and works well for larger datasets, making it a good alternative for non-linear dimensionality reduction with good clustering quality.

# Ultimately, if your goal is to achieve the best cluster separation, **PCA** is recommended based on the Silhouette Scores in this analysis.
# If you need more detailed visualization of local patterns or non-linear relationships, **t-SNE** or **UMAP** may still be valuable depending on the dataset size and computational constraints.


##############################################################################################
# ## 1. PCA (Principal Component Analysis)
# Observations:

# PCA is a linear dimensionality reduction method that focuses on preserving the global variance of the dataset.
# In the Original High-Dimensional Neighbors plot, the structure is compact and well-connected.
# In the Reduced Low-Dimensional Neighbors plot, PCA maintains some of the neighboring relationships, but the data appears stretched linearly.
# PCA captures global relationships effectively, but the transformation tends to distort local structures, as it cannot handle non-linear relationships.
# Strengths:

# Global variance preservation: PCA captures the major directions in which the data varies, making it suitable for datasets with linear relationships.
# Computational efficiency: PCA is one of the fastest dimensionality reduction methods and can handle large datasets efficiently.
# Interpretability: The output is easy to understand, as each principal component represents a direction of maximal variance.
# Weaknesses:

# Poor handling of non-linear structures: PCA cannot capture non-linear relationships between features.
# Local distortions: Neighboring relationships can become distorted, especially if the data has a complex, non-linear structure.
# Best Used For:

# When you want to understand the global structure or variance in the data and have datasets with linear relationships.
# Large datasets where computational efficiency is important.
# 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
# Observations:

# t-SNE is a non-linear method that focuses on preserving local relationships and small clusters within the data.
# In the Original High-Dimensional Neighbors plot, the local structures are tightly clustered.
# In the Reduced Low-Dimensional Neighbors plot, t-SNE successfully captures the local clusters but distorts the global relationships, spreading the clusters far apart.
# While t-SNE excels at visualizing small, well-separated clusters, it struggles with global separation.
# Strengths:

# Captures local structure: t-SNE is excellent at preserving neighboring relationships, making it ideal for visualizing clusters.
# Reveals hidden patterns: t-SNE uncovers detailed structures that other methods may miss, especially for non-linear datasets.
# Weaknesses:

# Global structure distortion: t-SNE often fails to preserve the larger structure, leading to distorted relationships between clusters.
# Computationally expensive: t-SNE is slow to compute, especially on larger datasets, which limits its scalability.
# Best Used For:

# When you need to visualize local clusters in high-dimensional data, particularly when the data has non-linear relationships.
# Suitable for small datasets or when the goal is to uncover fine-grained local structures.
# 3. UMAP (Uniform Manifold Approximation and Projection)
# Observations:

# UMAP is a non-linear dimensionality reduction method that aims to balance between preserving both local and global structures.
# In the Original High-Dimensional Neighbors plot, UMAP clearly shows connected local neighborhoods.
# In the Reduced Low-Dimensional Neighbors plot, UMAP retains both local clusters and global separation better than t-SNE.
# UMAP provides a clean separation between clusters without excessive global distortion, making it a good balance between t-SNE and PCA.
# Strengths:

# Balances local and global structure: UMAP preserves both local clusters and some global structure, providing a better balance than t-SNE.
# Scalable: UMAP is faster and more scalable than t-SNE, making it suitable for large datasets.
# Good cluster separation: UMAP excels in separating clusters while maintaining local neighborhoods.
# Weaknesses:

# Global structure may still be slightly distorted, especially with complex datasets.
# Parameter tuning: UMAP's performance can be sensitive to the choice of hyperparameters (e.g., the number of neighbors).
# Best Used For:

# When you need a balance between capturing local structure and preserving global relationships, especially for larger datasets.
# Suitable for non-linear datasets where both local and global features are important.
# 4. SVD (Singular Value Decomposition)
# Observations:

# SVD is a linear method like PCA, used to decompose data into its principal components.
# In the Original High-Dimensional Neighbors plot, SVD captures the global structure and maintains connectivity between data points.
# In the Reduced Low-Dimensional Neighbors plot, SVD produces a linear projection similar to PCA, but local clusters may not be preserved as well.
# SVD, like PCA, is good at capturing global variance but struggles to handle complex non-linear relationships.
# Strengths:

# Global structure preservation: SVD excels in preserving large-scale relationships between features.
# Efficient and scalable: Similar to PCA, SVD is computationally efficient and suitable for large-scale data processing.
# Good for matrix factorization: SVD is often used for tasks such as text analysis (e.g., Latent Semantic Analysis) and image compression.
# Weaknesses:

# Local structure distortion: Like PCA, SVD struggles to capture non-linear relationships, leading to distorted local clusters.
# Linear method: SVD does not work well with non-linear structures.
# Best Used For:

# When working with linear datasets or applications such as text processing, image analysis, or matrix factorization.
# Best for global structure preservation and when computational efficiency is key.
# Summary of the Comparison:
# Method	Strengths	Weaknesses	Best Used For
# PCA	- Preserves global variance well.
# - Computationally efficient.
# - Easy to interpret.	- Fails to preserve local non-linear relationships.
# - Not suitable for non-linear structures.	Understanding the global structure or variance in the data and for linear relationships.
# t-SNE	- Excellent at capturing local structures.
# - Reveals hidden clusters and patterns.	- Poor at preserving global structure.
# - Computationally expensive and does not scale well to large datasets.	Best for visualizing local clusters and revealing non-linear relationships, especially for small datasets.
# UMAP	- Balances local and global structures well.
# - Faster and more scalable than t-SNE.
# - Good for cluster separation while preserving global structure.	- Some global structure might still be lost.
# - Requires parameter tuning for optimal performance.	Ideal when you need a balance between local structure preservation and global relationships, especially for large datasets.
# SVD	- Global structure preservation.
# - Efficient and scalable.
# - Useful for matrix factorization tasks.	- Poor at preserving non-linear relationships.
# - Local clusters might be distorted.	Useful for linear datasets and tasks like text analysis or image processing.
# Final Recommendation on Dimensionality Reduction Method:
# If global structure is important: PCA or SVD is the better choice, as they focus on preserving overall variance and give a clearer picture of large-scale relationships.
# If local relationships are critical: t-SNE is ideal for capturing small clusters and local structures but may distort global relationships.
# If you need a balance: UMAP is the best compromise, as it preserves both local clusters and some global structures, while being faster and more scalable than t-SNE.
# Silhouette Scores:
# PCA: 0.589
# t-SNE: 0.404
# UMAP: 0.510
# SVD: 0.589
# Based on the Silhouette Scores:

# PCA and SVD are best for capturing global structures, providing the best clustering performance.
# UMAP offers a balance between local preservation and global structure, performing better than t-SNE but slightly less effective than PCA and SVD.
# t-SNE is better suited for visualizing fine-grained local clusters, but its distortion of global relationships results in a lower Silhouette Score


##################################################################################################