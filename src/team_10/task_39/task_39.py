# ====================== Dimension Reduction to erBeforeHospitalization ======================

# =========== Imports ===========
import numpy as np
import pandas as pd
import umap.umap_ as umap  # You may need to install package umap-learn for this import
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# ========== Path Config ==========
load_path = '../../team_9/assets/erBeforeHospitalization.csv'


# =========== Functions ===========


def plot_pca(pca_features_cp):
    plt.scatter(pca_features_cp[:, 0], pca_features_cp[:, 1])
    plt.title('2D Projection of Data after PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


def plot_tsne(tsne_features_cp):
    plt.scatter(tsne_features_cp[:, 0], tsne_features_cp[:, 1])
    plt.title('t-SNE Visualization')
    plt.show()


def plot_umap(umap_features_cp):
    plt.scatter(umap_features_cp[:, 0], umap_features_cp[:, 1])
    plt.title('UMAP Visualization')
    plt.show()


# =========== Dataset loading & initial fix ===========
df = pd.read_csv('erBeforeHospitalization.csv')

print(np.sum(df.isnull(), axis=0), '\n')

df_copy = df
df_copy.rename(columns={'מחלקה מאשפזת1': 'Hospitalizing_Department1'}, inplace=True)
df_copy.rename(columns={'דרך הגעה למיון': 'Arrival_To_ER'}, inplace=True)
df_copy.rename(columns={'מיון': 'ER'}, inplace=True)
df_copy.rename(columns={'אבחנות במיון': 'ER_Diagnosis'}, inplace=True)
df_copy.rename(columns={'מחלקה מאשפזת2': 'Hospitalizing_Department2'}, inplace=True)

df_copy['ev_Release_Time'] = pd.to_datetime(df_copy['ev_Release_Time']).dt.date
df_copy['ev_Admission_Date'] = pd.to_datetime(df_copy['ev_Admission_Date']).dt.date
df_copy['Release_Date'] = pd.to_datetime(df_copy['Release_Date']).dt.date
df_copy['Release_Date2'] = pd.to_datetime(df_copy['Release_Date2']).dt.date
df_copy['Admission_Entry_Date'] = pd.to_datetime(df_copy['Admission_Entry_Date'], format='%Y-%m-%d %H:%M:%S.%f',
                                                 errors='coerce')
df_copy['Admission_Entry_Date'] = pd.to_datetime(df_copy['Admission_Entry_Date']).dt.date
df_copy['Admission_Entry_Date2'] = pd.to_datetime(df_copy['Admission_Entry_Date2'], format='%Y-%m-%d %H:%M:%S.%f',
                                                  errors='coerce')
df_copy['Admission_Entry_Date2'] = pd.to_datetime(df_copy['Admission_Entry_Date2']).dt.date

df_copy.drop(columns=['Patient', 'Admission_Medical_Record', 'Medical_Record', 'Admission_Medical_Record2'],
             inplace=True)

df_copy.dropna(inplace=True)

label_encoder = LabelEncoder()
df_test = df_copy.drop(columns=['Admission_Entry_Date', 'Release_Date', 'ev_Release_Time', 'ev_Admission_Date',
                                'Admission_Entry_Date2', 'Release_Date2'])
for col in df_copy.columns:
    df_copy[col] = label_encoder.fit_transform(df_copy[col])

# ============= Dimension Reduction with PCA =============
pca = PCA(n_components=2)
pca_features = pca.fit_transform(df_copy)

plot_pca(pca_features)

pca = PCA(n_components=2)
pca_features = pca.fit_transform(df_test)

plot_pca(pca_features)

# ============= Dimension Reduction with TSNE =============
tsne = TSNE(n_components=2)
tsne_features = tsne.fit_transform(df_copy)

plot_tsne(tsne_features)

tsne = TSNE(n_components=2)
tsne_features = tsne.fit_transform(df_test)

plot_tsne(tsne_features)

# ============= Dimension Reduction with UMAP =============
reducer = umap.UMAP(n_components=2)
umap_features = reducer.fit_transform(df_copy)

plot_umap(umap_features)

reducer = umap.UMAP(n_components=2)
umap_features = reducer.fit_transform(df_test)

plot_umap(umap_features)
