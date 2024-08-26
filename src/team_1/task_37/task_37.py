import os

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class DimensionalityReducer:
    def __init__(self, df: pd.DataFrame, target_variable: str = "is_rehospitalization"):
        """
        Initialize with the DataFrame and standardize the features.
        """
        self.df = df
        self.X = df.drop(
            columns=["is_rehospitalization"]
        )  # Assuming 'is_rehospitalization' is the target variable
        self.y = df["is_rehospitalization"]

        # Standardize the features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def apply_pca(self, n_components=2):
        """
        Apply Principal Component Analysis (PCA).
        """
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(self.X_scaled)
        return X_pca

    def apply_tsne(self, n_components=2, perplexity=30, random_state=42):
        """
        Apply t-Distributed Stochastic Neighbor Embedding (t-SNE).
        """
        tsne = TSNE(
            n_components=n_components, perplexity=perplexity, random_state=random_state
        )
        X_tsne = tsne.fit_transform(self.X_scaled)
        return X_tsne

    def apply_lda(self, n_components=2):
        """
        Apply Linear Discriminant Analysis (LDA).
        """
        lda = LDA(n_components=n_components)
        X_lda = lda.fit_transform(self.X_scaled, self.y)
        return X_lda

    def apply_feature_selection(self, k=2):
        """
        Apply Feature Selection using SelectKBest.
        """
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(self.X_scaled, self.y)
        return X_selected


if __name__ == "__main__":
    # Modify as you wish this is an example how to use the reducer
    data_dir = os.path.join(os.getcwd(), "..", "..", "data")
    filepath = os.path.join(data_dir, "rehospitalization.xlsx")
    df_rehospitalization = pd.read_excel(filepath, sheet_name="hospitalization1")
    reducer = DimensionalityReducer(df_rehospitalization)
    X_pca = reducer.apply_pca(n_components=2)
    X_tsne = reducer.apply_tsne(n_components=2, perplexity=30)
    X_lda = reducer.apply_lda(n_components=2)
    X_selected = reducer.apply_feature_selection(k=2)
