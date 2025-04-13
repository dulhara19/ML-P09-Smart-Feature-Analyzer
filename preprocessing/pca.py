# preprocessing/pca.py

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(df, n_components=None):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if numeric_df.empty:
        return None, None, None

    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)

    # Build PCA result DataFrame
    pc_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    pca_df = pd.DataFrame(principal_components, columns=pc_columns)

    return pca_df, pca.explained_variance_ratio_, pc_columns
