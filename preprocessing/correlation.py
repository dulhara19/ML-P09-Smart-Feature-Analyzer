# preprocessing/correlation.py

import pandas as pd

def find_high_correlations(df, threshold=0.9):
    corr_matrix = df.corr()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                high_corr_pairs.append({
                    "Feature 1": corr_matrix.columns[i],
                    "Feature 2": corr_matrix.columns[j],
                    "Correlation": round(corr_value, 3)
                })

    return pd.DataFrame(high_corr_pairs)
