# app.py

import streamlit as st
import pandas as pd
from preprocessing.correlation import find_high_correlations
from preprocessing.pca import apply_pca


st.set_page_config(page_title="Smart Feature Analyzer", layout="wide")

st.title("🧠 Smart Feature Analyzer")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### 🔍 Preview of Dataset:")
    st.dataframe(df.head())
else:
    st.info("👈 Upload a dataset to get started.")


#-------------------------

if uploaded_file is not None:
        st.write("### 📊 High Correlation Detection")

        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        threshold = st.slider("Set correlation threshold:", 0.0, 1.0, 0.9)

        if numeric_df.shape[1] >= 2:
            result = find_high_correlations(numeric_df, threshold=threshold)

            if result.empty:
                st.success("✅ No feature pairs found above the threshold.")
            else:
                st.warning("⚠️ Highly correlated features detected:")
                st.dataframe(result)
        else:
            st.info("Need at least two numeric features to calculate correlation.")

#-------------------------

st.write("### 🌈 Principal Component Analysis (PCA)")

max_components = min(numeric_df.shape[1], numeric_df.shape[0])
num_components = st.slider("Select number of principal components", 1, max_components, 2)

pca_df, explained_variance, pc_columns = apply_pca(df, n_components=num_components)

if pca_df is not None:
        st.write("#### 🔍 PCA Components:")
        st.dataframe(pca_df)

        # Variance chart
        st.write("#### 📈 Explained Variance Ratio")
        exp_var_df = pd.DataFrame({
            'Principal Component': pc_columns,
            'Explained Variance': explained_variance
        })
        st.bar_chart(exp_var_df.set_index('Principal Component'))
else:
        st.warning("⚠️ PCA requires at least some numeric features.")
