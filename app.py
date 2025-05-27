# app.py

import streamlit as st
import pandas as pd
from preprocessing.correlation import find_high_correlations
from preprocessing.pca import apply_pca 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import io

st.set_page_config(page_title="Smart Feature Analyzer", layout="wide")

st.title("🌀 Smart Feature Analyzer")

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

st.write("### 🌀 Principal Component Analysis (PCA)")

max_components = min(numeric_df.shape[1], numeric_df.shape[0]) 
num_components = st.slider("Select number of principal components", 1, max_components, 2)

pca_df, explained_variance, pc_columns = apply_pca(df, n_components=num_components)

if pca_df is not None:
    st.write("#### 🔍 PCA Components:")
    st.dataframe(pca_df)

    st.write("#### 📈 Explained Variance Ratio")
    exp_var_df = pd.DataFrame({
        'Principal Component': pc_columns,
        'Explained Variance': explained_variance
    })
    st.bar_chart(exp_var_df.set_index('Principal Component'))

    st.download_button(
        label="📅 Download PCA Result",
        data=pca_df.to_csv(index=False).encode('utf-8'),
        file_name='pca_result.csv',
        mime='text/csv'
    )
else:
    st.warning("⚠️ PCA requires at least some numeric features.")

#-------------------------

st.write("### 🌀 Build a Quick Machine Learning Model")

# Target selection
target_col = st.selectbox("🎯 Select target column", df.columns)

# Drop non-numeric features (except target)
numeric_df = df.select_dtypes(include=['float64', 'int64', 'int32'])

# Feature selection
selected_features = st.multiselect("🧬 Select features for modeling", list(numeric_df.columns), default=list(numeric_df.columns))

# Drop target temporarily from features
if target_col in selected_features:
    selected_features.remove(target_col)

X = numeric_df[selected_features]

# PCA on selected features
num_components = min(X.shape[1], 3)
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X)

# Encode target if needed
y = df[target_col]
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Model selection
model_type = st.selectbox("🧠 Choose your model", ["Logistic Regression", "Random Forest"])

if st.button("Train Model"):
    if model_type == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Accuracy: {acc * 100:.2f}%")

    # Confusion matrix
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("### 📋 Classification Report")
    st.dataframe(report_df)

    # Feature importances (if Random Forest)
    if model_type == "Random Forest":
        importances = model.feature_importances_
        feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        st.write("### 🔍 Feature Importances")
        st.bar_chart(importance_df.set_index('Feature'))

    # PCA Scatter Plot
    if X_pca.shape[1] >= 2:
        pca_plot_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
        pca_plot_df['Target'] = y
        fig = px.scatter(pca_plot_df, x='PC1', y='PC2', color=pca_plot_df['Target'].astype(str),
                         title="PCA Scatter Plot")
        st.plotly_chart(fig)

    # Model download
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    st.download_button("📦 Download Trained Model", buffer.getvalue(), file_name="trained_model.pkl")
