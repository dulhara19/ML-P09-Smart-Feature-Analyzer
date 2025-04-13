# app.py

import streamlit as st
import pandas as pd

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
