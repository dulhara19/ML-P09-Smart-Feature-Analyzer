# Smart Feature Analyzer

Smart Feature Analyzer is a Streamlit-based tool that allows you to explore, analyze, and build quick machine learning models from your datasets. It helps identify highly correlated features, reduces dimensionality using PCA (Principal Component Analysis), and provides interactive visualizations and model insights.

---

## 📌Features

- ✅ Upload `.csv` datasets
- ✅ View basic data stats and column summaries
- ✅ Automatically detect numeric features
- ✅ Calculate rank of the correlation matrix
- ✅ Detect highly correlated feature pairs
- ✅ Perform PCA to reduce dimensionality
- ✅ Visualize explained variance by PCA components
- ✅ Build a machine learning model (Random Forest Classifier)
- ✅ Evaluate accuracy of the model
- ✅ Visualize feature importances (original or PCA-based)

---

## Example Use Case

You can use the **Iris dataset** for testing, which includes:
- Features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- Target: `species`

This dataset works great to demonstrate feature correlation, PCA dimensionality reduction, and quick classification.

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/dulhara19/smart-feature-analyzer.git
cd smart-feature-analyzer
pip install -r requirements.txt
```

