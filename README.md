# Smart Feature Analyzer

Smart Feature Analyzer is a Streamlit-based tool that allows you to explore, analyze, and build quick machine learning models from your datasets. It helps identify highly correlated features, reduces dimensionality using PCA (Principal Component Analysis), and provides interactive visualizations and model insights.

<img src="docs/screenshots/Screenshot 2025-05-28 011753.png" width="100%" alt="user can add csv files"> users can add csv files, it will show the first few rows
<img src="docs/screenshots/Screenshot 2025-05-28 012135.png" alt="Mobile demo">
<img src="docs/screenshots/Screenshot 2025-05-28 012152.png" alt="Mobile demo">
<img src="docs/screenshots/Screenshot 2025-05-28 012315.png" alt="Mobile demo">
<img src="docs/screenshots/Screenshot 2025-05-28 013030.png" alt="Mobile demo">
<img src="docs/screenshots/Screenshot 2025-05-28 013138.png" alt="Mobile demo">

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
Then run the app:
```bash
streamlit run app.py
```
## Built With

This project is built using Python 3.x, with the help of powerful libraries such as Pandas and NumPy for data manipulation, Scikit-learn for machine learning tasks including PCA and model building, and Streamlit for creating an interactive web application interface. For visualizations, it uses Seaborn and Matplotlib to provide clear insights into correlations, PCA results, and feature importances.

## How It Works

   - Upload your dataset (.csv)
   - Select the target variable
   - Check for feature correlations
   - Perform PCA to reduce dimensions
   - Select number of components to keep
   - Train a quick ML model and check accuracy
   - Visualize feature importances

## Future Improvements

In future versions of this project, we plan to extend its capabilities by adding support for regression models, allowing users to work with both classification and regression problems. There will also be an option to save or export trained models for future use. To enhance model explainability, we aim to integrate SHAP visualizations, helping users understand the impact of each feature more intuitively. Additionally, we plan to include a wider range of machine learning algorithms such as Support Vector Machines (SVM), Logistic Regression, and others to offer greater flexibility and experimentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Let me know if you want help for deploying this project publicly!

Created with ❤️ by Dulhara :) feel free to pull request
For learning, exploration, and rapid prototyping of feature engineering workflows.

