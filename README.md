# 🏠 Ames Housing — Price Prediction ML Pipeline

> An end-to-end Machine Learning project that predicts house sale prices using the Ames Housing dataset.  
> Built with Python, Scikit-learn, XGBoost, and Streamlit.

---

## 📌 Project Overview

This project builds a complete ML pipeline to predict residential property prices in Ames, Iowa.  
It covers everything from raw data cleaning to model comparison and an interactive web app for real-time predictions.

---

## 🚀 Live Demo

> Run the app locally with:
```bash
streamlit run app.py
```

---

## 🧠 Models Used

| Model | Description |
|---|---|
| Linear Regression | Baseline model |
| Ridge Regression | Regularized linear model |
| Decision Tree | Non-linear tree-based model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Best performing boosting model |

> 🏆 **Best Model: Gradient Boosting — R² ≈ 0.91**

---

## 📊 Pipeline Steps

1. **Load Data** — Ames Housing CSV dataset
2. **Drop Irrelevant Columns** — Remove IDs and leakage columns
3. **Remove Duplicates**
4. **Handle Missing Values** — Median imputation for numeric, mode for categorical
5. **Encode Categorical Variables** — Ordinal + One-Hot Encoding
6. **Train/Test Split** — 80% train / 20% test
7. **Feature Scaling** — StandardScaler
8. **Train 5 Models**
9. **Evaluate & Compare** — R², MAE, RMSE
10. **Save Best Model** — via Joblib

---

## 📁 Project Structure

```
ames-housing-ml/
│
├── app.py                    # Streamlit web app
├── housing_ml_pipeline.py    # Full ML pipeline
├── housing_sample.csv        # Sample dataset
│
├── notebooks/
│   ├── Dataset_overview.ipynb
│   └── EDA.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/Toni-11/my-projects.git
cd my-projects

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📦 Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- joblib
- matplotlib
- seaborn

---

## 📈 Dataset

- **Source:** [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- **Size:** 2,930 rows × 82 columns
- **Target:** `SalePrice` (house sale price in USD)

---

## 👤 Author

**Antton Mikhael**  
GitHub: [@Toni-11](https://github.com/Toni-11)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
