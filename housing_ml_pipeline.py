# =============================================================================
# End-to-End Machine Learning Pipeline — Ames Housing Dataset (Regression)
# Target: SalePrice
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================================================================
# CONFIGURATION — tweak here without touching the rest of the code
# =============================================================================

TARGET_COLUMN   = "SalePrice"
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
NAN_DROP_THRESH = 0.02   # Drop rows when column missingness is below this %
MODEL_SAVE_PATH = "best_model.joblib"

# Ordinal columns: define the natural rank order for each.
# Values not in the list are treated as NaN and will be imputed.
ORDINAL_COLUMNS = {
    "Exter Qual":    ["Po", "Fa", "TA", "Gd", "Ex"],
    "Exter Cond":    ["Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Qual":     ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Cond":     ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Exposure": ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFin Type 1":["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFin Type 2":["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Heating QC":    ["Po", "Fa", "TA", "Gd", "Ex"],
    "Kitchen Qual":  ["Po", "Fa", "TA", "Gd", "Ex"],
    "Fireplace Qu":  ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "Garage Finish": ["NA", "Unf", "RFn", "Fin"],
    "Garage Qual":   ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "Garage Cond":   ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "Paved Drive":   ["N", "P", "Y"],
    "Lot Shape":     ["IR3", "IR2", "IR1", "Reg"],
    "Land Slope":    ["Sev", "Mod", "Gtl"],
    "Functional":    ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "Fence":         ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "Pool QC":       ["NA", "Fa", "TA", "Gd", "Ex"],
    "Utilities":     ["ELO", "NoSeWa", "NoSewr", "AllPub"],
}

# Columns to drop entirely (IDs / leakage columns — not predictive features)
DROP_COLUMNS = ["Order", "PID"]

# =============================================================================
# STEP 0 — Load Data
# =============================================================================
# Replace the line below with your actual data source, e.g.:
#   df = pd.read_csv("ames_housing.csv")
# For demonstration, we generate a tiny synthetic placeholder so the script
# is runnable as-is; swap in your real DataFrame before use.

print("=" * 65)
print("  Ames Housing — ML Pipeline")
print("=" * 65)

try:
    df = pd.read_csv("ames_housing.csv")
    print(f"✓ Data loaded — {df.shape[0]:,} rows × {df.shape[1]} columns\n")
except FileNotFoundError:
    print("⚠  'ames_housing.csv' not found.")
    print("   Place your CSV next to this script and re-run.\n")
    raise SystemExit(1)

# =============================================================================
# STEP 1 — Drop Irrelevant Columns
# =============================================================================
print("── Step 1 · Drop irrelevant columns ─────────────────────────")
cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"   Dropped: {cols_to_drop}\n")

# =============================================================================
# STEP 2 — Remove Duplicate Rows
# =============================================================================
print("── Step 2 · Remove duplicates ────────────────────────────────")
n_before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"   Removed {n_before - len(df)} duplicate row(s). Rows remaining: {len(df):,}\n")

# =============================================================================
# STEP 3 — Separate Target Variable
# =============================================================================
print("── Step 3 · Separate target ──────────────────────────────────")
y = df[TARGET_COLUMN].copy()
X = df.drop(columns=[TARGET_COLUMN])
print(f"   Target '{TARGET_COLUMN}' isolated. Feature matrix: {X.shape}\n")

# =============================================================================
# STEP 4 — Handle Missing Values
# =============================================================================
print("── Step 4 · Handle missing values ────────────────────────────")

# --- 4a. Drop rows for columns with very few NaNs (< NAN_DROP_THRESH) -------
n_rows = len(X)
low_nan_cols = [
    col for col in X.columns
    if 0 < X[col].isna().sum() / n_rows < NAN_DROP_THRESH
]
print(f"   Columns with <{NAN_DROP_THRESH*100:.0f}% NaNs (drop affected rows): {low_nan_cols}")

if low_nan_cols:
    mask_keep = X[low_nan_cols].notna().all(axis=1)
    X = X[mask_keep].reset_index(drop=True)
    y = y[mask_keep].reset_index(drop=True)
    print(f"   Rows after dropping: {len(X):,}")

# --- 4b. Identify remaining columns with NaNs --------------------------------
# Separate numeric vs categorical
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Impute numerics with median
num_nan_cols = [c for c in num_cols if X[c].isna().any()]
if num_nan_cols:
    num_imputer = SimpleImputer(strategy="median")
    X[num_nan_cols] = num_imputer.fit_transform(X[num_nan_cols])
    print(f"   Median-imputed numeric cols: {num_nan_cols}")

# Impute categoricals with mode
cat_nan_cols = [c for c in cat_cols if X[c].isna().any()]
if cat_nan_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[cat_nan_cols] = cat_imputer.fit_transform(X[cat_nan_cols])
    print(f"   Mode-imputed categorical cols: {cat_nan_cols}")

print(f"   Remaining NaNs: {X.isna().sum().sum()}\n")

# =============================================================================
# STEP 5 — Encode Categorical Variables
# =============================================================================
print("── Step 5 · Encode categorical variables ─────────────────────")

# --- 5a. Ordinal Encoding for columns with a natural order ------------------
ordinal_cols_present = [c for c in ORDINAL_COLUMNS if c in X.columns]
if ordinal_cols_present:
    categories = [ORDINAL_COLUMNS[c] for c in ordinal_cols_present]
    ord_encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    X[ordinal_cols_present] = ord_encoder.fit_transform(X[ordinal_cols_present])
    print(f"   Ordinal-encoded ({len(ordinal_cols_present)} cols): {ordinal_cols_present}")

# --- 5b. One-Hot Encoding for all remaining object columns ------------------
nominal_cols = [
    c for c in X.select_dtypes(include=["object"]).columns
    if c not in ordinal_cols_present
]
if nominal_cols:
    X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)
    print(f"   One-hot encoded {len(nominal_cols)} nominal column(s).")

print(f"   Final feature count after encoding: {X.shape[1]}\n")

# Ensure all remaining columns are numeric (get_dummies may create bool cols)
X = X.astype(float)

# =============================================================================
# STEP 6 — Train / Test Split
# =============================================================================
print("── Step 6 · Train/Test split (80/20) ────────────────────────")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"   Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}\n")

# =============================================================================
# STEP 7 — Feature Scaling (StandardScaler)
# =============================================================================
print("── Step 7 · Scale features (StandardScaler) ─────────────────")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled  = scaler.transform(X_test)         # apply same transform to test
print("   Scaling complete.\n")

# =============================================================================
# STEP 8 — Define & Train Models
# =============================================================================
print("── Step 8 · Train 5 regression models ───────────────────────")

models = {
    "Linear Regression":          LinearRegression(),
    "Ridge Regression":           Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Decision Tree":              DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest":              RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting":          GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE),
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print(f"   ✓ {name} trained")

print()

# =============================================================================
# STEP 9 — Evaluate All Models
# =============================================================================
print("── Step 9 · Evaluate on test set ────────────────────────────")

results = []
for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({
        "Model":  name,
        "R² Score": round(r2, 4),
        "MAE ($)":  round(mae, 2),
        "RMSE ($)": round(rmse, 2),
    })

results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False).reset_index(drop=True)
results_df.index += 1   # rank starts at 1

print("\n  ┌─ Model Performance (sorted by R² Score) ─────────────────┐")
print(results_df.to_string())
print("  └──────────────────────────────────────────────────────────┘\n")

# =============================================================================
# STEP 10 — Save the Best Model
# =============================================================================
print("── Step 10 · Save the best model ────────────────────────────")

best_model_name = results_df.iloc[0]["Model"]
best_model      = trained_models[best_model_name]
best_r2         = results_df.iloc[0]["R² Score"]

joblib.dump(best_model, MODEL_SAVE_PATH)
print(f"   🏆 Best model : {best_model_name}  (R² = {best_r2})")
print(f"   💾 Saved to   : '{MODEL_SAVE_PATH}'\n")

# =============================================================================
# QUICK USAGE GUIDE — how to load and use the saved model later
# =============================================================================
print("── How to reload and predict later ──────────────────────────")
print("""
   import joblib, numpy as np
   model = joblib.load("best_model.joblib")

   # X_new must be preprocessed (encoded + scaled) the same way as X_train
   predictions = model.predict(X_new_scaled)
   print(predictions)
""")
print("=" * 65)
print("  Pipeline complete!")
print("=" * 65)
