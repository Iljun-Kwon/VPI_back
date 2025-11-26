import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# -----------------------------
# 1) Load your model
# -----------------------------
# Assume you have a CSV with one row per training example and a column 'views_7d'
df = pd.read_csv("day7_data.csv")
print("views_7d stats:")
print(df["views_7d"].describe(percentiles=[0.5, 0.9, 0.99]))
# Drop rows with missing target
#df = df.dropna(subset=["views_7d"])

# -----------------------------
# 2) Define features and target
# -----------------------------
NUM_COLS = [
    "subscriber_count",
    "video_length",
    "hour_sin",
    "hour_cos",
]

CAT_COLS = ["category", "day_of_week"]

# Target: log1p of 7-day views
df["log_views_7d"] = np.log1p(df["views_7d"])
y = df["log_views_7d"]

X = df[NUM_COLS + CAT_COLS]

# -----------------------------
# 3) Train/valid split
#    (If you have upload_date, better to do time-based split)
# -----------------------------
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X, y, test_size=0.2, random_state=42
#)

groups = df["video_id"]  # same length as X/y
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(gss.split(X, y, groups))

X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

# -----------------------------
# 4) Preprocessing and model
# -----------------------------
numeric_transformer = "passthrough"

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_COLS),
        ("cat", categorical_transformer, CAT_COLS),
    ]
)

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    tree_method="hist",  # or "gpu_hist" if you want GPU and have it
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# -----------------------------
# 5) Train
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# 6) Evaluate
# -----------------------------
y_pred_log = pipeline.predict(X_valid)
y_true_log = y_valid.values

y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_true_log)
y_valid_real = np.expm1(y_valid)

rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
mae_log = mean_absolute_error(y_true_log, y_pred_log)
rmse = np.sqrt(mean_squared_error(y_valid_real, y_pred))
mae = mean_absolute_error(y_valid_real, y_pred)
mape = 100 * np.mean(np.abs((y_pred - y_true) / (y_true + 1e-9)))
smape = 100 * np.mean(
    np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-9)
)

print(f"Validation RMSE (views): {rmse:,.2f}")
print(f"Validation MAE  (views): {mae:,.2f}")
print(f"RMSE_log: {rmse_log:.4f}")
print(f"MAE_log : {mae_log:.4f}")
print(f"MAPE  (%): {mape:.2f}")
print(f"sMAPE (%): {smape:.2f}")

# Optionally, save model
import joblib
joblib.dump(pipeline, "xgb_7day_baseline_views_model.pkl")
print("Model saved to xgb_7day_baseline_views_model.pkl")