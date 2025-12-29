import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

def train_baseline_model(input_csv, target_col, output_model_name):
    print(f"\n{'='*60}")
    print(f"TRAINING BASELINE MODEL: {input_csv} -> {output_model_name}")
    print(f"{'='*60}")

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Skipping {input_csv} (File not found)")
        return

    # 2) Define features (Reduced set for baseline)
    NUM_COLS = ["subscriber_count", "video_length", "hour_sin", "hour_cos"]
    CAT_COLS = ["category", "day_of_week", "is_short"]

    # Target: log1p
    df["log_target"] = np.log1p(df[target_col])
    y = df["log_target"]
    X = df[NUM_COLS + CAT_COLS]

    # 3) Split
    groups = df["video_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    try:
        train_idx, valid_idx = next(gss.split(X, y, groups))
    except ValueError:
        print("Not enough data to split.")
        return

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # 4) Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # 5) Train
    pipeline.fit(X_train, y_train)

    # 6) Evaluate
    y_pred_log = pipeline.predict(X_valid)
    y_pred = np.expm1(y_pred_log)
    y_valid_real = np.expm1(y_valid)

    rmse = np.sqrt(mean_squared_error(y_valid_real, y_pred))
    mape = 100 * np.mean(np.abs((y_pred - y_valid_real) / (y_valid_real + 1e-9)))
    smape = 100 * np.mean(np.abs(y_pred - y_valid_real) / (np.abs(y_pred) + np.abs(y_valid_real) / 2 + 1e-9))
    wmape = 100 * np.sum(np.abs(y_valid_real - y_pred)) / (np.sum(np.abs(y_valid_real)) + 1e-9)

    print(f"Validation RMSE: {rmse:,.2f}")
    print(f"Validation MAPE: {mape:.2f}%")
    print(f"Validation SMAPE: {smape:.2f}%")
    print(f"Validation wMAPE: {wmape:.2f}%")

    # 7) Save
    joblib.dump(pipeline, output_model_name)
    print(f"Saved baseline to: {output_model_name}")

# --- Execution Loop ---
if __name__ == "__main__":
    tasks = [
        # (Input CSV, Target Column, Output Model File)
        ("day3_to_day7_data.csv",   "views_7d",  "xgb_baseline_day3_to_day7.pkl"),
        ("day3_to_day30_data.csv",  "views_30d", "xgb_baseline_day3_to_day30.pkl"),
        ("day7_to_day30_data.csv",  "views_30d", "xgb_baseline_day7_to_day30.pkl"),
    ]

    for csv_file, tgt_col, model_file in tasks:
        train_baseline_model(csv_file, tgt_col, model_file)