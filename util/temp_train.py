# for checking subscriber diff
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import os

def train_main_model(input_csv, target_col, output_model_name):
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL: {input_csv} -> {output_model_name}")
    print(f"{'='*60}")

    # 1) Load Data
    if not os.path.exists(input_csv):
        print(f"Skipping {input_csv} (File not found)")
        return

    df = pd.read_csv(input_csv)
    
    if df.empty:
        print(f"Skipping {input_csv} (DataFrame is empty)")
        return

    print(f"Target '{target_col}' stats:")
    print(df[target_col].describe(percentiles=[0.5, 0.9, 0.99]))

    # 2) Define features and target
    NUM_COLS = [
        "subscriber_count",
        "hours_since_upload",
        "video_length",
        "view_count",
        "like_count",
        "comment_count",
        "hour_sin",
        "hour_cos",
        "likes_per_subscriber",
        "likes_per_time",
    ]
    # Ensure these columns actually exist in the dataframe before selecting them
    # (Just in case some preprocessing dropped them, though unlikely)
    available_num_cols = [c for c in NUM_COLS if c in df.columns]
    
    CAT_COLS = ["category", "day_of_week", "is_short"]
    available_cat_cols = [c for c in CAT_COLS if c in df.columns]

    # Target: log1p
    df["log_target"] = np.log1p(df[target_col])
    y = df["log_target"]
    X = df[available_num_cols + available_cat_cols]

    # 3) Train/valid split (Grouped by video_id)
    groups = df["video_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    try:
        train_idx, valid_idx = next(gss.split(X, y, groups))
    except ValueError:
        print("Not enough data to split. Skipping.")
        return

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # 4) Preprocessing and model
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", available_num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), available_cat_cols),
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
    print(f"Saved model to: {output_model_name}")

# --- Execution Loop ---
if __name__ == "__main__":
    # We now look for the LOW and HIGH variants of each file
    tasks = [
        # --- LOW SUBSCRIBER MODELS ---
        ("day3_to_day7_LOW.csv",   "views_7d",  "xgb_day3_to_day7_LOW.pkl"),
        ("day3_to_day30_LOW.csv",  "views_30d", "xgb_day3_to_day30_LOW.pkl"),
        ("day7_to_day30_LOW.csv",  "views_30d", "xgb_day7_to_day30_LOW.pkl"),

        # --- HIGH SUBSCRIBER MODELS ---
        ("day3_to_day7_HIGH.csv",   "views_7d",  "xgb_day3_to_day7_HIGH.pkl"),
        ("day3_to_day30_HIGH.csv",  "views_30d", "xgb_day3_to_day30_HIGH.pkl"),
        ("day7_to_day30_HIGH.csv",  "views_30d", "xgb_day7_to_day30_HIGH.pkl"),
    ]

    for csv_file, tgt_col, model_file in tasks:
        train_main_model(csv_file, tgt_col, model_file)