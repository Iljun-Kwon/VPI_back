import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

def train_main_model(input_csv, target_col, output_model_name, output_csv_name):
    print(f"\n{'='*60}")
    print(f"TRAINING MAIN MODEL: {input_csv} -> {output_model_name}")
    print(f"{'='*60}")

    # 1) Load Data
    try:
        df_raw = pd.read_csv(input_csv)
        df = df_raw.copy()
    except FileNotFoundError:
        print(f"Skipping {input_csv} (File not found)")
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
    CAT_COLS = ["category", "day_of_week", "is_short"]

    # Target: log1p
    df["log_target"] = np.log1p(df[target_col])
    y = df["log_target"]
    X = df[NUM_COLS + CAT_COLS]

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
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=6,
        learning_rate=0.02,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        tree_method="hist", 
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # 5) Train
    pipeline.fit(X_train, y_train)

    # 6) Evaluate
    y_pred_log = pipeline.predict(X_valid)
    y_true_log = y_valid.values

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

    # 6.5) Predict on FULL dataset and save CSV
    print("Generating estimated_7d for full dataset...")
    full_pred_log = pipeline.predict(X)
    full_pred = np.expm1(full_pred_log)
    df_out = df_raw.copy()
    if "views_7d" in df_out.columns:
        df_out["estimated_7d"] = full_pred
        df_out["CPI"] = (df_out["views_7d"] / (df_out["estimated_7d"] + 1e-9)).round(3)
    elif "views_30d" in df_out.columns:
        df_out["estimated_30d"] = full_pred
        df_out["CPI"] = (df_out["views_30d"] / (df_out["estimated_30d"] + 1e-9)).round(3)
    else:
        print("⚠️ 'views_7d' not found in input CSV. Skipping ratio column.")

    df_out.to_csv(output_csv_name, index=False)
    print(f"Saved prediction CSV to: {output_csv_name}")

    # 7) Save
    joblib.dump(pipeline, output_model_name)
    print(f"Saved model to: {output_model_name}")

# --- Execution Loop ---
if __name__ == "__main__":
    tasks = [
        # (Input CSV, Target Column, Output Model File)
        ("day3_to_day7_data_ver2.csv",   "views_7d",  "xgb_day3_to_day7_model.pkl", "day3_estimated_7d_ver2.csv"),
        #("day3_to_day30_data.csv",  "views_30d", "xgb_day3_to_day30_model.pkl"),
        #("day1_to_day30_data.csv",  "views_30d", "xgb_day1_to_day30_model.pkl"),
        #("day2_to_day30_data.csv",  "views_30d", "xgb_day2_to_day30_model.pkl"),
        #("day3_to_day30_data.csv",  "views_30d", "xgb_day3_to_day30_model.pkl", "day3_estimated_30d.csv"),
        #("day4_to_day30_data.csv",  "views_30d", "xgb_day4_to_day30_model.pkl"),
        #("day5_to_day30_data.csv",  "views_30d", "xgb_day5_to_day30_model.pkl"),
        #("day6_to_day30_data.csv",  "views_30d", "xgb_day6_to_day30_model.pkl"),
        #("day7_to_day30_data.csv",  "views_30d", "xgb_day7_to_day30_model.pkl"),
    ]

    for csv_file, tgt_col, model_file, out_csv in tasks:
        train_main_model(csv_file, tgt_col, model_file, out_csv)