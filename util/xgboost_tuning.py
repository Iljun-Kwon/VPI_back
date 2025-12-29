import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

def run_hyperparameter_search(input_csv, target_col):
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING: {input_csv} -> Target: {target_col}")
    print(f"{'='*80}")

    # 1) Load Data
    try:
        df = pd.read_csv(input_csv)
        print(f"Data loaded: {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: {input_csv} not found.")
        return

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

    # Log-transform target
    df["log_target"] = np.log1p(df[target_col])
    y = df["log_target"]
    X = df[NUM_COLS + CAT_COLS]

    # 3) Train/valid split (Grouped by video_id)
    groups = df["video_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    try:
        train_idx, valid_idx = next(gss.split(X, y, groups))
    except ValueError:
        print("Not enough data to split. Exiting.")
        return

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # Precompute y_valid_real for metrics to save time in loop
    y_valid_real = np.expm1(y_valid)

    # 4) Define Hyperparameter Grid
    # Feel free to adjust these lists to test different ranges
    param_grid = {
        'max_depth': [2, 4, 6, 8, 10],
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'n_estimators': [200, 500, 1000],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_alpha': [0.0, 0.1],
        # Fixed parameters
        'objective': ['reg:squarederror'],
        'tree_method': ['hist']
    }

    grid = list(ParameterGrid(param_grid))
    print(f"Testing {len(grid)} combinations...\n")

    results = []

    # 5) Preprocessor (defined once)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    # 6) Loop through parameters
    for i, params in enumerate(grid):
        print(f"Training combo {i+1}/{len(grid)}: {params}")

        # Initialize model with current params
        model = xgb.XGBRegressor(**params)

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred_log = pipeline.predict(X_valid)
        y_pred = np.expm1(y_pred_log)

        # Calculate Metrics
        rmse = np.sqrt(mean_squared_error(y_valid_real, y_pred))
        
        # Avoid division by zero in MAPE/wMAPE
        mape = 100 * np.mean(np.abs((y_pred - y_valid_real) / (y_valid_real + 1e-9)))
        smape = 100 * np.mean(np.abs(y_pred - y_valid_real) / (np.abs(y_pred) + np.abs(y_valid_real) / 2 + 1e-9))
        wmape = 100 * np.sum(np.abs(y_valid_real - y_pred)) / (np.sum(np.abs(y_valid_real)) + 1e-9)

        # Store result
        result_row = params.copy()
        result_row['RMSE'] = rmse
        result_row['MAPE'] = mape
        result_row['SMAPE'] = smape
        result_row['wMAPE'] = wmape
        results.append(result_row)

    # 7) Output Results Table
    results_df = pd.DataFrame(results)
    
    # Sort by wMAPE (or RMSE) to see best models at the top
    results_df = results_df.sort_values(by="MAPE", ascending=True)

    print(f"\n{'='*80}")
    print("TOP 10 HYPERPARAMETER COMBINATIONS (Sorted by MAPE)")
    print(f"{'='*80}")
    
    # Select columns to display
    display_cols = ['max_depth', 'learning_rate', 'n_estimators', 'reg_alpha', 'RMSE', 'MAPE', 'SMAPE', 'wMAPE']
    print(results_df[display_cols].head(10).to_string(index=False))

    # Optional: Save full results to CSV
    results_df.to_csv("hyperparameter_tuning_results1.csv", index=False)
    print(f"\nFull results contains {len(results_df)} rows.")

if __name__ == "__main__":
    # ONLY checking day 7 -> day 30 as requested
    train_file = "day3_to_day30_data.csv"
    target = "views_30d"
    
    run_hyperparameter_search(train_file, target)