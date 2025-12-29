# for checking subscriber diff
import pandas as pd
import numpy as np

INPUT_CSV = "data.csv"

# --- Configuration Constants ---
FEATURE_MIN_HOUR = 0
HOUR_TOL = 24             # Tolerance for finding the target row
USE_ALL_SNAPSHOTS = False # True: keep all snapshots; False: keep only latest per video

# --- Subscriber Split Configuration ---
# IMPORTANT: Change 'subscriber_count' to the exact column name in your CSV
SUB_COL_NAME = "subscriber_count" 

# Set to True to split by the median (50% high / 50% low).
# Set to False to use the MANUAL_THRESHOLD value.
USE_MEDIAN_SPLIT = False
MANUAL_THRESHOLD = 200000 

def generate_dataset(source_df, max_feature_hour, target_hour, target_col_name, output_filename):
    """
    Generates a dataset based on specific feature windows and target hours.
    """
    if source_df.empty:
        print(f"Skipping {output_filename}: Source DataFrame is empty.")
        return

    print(f"--- Processing: Features <= {max_feature_hour}h -> Target @ {target_hour}h ---")
    
    # 1) Filter Features (0 to max_feature_hour)
    feat = source_df.loc[
        (source_df["hours_since_upload"] >= FEATURE_MIN_HOUR)
        & (source_df["hours_since_upload"] <= max_feature_hour)
    ].copy()

    if not USE_ALL_SNAPSHOTS:
        feat = feat.sort_values(["video_id", "hours_since_upload"], ascending=[True, False]) \
                   .drop_duplicates("video_id", keep="first")

    # 2) Filter Label (Target at target_hour)
    tgt = source_df.loc[
        (source_df["hours_since_upload"] >= target_hour)
        & (source_df["hours_since_upload"] <= target_hour + HOUR_TOL)
    ].copy()

    tgt["tgt_dist"] = (tgt["hours_since_upload"] - target_hour).abs()
    tgt = tgt.sort_values(["video_id", "tgt_dist"]).drop_duplicates("video_id")
    tgt = tgt.drop(columns=["tgt_dist"])

    # Rename target column
    tgt = tgt[["video_id", "view_count"]].rename(columns={"view_count": target_col_name})

    # 3) Join
    dataset = feat.merge(tgt, on="video_id", how="inner")
    
    print(f"Shape: {dataset.shape}")
    dataset.to_csv(output_filename, index=False)
    print(f"Saved: {output_filename}\n")

# --- Main Execution ---

print("Loading raw snapshots...")
df = pd.read_csv(INPUT_CSV)

# --- Check Subscriber Column ---
if SUB_COL_NAME not in df.columns:
    raise ValueError(f"Column '{SUB_COL_NAME}' not found in CSV. Available columns: {list(df.columns)}")

# --- Perform Split ---
# We take the max subscriber count per video to determine that video's 'category'
# This prevents a video from floating between High/Low if their count changes slightly over time.
video_subs = df.groupby("video_id")[SUB_COL_NAME].max()

if USE_MEDIAN_SPLIT:
    threshold = video_subs.median()
    print(f"Splitting data by Median Subscriber Count: {int(threshold)}")
else:
    threshold = MANUAL_THRESHOLD
    print(f"Splitting data by Manual Threshold: {threshold}")

# Identify High vs Low Video IDs
low_ids = video_subs[video_subs < threshold].index
high_ids = video_subs[video_subs >= threshold].index

df_low = df[df["video_id"].isin(low_ids)].copy()
df_high = df[df["video_id"].isin(high_ids)].copy()

print(f"Low Sub Data Rows: {len(df_low)} | High Sub Data Rows: {len(df_high)}")
print("-" * 40)

# --- Processing Loop ---

# We define the 3 scenarios as a list of dictionaries to avoid code repetition
scenarios = [
    {"max_h": 72,  "tgt_h": 168, "col": "views_7d",  "file_prefix": "day3_to_day7"},
    {"max_h": 72,  "tgt_h": 720, "col": "views_30d", "file_prefix": "day3_to_day30"},
    {"max_h": 168, "tgt_h": 720, "col": "views_30d", "file_prefix": "day7_to_day30"},
]

# Run generation for LOW subscribers
print("\n=== GENERATING LOW SUBSCRIBER DATASETS ===")
for sc in scenarios:
    generate_dataset(
        source_df=df_low,
        max_feature_hour=sc["max_h"],
        target_hour=sc["tgt_h"],
        target_col_name=sc["col"],
        output_filename=f"{sc['file_prefix']}_LOW.csv"
    )

# Run generation for HIGH subscribers
print("\n=== GENERATING HIGH SUBSCRIBER DATASETS ===")
for sc in scenarios:
    generate_dataset(
        source_df=df_high,
        max_feature_hour=sc["max_h"],
        target_hour=sc["tgt_h"],
        target_col_name=sc["col"],
        output_filename=f"{sc['file_prefix']}_HIGH.csv"
    )