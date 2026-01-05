import pandas as pd
import numpy as np

INPUT_CSV = "data.csv"

# Configuration Constants
FEATURE_MIN_HOUR = 0
HOUR_TOL = 24            # Tolerance for finding the target row
USE_ALL_SNAPSHOTS = False # True: keep all snapshots in window; False: keep only latest per video

def generate_dataset(source_df, max_feature_hour, target_hour, target_col_name, output_filename):
    """
    Generates a dataset based on specific feature windows and target hours.
    """
    print(f"--- Processing: Features <= {max_feature_hour}h -> Target @ {target_hour}h ---")
    
    # 1) Filter Features (0 to max_feature_hour)
    feat = source_df.loc[
        (source_df["hours_since_upload"] >= FEATURE_MIN_HOUR)
        & (source_df["hours_since_upload"] <= max_feature_hour)
    ].copy()

    # Optional: If you only want the latest single snapshot per video within window
    if not USE_ALL_SNAPSHOTS:
        feat = feat.sort_values(["video_id", "hours_since_upload"], ascending=[True, False]) \
                   .drop_duplicates("video_id", keep="first")

    # 2) Filter Label (Target at target_hour)
    tgt = source_df.loc[
        (source_df["hours_since_upload"] >= target_hour)
        & (source_df["hours_since_upload"] <= target_hour + HOUR_TOL)
    ].copy()

    # Find the row closest to the specific target hour
    tgt["tgt_dist"] = (tgt["hours_since_upload"] - target_hour).abs()
    tgt = tgt.sort_values(["video_id", "tgt_dist"]).drop_duplicates("video_id")
    tgt = tgt.drop(columns=["tgt_dist"])

    # Rename target column (e.g., view_count -> views_30d)
    tgt = tgt[["video_id", "view_count"]].rename(columns={"view_count": target_col_name})

    # 3) Join
    dataset = feat.merge(tgt, on="video_id", how="inner")
    
    print(f"Shape: {dataset.shape}")
    dataset.to_csv(output_filename, index=False)
    print(f"Saved: {output_filename}\n")

# --- Main Execution ---

print("Loading raw snapshots...")
df = pd.read_csv(INPUT_CSV)

# 1. ORIGINAL: Under Day 3 (72h) features -> Day 7 (168h) view
generate_dataset(
    source_df=df,
    max_feature_hour=72,
    target_hour=168,
    target_col_name="views_7d",
    output_filename="day3_to_day7_data_ver2.csv"
)
'''
# 2. NEW: Under Day 3 (72h) features -> Day 30 (720h) view
generate_dataset(
    source_df=df,
    max_feature_hour=72,
    target_hour=720,
    target_col_name="views_30d",
    output_filename="day3_to_day30_data.csv"
)

# 3. NEW: Under Day 7 (168h) features -> Day 30 (720h) view
generate_dataset(
    source_df=df,
    max_feature_hour=168,
    target_hour=720,
    target_col_name="views_30d",
    output_filename="day7_to_day30_data.csv"
)

# 3. NEW: Under Day 1~7 features -> Day 30 (720h) view
for day in range(1, 8):
    feature_hour = day * 24  # convert days to hours (1d=24h)
    generate_dataset(
        source_df=df,
        max_feature_hour=feature_hour,
        target_hour=720,
        target_col_name="views_30d",
        output_filename=f"day{day}_to_day30_data.csv"
    )
'''