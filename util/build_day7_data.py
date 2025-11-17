import pandas as pd
import numpy as np

INPUT_CSV = "data.csv"       # your raw snapshot model
OUTPUT_CSV = "day7_data.csv"   # will be used by xgboosttrain.py

#FEATURE_HOUR = 24      # when you want to take the features (e.g., 1 day after upload)
FEATURE_MIN_HOUR = 0           # include from upload
FEATURE_MAX_HOUR = 72          # <= 3 days
USE_ALL_SNAPSHOTS = True       # True: keep all snapshots <=72h; False: keep only the latest snapshot per video <=72h

TARGET_HOUR = 168      # 7 days
HOUR_TOL = 24           # +/- tolerance in hours when matching

print("Loading snapshots...")
df = pd.read_csv(INPUT_CSV)

# ---- 1) Features at FEATURE_HOUR ----
feat = df.loc[
    (df["hours_since_upload"] >= FEATURE_MIN_HOUR)
    & (df["hours_since_upload"] <= FEATURE_MAX_HOUR)
].copy()

# if multiple rows in that window, pick the one closest to FEATURE_HOUR
# feat["feat_dist"] = (feat["hours_since_upload"] - FEATURE_HOUR).abs()
# feat = feat.sort_values(["video_id", "feat_dist"]).drop_duplicates("video_id")
# feat = feat.drop(columns=["feat_dist"])

if not USE_ALL_SNAPSHOTS:
    # keep exactly ONE snapshot per video: the latest within the 0~72h window
    feat = feat.sort_values(["video_id", "hours_since_upload"], ascending=[True, False]) \
               .drop_duplicates("video_id", keep="first")

# ---- 2) Label at TARGET_HOUR (views_7d) ----
tgt = df.loc[
    (df["hours_since_upload"] >= TARGET_HOUR)
    & (df["hours_since_upload"] <= TARGET_HOUR + HOUR_TOL)
].copy()

tgt["tgt_dist"] = (tgt["hours_since_upload"] - TARGET_HOUR).abs()
tgt = tgt.sort_values(["video_id", "tgt_dist"]).drop_duplicates("video_id")
tgt = tgt.drop(columns=["tgt_dist"])

# we only need video_id and view_count for the target
tgt = tgt[["video_id", "view_count"]].rename(columns={"view_count": "views_7d"})

# ---- 3) Join features and label ----
dataset = feat.merge(tgt, on="video_id", how="inner")

print("Final dataset shape:", dataset.shape)
print(dataset.head())

dataset.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)