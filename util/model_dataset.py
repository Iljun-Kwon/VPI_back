import pandas as pd
import numpy as np
#from matplotlib.image import thumbnail
from util.metrics import parse_published_at
from pathlib import Path

def merge_datasets():
    #df_videos, df_video_snapshots, df_channels = fetch_all_data()
    BASE_DIR = Path(__file__).resolve().parent.parent
    CSV_FILES = {
        "videos": BASE_DIR / "videos.csv",
        "video_snapshots": BASE_DIR / "video_snapshots.csv",
        "channels": BASE_DIR / "channels.csv",
        "channel_snapshots": BASE_DIR / "channel_snapshots.csv"
    }

    df_videos = pd.read_csv(CSV_FILES["videos"], encoding="utf-8-sig")
    df_video_snapshots = pd.read_csv(CSV_FILES["video_snapshots"], encoding="utf-8-sig")
    df_channels = pd.read_csv(CSV_FILES["channels"], encoding="utf-8-sig")
    df_channel_snapshots = pd.read_csv(CSV_FILES["channel_snapshots"], encoding="utf-8-sig")

    df_video_snapshots = df_video_snapshots.drop(columns=['id'])
    df_video_snapshots = df_video_snapshots.rename(columns={"collected_at": "timestamp"})
    df_video_snapshots["timestamp"] = parse_published_at(df_video_snapshots["timestamp"])

    df_videos = df_videos.drop(columns=["saved_at", "thumbnail_url"])

    df = df_video_snapshots.merge(df_videos, left_on="video_id", right_on="video_id", suffixes=("", "_video"))
    df = df.drop(columns=["video_id_video"], errors="ignore")
    df["published_at"] = parse_published_at(df["published_at"])
    df["hours_since_upload"] = round((df["timestamp"] - df["published_at"]).dt.total_seconds() / 3600, 2)
    #df: video_id, timestamp, view_count, like_count, comment_count, channel_id, title, published_at, is_short, video_length, category_id

    df_channels = df_channels.drop(columns=["description", "profile_image", "banner_image", "video_count", "total_view_count", "join_date"])
    df = df.merge(df_channels, left_on="channel_id", right_on="id", suffixes=("", "_channel"))
    #df: video_id, timestamp, view_count, like_count, comment_count, channel_id, title, published_at, is_short, video_length, category_id, id, title_channel, title, handle, category,

    # df = df.merge(df_channel_snapshots, left_on="channel_id", right_on="id", suffixes=("", "_channel"))
    #df_channel_snapshots["collected_at"] = parse_published_at(df_channel_snapshots["collected_at"])
    ch = df_channel_snapshots[['channel_id', 'collected_at', 'subscriber_count']].copy()
    ch['collected_at'] = pd.to_datetime(ch['collected_at'], format='ISO8601', utc=True, errors='coerce')
    ch = ch.dropna(subset=['channel_id', 'collected_at'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True, errors='coerce')
    df_sort = df.sort_values('timestamp')
    df_channel_snapshots_sort = ch.sort_values('collected_at')
    df_merge = pd.merge_asof(df_sort, df_channel_snapshots_sort, left_on="timestamp", right_on="collected_at", by="channel_id", direction="nearest", tolerance=pd.Timedelta(hours=12))

    df_merge['day_of_week'] = df_merge['timestamp'].dt.day_name()
    df_merge['hour'] = df_merge['timestamp'].dt.hour
    df_merge['hour_sin'] = np.sin(2 * np.pi * df_merge['hour'] / 24)
    df_merge['hour_cos'] = np.cos(2 * np.pi * df_merge['hour'] / 24)
    df_merge['likes_per_subscriber'] = df_merge['like_count'] / df_merge['subscriber_count']
    df_merge['likes_per_time'] = df_merge['like_count'] / df_merge['hours_since_upload']


    print(df_merge.head())

    # category
    # df_merge = df_merge[df_merge["category"] == "Sports"]

    return df_merge

def adjust_dataset(df):
    ch_df = df

    ch_df = ch_df[ch_df["view_count"] > 0]
    ch_df = ch_df[ch_df["subscriber_count"] > 0]
    ch_df = ch_df.drop(ch_df[(ch_df["view_count"] > 10000) & (ch_df["like_count"] == 0)].index)
    #ch_df = ch_df[ch_df["like_count"] >= 0]
    ch_df = ch_df[ch_df["hours_since_upload"] > 0]
    #ch_df = ch_df[ch_df["hours_since_upload"] < 360]

    df_final = ch_df[[
        "video_id",
        "subscriber_count",
        "hours_since_upload",
        "video_length",
        "is_short",
        "view_count",
        "category",
        "like_count",
        "comment_count",
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "likes_per_subscriber",
        "likes_per_time",
    ]].dropna()

    df_final = df_final.drop_duplicates()
    df_final.to_csv("data.csv", index=False, encoding="utf-8-sig")

    #df_final.to_csv("final_dataset.csv", index=False)
    print("✅ final_dataset.csv saved with", len(df_final), "rows")
    return df_final