import joblib
import numpy as np
import pandas as pd

# Load trained pipeline
pipeline = joblib.load("xgb_day3_to_day7_model.pkl")

def predict_views_day7(
    subscriber_count,
    hours_since_upload,
    video_length,
    view_count,
    category,
    like_count,
    comment_count,
    day_of_week,
    hour_sin,
    hour_cos,
    likes_per_subscriber,
):
    data = {
        "subscriber_count": [subscriber_count],
        "hours_since_upload": [hours_since_upload],
        "video_length": [video_length],
        "view_count": [view_count],
        "category": [category],
        "like_count": [like_count],
        "comment_count": [comment_count],
        "day_of_week": [day_of_week],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
        "likes_per_subscriber": [likes_per_subscriber],
    }
    df = pd.DataFrame(data)
    log_pred = pipeline.predict(df)[0]
    views_7d_pred = float(np.expm1(log_pred))  # back to views space
    return views_7d_pred

# Example call
# pred = predict_views_day7(
#    subscriber_count=100000,
#    hours_since_upload=24,
#    is_short=0,
#    view_count=50000,
#    category="Entertainment",
#    like_count=3000,
#    comment_count=300
#    day_of_week="Monday",
#    hour_sin=0.5,
#    hour_cos=0.866,
#    likes_per_subscriber=0.03,
#)
#print("Predicted 7-day views:", round(pred))