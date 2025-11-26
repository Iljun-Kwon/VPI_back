import joblib
import numpy as np
import pandas as pd

# Load trained pipeline
pipeline = joblib.load("xgb_7day_views_model.pkl")

def predict_baseline_views_day7(
    subscriber_count,
    video_length,
    category,
    day_of_week,
    hour_sin,
    hour_cos,
):
    data = {
        "subscriber_count": [subscriber_count],
        "video_length": [video_length],
        "category": [category],
        "day_of_week": [day_of_week],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
    }
    df = pd.DataFrame(data)
    log_pred = pipeline.predict(df)[0]
    views_baseline_7d_pred = float(np.expm1(log_pred))  # back to views space
    return views_baseline_7d_pred