# infer_bundle.py
# old version when used with model/shorts_bundle.pkl and model/long_bundle.pkl
import os
import sys
import numpy as np
import torch
from functools import lru_cache
import joblib

from util.vpi_modeltrain import ViewPredictor, Log1pTransform

sys.modules['__main__'].__dict__['Log1pTransform'] = Log1pTransform

def _featurize_for_predict(subs: float, hours: float, likes: float, comments:float):
    subs = float(subs); hours = float(hours)
    log_subs  = np.log1p(subs)
    log_hours = np.log1p(max(hours, 0.0) + 1e-6)
    log_likes = np.log1p(likes)
    log_comments = np.log1p(comments)
    return np.array([[log_subs, log_hours, log_likes, log_comments]], dtype=np.float32)

@lru_cache(maxsize=2)
def load_bundle(path):
    art = joblib.load(path)  # contains: model_arch, state_dict, scaler_X, scaler_y
    model = ViewPredictor(input_dim=art["model_arch"]["input_dim"])
    model.load_state_dict(art["state_dict"])
    model.eval()
    return model, art["scaler_X"], art["scaler_y"]

def predict_single(bundle_path, subs, hours, like_count, comment_count) -> int:
    model, scaler_X, scaler_y = load_bundle(bundle_path)
    X = _featurize_for_predict(subs, hours, like_count, comment_count)
    Xs = scaler_X.transform(X)
    with torch.no_grad():
        pred_log = model(torch.tensor(Xs, dtype=torch.float32)).numpy()
    pred = scaler_y.inverse_transform(pred_log)[0][0]
    return int(max(0, round(pred)))

CATEGORIES = ["entertainment", "film", "gaming", "knowledge", "life", "music", "news", "sports"]

def get_bundle_path(category: str, is_short: bool, base_dir: str = "model") -> str:
    """
    Returns path like models/sports_short.pkl
    """
    cat = category.lower()
    if cat not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    kind = "short" if is_short else "long"
    return os.path.join(base_dir, f"{cat}_{kind}.pkl")

if __name__ == "__main__":
    # Example usage
    subs_short, hours_short = 100000, 18
    subs_long, hours_long = 1000000, 24

    #views_short = predict_single("model/shorts_bundle.pkl", subs=subs_short, hours=hours_short)
    #views_long = predict_single("model/long_bundle.pkl", subs=subs_long, hours=hours_long)

    #print(f"For {subs_short} subscribers in {hours_short} hours (Shorts): {views_short} Views")
    #print(f"For {subs_long} subscribers in {hours_long} hours (Long): {views_long} Views")
