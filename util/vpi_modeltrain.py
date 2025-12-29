import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

from util.model_dataset import merge_datasets, adjust_dataset


# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

class Log1pTransform:
    def fit(self, y):  # API parity
        return self
    def transform(self, y):
        return np.log1p(y)
    def inverse_transform(self, y):
        return np.expm1(y)

_DOW_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

input_dim = 5
# -------------------------------
# Model
# -------------------------------
class ViewPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------------
# Loss factory
# -------------------------------
def make_loss(name="huber"):
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    return nn.SmoothL1Loss(beta=1.0)  # Huber

# -------------------------------
# Feature builder (3 features)
# -------------------------------
def build_features(df: pd.DataFrame) -> np.ndarray:
    subs = np.log1p(df["subscriber_count"].astype("float32").values)
    hors_since_upload_raw = df["hours_since_upload"].astype("float32").values
    hsu = np.log1p(np.maximum(hors_since_upload_raw, 0.0) + 1e-6)
    likes = np.log1p(df["like_count"].astype("float32").values)
    comments = np.log1p(df["comment_count"].astype("float32").values)
    like_per_sub = np.log1p(df["like_count"] / df["subscriber_count"]).astype("float32")
    #hour_sin = df["hour_sin"].astype("float32")
    #hour_cos = df["hour_cos"].astype("float32")
    X = np.stack([subs, hsu, likes, comments, like_per_sub], axis=1).astype(np.float32)
    return X

# -------------------------------
# Grouped split helper
# -------------------------------
def grouped_train_val_test_split(X, y, groups, test_size=0.2, val_size=0.1, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    groups_trainval = groups[trainval_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups=groups_trainval))
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test

# -------------------------------
# Training one model
# -------------------------------
def train_one_model(
    df,
    loss_name="huber",
    max_epochs=400,
    batch_size=256,
    lr=1e-3,
    patience=5,
    winsorize_top_q: float = 0.0  # e.g., 0.005 to cap top 0.5%
):
    df = df.copy()

    # hygiene
    df["subscriber_count"] = df["subscriber_count"].astype("int64")
    df = df[df["view_count"] > 0]
    df["hours_since_upload"] = df["hours_since_upload"].round(2)

    # optional: tame extreme tails
    if winsorize_top_q and 0 < winsorize_top_q < 1:
        cap = df["view_count"].quantile(1 - winsorize_top_q)
        df.loc[:, "view_count"] = np.minimum(df["view_count"].values, cap)

    # features (3) + target (log1p)
    X_raw = build_features(df)
    y_raw = df["view_count"].astype(np.float32).values.reshape(-1, 1)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw).astype(np.float32)

    scaler_y = Log1pTransform()
    y_scaled = scaler_y.transform(y_raw).astype(np.float32)

    # grouped split by video_id if present; else fallback to random split
    if "video_id" in df.columns:
        groups = df["video_id"].astype(str).values
        X_train, X_val, X_test, y_train, y_val, y_test = grouped_train_val_test_split(
            X_scaled, y_scaled, groups, test_size=0.2, val_size=0.1, random_state=42
        )
    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.1, random_state=42
        )

    # loaders
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                              batch_size=batch_size, shuffle=False)

    # model / opt / sched
    model = ViewPredictor(input_dim)
    criterion = make_loss(loss_name)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val {best_val:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test metrics
    model.eval()
    preds_log, targs_log = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds_log.append(model(xb).numpy())
            targs_log.append(yb.numpy())
    preds_log = np.vstack(preds_log)
    targs_log = np.vstack(targs_log)

    log_mse = ((targs_log - preds_log) ** 2).mean()
    log_mae = np.abs(targs_log - preds_log).mean()

    preds = scaler_y.inverse_transform(preds_log).ravel()
    targets = scaler_y.inverse_transform(targs_log).ravel()

    mse = ((targets - preds) ** 2).mean()
    mae = np.abs(targets - preds).mean()
    mape = (2 * np.abs(preds - targets) / np.abs(targets) + 1e-9).mean() * 100
    smape = (2 * np.abs(preds - targets) / (np.abs(preds) + np.abs(targets) + 1e-9)).mean() * 100
    rmsle = np.sqrt(np.mean((np.log1p(preds + 1e-9) - np.log1p(targets + 1e-9))**2))

    print(f"Test (log space)  MSE: {log_mse:.4f} | MAE: {log_mae:.4f}")
    print(f"Test (original)   MSE: {mse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}% | sMAPE: {smape:.2f}% | rmsle: {rmsle:.2f}")

    return model, scaler_X, scaler_y

# -------------------------------
# Prediction helpers
# -------------------------------
def _featurize_for_predict(subs: float, hours: float, likes:float, comments:float) -> np.ndarray:
    subs = float(subs)
    hours = float(hours)
    likes = float(likes)
    comments = float(comments)
    log_subs = np.log1p(subs)
    log_hours = np.log1p(max(hours, 0.0) + 1e-6)
    log_likes = np.log1p(likes)
    log_comments = np.log1p(comments)
    return np.array([[log_subs, log_hours, log_likes, log_comments]], dtype=np.float32)

def predict_views_model(model, scaler_X, scaler_y, subs, hours, likes, comments):
    X_in = _featurize_for_predict(subs, hours, likes, comments)
    X_scaled = scaler_X.transform(X_in)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred_log = model(x_tensor).numpy()
    pred = scaler_y.inverse_transform(pred_log)[0][0]
    return int(max(0, round(pred)))  # non-negative

def predict_views(subs, hours, is_short, likes, comments):
    if is_short:
        return predict_views_model(model_short, scaler_short_X, scaler_short_y, subs, hours, likes, comments)
    else:
        return predict_views_model(model_long, scaler_long_X, scaler_long_y, subs, hours, likes, comments)

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    df = merge_datasets()
    df = adjust_dataset(df)

    CATEGORIES = ["Entertainment", "Film", "Gaming", "Knowledge", "Life", "Music", "News" "Sports"]

    for cat in CATEGORIES:
        df_cat = df[df["category"] == cat]

        # Split shorts and long videos
        df_shorts = df_cat[df_cat["is_short"] == 1]
        df_long   = df_cat[df_cat["is_short"] == 0]

        print(f"\nTraining {cat} Shorts Model...")
        model_short, scaler_short_X, scaler_short_y = train_one_model(
            df_shorts, loss_name="huber", max_epochs=400, batch_size=256, lr=1e-3, patience=5, winsorize_top_q=0.0  # try 0.005 to cap top 0.5% if tails dominate
        )
        shorts_bundle = {
            "model_arch": {"input_dim": input_dim},  # must match training features
            "state_dict": model_short.state_dict(),
            "scaler_X": scaler_short_X,
            "scaler_y": scaler_short_y,
        }
        joblib.dump(shorts_bundle, f"{cat.lower()}_short.pkl")

        print("\nTraining Long Model...")
        model_long, scaler_long_X, scaler_long_y = train_one_model(
            df_long, loss_name="huber", max_epochs=400, batch_size=256, lr=1e-3, patience=5, winsorize_top_q=0.0
        )
        long_bundle = {
            "model_arch": {"input_dim": input_dim},
            "state_dict": model_long.state_dict(),
            "scaler_X": scaler_long_X,
            "scaler_y": scaler_long_y,
        }
        joblib.dump(long_bundle, f"{cat.lower()}_long.pkl")

    print("\n✅ All category models saved (short/long)")

    # Example prediction
    #print("\nExample predictions:")
    #print("Short:", predict_views(10000, 6, is_short=1, likes=500))
    #print("Long: ", predict_views(10000, 6, is_short=0, likes=500))
