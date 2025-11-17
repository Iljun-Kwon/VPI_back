# vpi_module.py
import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator

# Your existing predictor
from util.bundle_predict import predict_single
from util import bundle_predict

# Env (shared)
BUNDLE_DIR = os.getenv("BUNDLE_DIR", "model")
CATEGORY_GROUPS: Dict[str, set] = {
    "film":           {1, 18, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44},
    "knowledge":      {2, 26, 27, 28, 29},
    "music":          {10},
    "life":           {15, 19, 21, 22},
    "sports":         {17},
    "gaming":         {20},
    "entertainment":  {23, 24, 32, 42, 43},
    "news":           {25},
}
def catid_to_group(cid: int) -> str | None:
    for group, idset in CATEGORY_GROUPS.items():
        if cid in idset:
            return group
    return None

# ---------- I/O models ----------
class VPIVideoIn(BaseModel):
    id: str = Field(..., description="Video id")
    actual_views: int = Field(None, ge=0, description="for VPI calc")
    subscriber_count: int = Field(..., ge=0)
    upload_date: str = Field(..., description="ISO 8601, e.g. '2025-10-15T13:05:22Z'")
    like_count: Optional[int] = Field(0, ge=0)
    duration_sec: int = Field(..., ge=0, description="<=100 : shorts; >100 : long-form")
    category_id: int = Field(..., description="YouTube videoCategoryId (int)")

    # Derived (server-side)
    is_short: Optional[bool] = None
    hours_since_upload: Optional[float] = None
    category_group: Optional[str] = None

    @validator("upload_date")
    def _validate_iso(cls, v: str) -> str:
        s = v.strip()
        if s.endswith("Z"):
            return s
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            raise ValueError("upload_date must be ISO 8601, e.g. 2025-10-15T13:05:22Z")

    @validator("is_short", always=True, pre=True)
    def _derive_is_short(cls, v, values):
        return (values.get("duration_sec", 0) <= 100)

    @validator("hours_since_upload", always=True, pre=True)
    def _derive_hours_since_upload(cls, v, values):
        s = values.get("upload_date")
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - dt
            return max(delta.total_seconds() / 3600.0, 0.0)
        except Exception:
            return 0.0

    @validator("category_group", always=True, pre=True)
    def _derive_category_group(cls, v, values):
        cid = values.get("category_id")
        try:
            grp = catid_to_group(int(cid)) if cid is not None else None
        except Exception:
            grp = None
        if grp is None:
            allowed = sorted({c for s in CATEGORY_GROUPS.values() for c in s})
            raise ValueError(f"Unsupported category_id={cid}. Allowed ids: {allowed}")
        return grp

class VPIOut(BaseModel):
    id: str
    vpi: Optional[float]
    pred: Optional[float]

# ---------- helpers ----------
def _bundle_path(category_group: str, is_short: bool) -> str:
    """
    Build the bundle filename like 'entertainment_short.pkl' or 'film_long.pkl'.
    """
    suffix = "short" if is_short else "long"
    filename = f"{category_group}_{suffix}.pkl"
    return os.path.join(BUNDLE_DIR, filename)

def vpi_build_features(v: VPIVideoIn) -> Dict[str, Any]:
    return {
        "subscriber_count": float(v.subscriber_count),
        "hours_since_upload": float(v.hours_since_upload or 0.0),
        "like_count": float(v.like_count or 0),
        "is_short": 1.0 if v.is_short else 0.0,
        "duration_sec": float(v.duration_sec),
        "category_id": float(v.category_id),
    }

def map_to_vpi(features: Dict[str, Any], model_output: float) -> float:
    # Identity for now; replace with your real VPI formula if different.
    try:
        return float(model_output)
    except Exception:
        return 0.0

# ---------- core inference (pure function) ----------
def run_vpi(payload: List[VPIVideoIn]) -> List[VPIOut]:
    out: List[VPIOut] = []
    for v in payload:
        # Pick the bundle by derived group + short/long using your helper
        bundle_path = bundle_predict.get_bundle_path(
            v.category_group, bool(v.is_short), base_dir=BUNDLE_DIR
        )
        # Run your existing predictor with the same signature you use in /predict/views
        y_pred = predict_single(
            bundle_path=bundle_path,
            subs=v.subscriber_count,
            hours=v.hours_since_upload or 0.0,
            like_count=(v.like_count or 0),
        )
        # Map to VPI (identity; swap in your real formula if needed)
        vpi_value = (float(v.actual_views) / y_pred) * 100
        out.append(VPIOut(id=v.id, vpi=vpi_value, pred=y_pred))

    return out
