# pred7_module.py using xgboost module
import math
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator, field_validator
from util.vpi_module import catid_to_group  # reuse the same mapping util
from util.xgboostpred import predict_views_day7  # your existing model helper
from util.xgboostbaselinepred import predict_baseline_views_day7


class Pred7In(BaseModel):
    # Required inputs (client sends these)
    id: str = Field(..., description="Video id")
    subscriber_count: float = Field(..., ge=0)
    upload_date: str = Field(..., description="ISO 8601, e.g. '2025-11-13T13:05:22Z'")
    video_length: float = Field(..., ge=0)
    view_count: float = Field(..., ge=0)
    like_count: float = Field(..., ge=0)
    comment_count: float = Field(..., ge=0)
    category_id: int = Field(..., description="YouTube videoCategoryId as integer")
    # Server-computed (not in request)
    hours_since_upload: Optional[float] = None
    day_of_week: Optional[str] = None
    hour_sin: Optional[float] = None
    hour_cos: Optional[float] = None
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

    @validator("hour_sin", always=True, pre=True)
    def _derive_hour_sin(cls, v, values):
        s = values.get("upload_date")
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            hour = dt.hour
            minute = dt.minute
            hour_float = hour + minute / 60.0
            angle = 2.0 * math.pi * hour_float / 24.0
            return math.sin(angle)
        except Exception:
            return 0.0

    @validator("hour_cos", always=True, pre=True)
    def _derive_hour_cos(cls, v, values):
        s = values.get("upload_date")
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            hour = dt.hour
            minute = dt.minute
            hour_float = hour + minute / 60.0
            angle = 2.0 * math.pi * hour_float / 24.0
            return math.cos(angle)
        except Exception:
            return 1.0

    @validator("category_group", always=True, pre=True)
    def _derive_category_group(cls, v, values):
        cid = values.get("category_id")
        try:
            grp = catid_to_group(int(cid)) if cid is not None else None
        except Exception:
            grp = None
        if grp is None:
            raise ValueError(f"Unsupported category_id={cid}")
        return grp

    def likes_per_subscriber(self) -> float:
        # Always computed on server; never accepted from the client
        denom = float(self.subscriber_count) + 1e-9
        val = float(self.like_count) / denom
        if math.isnan(val) or val < 0:
            return 0.0
        return val

    def is_short_numeric(self) -> float:
        return 1.0 if self.is_short else 0.0


class Pred7Out(BaseModel):
    id: str
    predicted_7day_views: int
    FI: float


def run_pred7(payload: List[Pred7In]) -> List[Pred7Out]:
    out: List[Pred7Out] = []

    for item in payload:
        # Derive server-side fields
        likes_per_subscriber = float(item.like_count) / (float(item.subscriber_count) + 1e-9)
        if math.isnan(likes_per_subscriber) or likes_per_subscriber < 0:
            likes_per_subscriber = 0.0

        # Call your XGBoost predictor
        y_hat = predict_views_day7(
            subscriber_count=item.subscriber_count,
            hours_since_upload=item.hours_since_upload,
            video_length=item.video_length,
            view_count=item.view_count,
            category=item.category_group,
            like_count=item.like_count,
            comment_count=item.comment_count,
            day_of_week=item.day_of_week,
            hour_sin=item.hour_sin,
            hour_cos=item.hour_cos,
            likes_per_subscriber=likes_per_subscriber,
        )

        base_y_hat = predict_baseline_views_day7(
            subscriber_count = item.subscriber_count,
            video_length = item.video_length,
            category = item.category_group,
            day_of_week = item.day_of_week,
            hour_sin = item.hour_sin,
            hour_cos = item.hour_cos,
        )

        predicted_7day_views=int(round(float(y_hat)))
        FI = int(round(float(y_hat))) / (int(round(float(base_y_hat))) + 1e-9)

        out.append(
            Pred7Out(
                id=item.id,
                predicted_7day_views=predicted_7day_views,
                FI = FI
            )
        )

    return out
