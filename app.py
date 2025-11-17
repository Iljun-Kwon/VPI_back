import joblib
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# import your predictor (from your uploaded file)
from util.bundle_predict import predict_single
from util.vpi_module import VPIVideoIn, VPIOut, run_vpi
from util.pred7_module import Pred7In, Pred7Out, run_pred7
from util import train_model, bundle_predict

# ---- Config (use env or defaults) ----
SHORTS_BUNDLE = os.getenv("SHORTS_BUNDLE", "model/shorts_bundle.pkl")
LONG_BUNDLE   = os.getenv("LONG_BUNDLE", "model/long_bundle.pkl")
PORT          = int(os.getenv("PORT", "5001"))

app = FastAPI(title="VPI Predictor")

# CORS so your frontend can call from http://localhost:5173 or similar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request model (loose) ----
# Your frontend might send extra fields; we’ll accept flexibly.
class PredictIn(BaseModel):
    subscriber_count: float
    hours_since_upload: float
    like_count: float
    is_short: bool | None = None  # optional flag; defaults to long if omitted
    category: str


# ---- Endpoint that matches: POST /predict/views ----
@app.post("/predict/views", tags=["predict"])
async def predict_views(body: PredictIn):
    try:
        # choose bundle
        #bundle_path = SHORTS_BUNDLE if body.is_short else LONG_BUNDLE
        bundle_path = bundle_predict.get_bundle_path(body.category, body.is_short, base_dir="model")

        # run prediction
        y = predict_single(
            bundle_path=bundle_path,
            subs=body.subscriber_count,
            hours=body.hours_since_upload,
            like_count=body.like_count
        )

        # match frontend expectation
        return {"predicted_view_count": int(y)}

    except Exception as e:
        # return an error JSON (non-OK) so your frontend gets {error: "..."}
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/predict/vpi", response_model=List[VPIOut], tags=["predict"])
async def predict_vpi(payload: List[VPIVideoIn]) -> List[VPIOut]:
    try:
        return run_vpi(payload)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/pred7", tags=["predict"])
async def predict_pred7(payload: List[Pred7In]) -> List[Pred7Out]:
    try:
        return run_pred7(payload)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        # invalid upload_date, unsupported category_id, etc.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: health check
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>VPI Predictor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
    h1 { margin-bottom: 8px; }
    .row { display: grid; grid-template-columns: 180px 1fr; gap: 10px; margin: 8px 0; align-items: center; }
    input[type="number"] { width: 180px; padding: 6px; }
    label { font-weight: 600; }
    button { padding: 8px 14px; border: 0; border-radius: 8px; cursor: pointer; }
    .primary { background: #2563eb; color: white; }
    .ghost { background: #f3f4f6; }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin: 16px 0; }
    .muted { color: #6b7280; }
    #trainResult, #predictResult { white-space: pre-wrap; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>✅ VPI Predictor</h1>
  <p class="muted">Quickly train and test your models from this page.</p>

  <!-- Train -->
  <div class="card">
    <h2>🔧 Train</h2>
    <p class="muted">Triggers <code>GET /train</code> to build and save both bundles.</p>
    <button class="primary" onclick="train()">Train Models</button>
    <button class="primary" onclick="data()">Remake Data</button>
    <button class="ghost" onclick="health()">Check Health</button>
    <div id="trainResult"></div>
  </div>

  <!-- Predict -->
  <div class="card">
    <h2>📈 Predict Views</h2>
    <div class="row">
      <label for="subscriber_count">Subscriber Count</label>
      <input id="subscriber_count" type="number" step="1" value="100000" />
    </div>
    <div class="row">
      <label for="hours_since_upload">Hours Since Upload</label>
      <input id="hours_since_upload" type="number" step="0.01" value="24" />
    </div>
    <div class="row">
      <label for="like_count">Like Count</label>
      <input id="like_count" type="number" step="1" value="1000" />
    </div>
    <div class="row">
      <label for="is_short">Is Short?</label>
      <input id="is_short" type="checkbox" />
    </div>
    <div class="row">
      <label for="category">Category</label>
      <select id="category">
        <option>Entertainment</option>
        <option>Film</option>
        <option>Gaming</option>
        <option>Knowledge</option>
        <option>Life</option>
        <option>Music</option>
        <option>News</option>
        <option>Sports</option>
      </select>
    </div>
    <button class="primary" onclick="predict()">Predict</button>
    <div id="predictResult"></div>
  </div>

  <script>
    async function train() {
      const out = document.getElementById("trainResult");
      out.textContent = "⏳ Training started...";
      try {
        const res = await fetch("/train");
        const data = await res.json();
        out.textContent = res.ok ? (data.message || "✅ Done") : ("❌ " + (data.error || JSON.stringify(data)));
      } catch (err) {
        out.textContent = "❌ " + err;
      }
    }

    async function health() {
      const out = document.getElementById("trainResult");
      out.textContent = "⏳ Checking health...";
      try {
        const res = await fetch("/health");
        const data = await res.json();
        out.textContent = JSON.stringify(data);
      } catch (err) {
        out.textContent = "❌ " + err;
      }
    }

    async function predict() {
      const out = document.getElementById("predictResult");
      out.textContent = "⏳ Predicting...";
      const subscriber_count = Number(document.getElementById("subscriber_count").value);
      const hours_since_upload = Number(document.getElementById("hours_since_upload").value);
      const like_count = Number(document.getElementById("like_count").value);
      const is_short = document.getElementById("is_short").checked;
      const category = document.getElementById("category").value;

      // basic guardrails
      if (!Number.isFinite(subscriber_count) || !Number.isFinite(hours_since_upload) || !Number.isFinite(like_count)) {
        out.textContent = "❌ Please enter valid numeric values.";
        return;
      }

      try {
        const res = await fetch("/predict/views", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ subscriber_count, hours_since_upload, like_count, is_short, category })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          out.textContent = "❌ " + (data.error || JSON.stringify(data));
          return;
        }
        const pretty = new Intl.NumberFormat().format(model.predicted_view_count);
        out.textContent = `🎯 Predicted Views: ${pretty}`;
        } catch (err) {
        out.textContent = "❌ " + err;
        }
      }
  </script>
</body>
</html>
"""

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/data")
def make_data():
    df = train_model.merge_datasets()
    df = train_model.adjust_dataset(df)
@app.get("/train")
def run_training():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    df = train_model.merge_datasets()
    df = train_model.adjust_dataset(df)

    CATEGORIES = ["Entertainment", "Film", "Gaming", "Knowledge", "Life", "Music", "News", "Sports"]
    input_dim = 4

    for cat in CATEGORIES:
        df_cat = df[df["category"] == cat]

        # Split shorts and long videos
        df_shorts = df_cat[df_cat["is_short"] == 1]
        df_long = df_cat[df_cat["is_short"] == 0]

        print(f"\nTraining {cat} Shorts Model...")
        model_short, scaler_short_X, scaler_short_y = train_model.train_one_model(
            df_shorts, loss_name="huber", max_epochs=400, batch_size=256, lr=1e-3, patience=5, winsorize_top_q=0.0
            # try 0.005 to cap top 0.5% if tails dominate
        )
        shorts_bundle = {
            "model_arch": {"input_dim": input_dim},  # must match training features
            "state_dict": model_short.state_dict(),
            "scaler_X": scaler_short_X,
            "scaler_y": scaler_short_y,
        }
        joblib.dump(shorts_bundle, f"{cat.lower()}_short.pkl")

        print(f"\nTraining {cat} Long Model...")
        model_long, scaler_long_X, scaler_long_y = train_model.train_one_model(
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