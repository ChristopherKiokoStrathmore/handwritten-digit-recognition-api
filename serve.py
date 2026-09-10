"""
serve.py  —  the web server that makes your model "live".

Run it with:   uvicorn serve:app --reload --port 8000
Then open:     http://localhost:8000     (the draw-a-digit page)
API docs at:   http://localhost:8000/docs (auto-generated, interactive)

What this does, in plain terms:
  - loads your trained model ONCE when the server starts (never per request)
  - serves a webpage where you draw a digit
  - exposes /predict, which takes a 28x28 image and returns the digit + confidence
  - refuses malformed input with a clear error (reliability)
  - routes low-confidence predictions to "please review" instead of guessing
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieten TensorFlow startup noise

from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# THE shared transform — the exact same one training used. No second copy.
from preprocessing import preprocess_image

# --- Configuration in one place (mirrors the notebook's Section 9.1) ---
# Paths are resolved relative to THIS file, not to wherever you happened to run
# uvicorn from. A relative "models/..." would silently break the moment you
# started the server from a different directory.
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

CONFIDENCE_THRESHOLD = 0.60   # below this, we ask a human rather than trust the guess
                              # (0.60 is gentle so the demo rarely abstains; raise it
                              #  toward 0.90 for a real system where mistakes cost money)


def find_model() -> Path:
    """
    Locate the trained model, in order of preference:
      1. whatever MODEL_PATH env var says (lets Docker/cloud point elsewhere)
      2. models/final_model.keras   (the name the notebook suggests)
      3. any single *.keras file in models/  (so a differently-named export
         like final_fc_model.keras still just works)
    """
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        candidate = Path(env_path)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"MODEL_PATH is set to {candidate}, but no file is there.")

    preferred = MODELS_DIR / "final_model.keras"
    if preferred.exists():
        return preferred

    found = sorted(MODELS_DIR.glob("*.keras"))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise FileNotFoundError(
            f"Found several models in {MODELS_DIR}: {[f.name for f in found]}. "
            f"Rename the one you want to final_model.keras, or set the MODEL_PATH "
            f"environment variable to pick one."
        )
    raise FileNotFoundError(
        f"No .keras model found in {MODELS_DIR}. Put your trained model there "
        f"(final_model.keras), or set the MODEL_PATH environment variable."
    )


# --- Load the model ONCE, at startup ---
MODEL_PATH = find_model()
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

app = FastAPI(title="MNIST Digit Classifier", version="1.0")


# --- Request and response shapes (FastAPI validates these automatically) ---
class PredictRequest(BaseModel):
    # a 28x28 grid of numbers 0-255
    pixels: List[List[float]]

class PredictResponse(BaseModel):
    predicted_digit: int
    confidence: float
    route: str                 # "auto" (trust it) or "human_review" (too unsure)
    all_probabilities: List[float]


@app.get("/health")
def health():
    """A simple 'are you alive?' check. Cloud hosts ping this."""
    return {"status": "ok", "model": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Take a 28x28 image, return the predicted digit and how confident we are."""
    # 1. Preprocess with the SHARED transform (raises ValueError on bad shape)
    try:
        x = preprocess_image(req.pixels)
    except ValueError as e:
        # 422 = "I understood the request but the data is unprocessable"
        raise HTTPException(status_code=422, detail=str(e))

    # 2. Run the model
    probabilities = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    # 3. Decide whether to trust it or route to a human
    route = "auto" if confidence >= CONFIDENCE_THRESHOLD else "human_review"

    return PredictResponse(
        predicted_digit=digit,
        confidence=round(confidence, 4),
        route=route,
        all_probabilities=[round(float(p), 4) for p in probabilities],
    )


# --- Serve the draw-a-digit webpage at the root URL ---
# (kept last so the API routes above take priority)
static_dir = BASE_DIR / "static"   # again: relative to this file, not the cwd
if static_dir.exists():
    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
