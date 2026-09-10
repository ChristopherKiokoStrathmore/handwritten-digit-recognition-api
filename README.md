---
title: MNIST Live — Handwritten Digit Recognition API
emoji: ✏️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Handwritten Digit Recognition API

A fully-connected neural network trained on MNIST, served as a live REST API with a
draw-a-digit web front end. The model artifact ships **inside** the container, so the
service has no runtime dependency on any model registry or object store.

**Live demo:** https://kioko1-mnist-live.hf.space
**Draw a digit:** https://kioko1-mnist-live.hf.space/
**Interactive API docs:** https://kioko1-mnist-live.hf.space/docs

---

## The model

| | |
|---|---|
| Architecture | `Sequential`: 784 → Dense 512 (ReLU) → Dropout → Dense 256 (ReLU) → Dropout → Dense 10 (softmax) |
| Parameters | 535,818 |
| Input | flattened 28×28 grayscale, scaled to 0–1 |
| Artifact | `models/final_fc_model.keras` (6.2 MB, Keras 3.13.2) |
| Framework | TensorFlow 2.21 / Keras 3.15 (CPU-only build) |

The network is a from-scratch dense classifier rather than a convolutional one; the
accompanying notebook benchmarks it against a CNN baseline.

---

## Why `preprocessing.py` is a separate module

The single most common way a model that scored well in a notebook produces garbage in
production is **training/serving skew** — the server preprocessing images differently
from how training did. Nothing crashes; the model just quietly gets worse.

The defence here is structural: training and serving both import the *same*
`preprocess_image()` function. There is no second copy that can drift out of sync.
It also raises `ValueError` on a wrong-shaped input rather than guessing, which the API
surfaces as HTTP 422.

---

## API

### `GET /health`

```json
{ "status": "ok", "model": "/app/models/final_fc_model.keras" }
```

### `POST /predict`

Request — `pixels` must be exactly 28×28, values 0–255:

```json
{ "pixels": [[0, 0, 0, "... 28 values ..."], "... 28 rows ..."] }
```

Response:

```json
{
  "predicted_digit": 7,
  "confidence": 0.9999,
  "route": "auto",
  "all_probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999, 0.0, 0.0]
}
```

| Field | Meaning |
|---|---|
| `predicted_digit` | `argmax` over the 10 class probabilities |
| `confidence` | the winning probability, rounded to 4 dp |
| `route` | `auto` when confidence ≥ 0.60, otherwise `human_review` |
| `all_probabilities` | all 10 class probabilities, in digit order |

`route` is the part that matters operationally: a low-confidence prediction is flagged
for a human instead of being asserted as fact.

**Errors:** a non-28×28 `pixels` array returns **422** with a message naming the shape
it received. Malformed JSON returns 422 from FastAPI's own validation.

### Try it

```bash
curl https://kioko1-mnist-live.hf.space/health

python - <<'PY' > digit.json
import json; d = json.load(open("sample_digit.json")); print(json.dumps({"pixels": d["pixels"]}))
PY

curl -X POST https://kioko1-mnist-live.hf.space/predict \
     -H "Content-Type: application/json" \
     --data @digit.json
```

---

## Running it yourself

### Docker (matches production exactly)

```bash
docker build -t mnist-live .
docker run --rm -p 7860:7860 mnist-live
```

Then open http://localhost:7860.

### Without Docker

```bash
pip install -r requirements.txt
uvicorn serve:app --host 0.0.0.0 --port 7860
```

### Smoke test

Loads a real MNIST digit from `sample_digit.json`, calls `/health` and `/predict`,
checks the prediction against the true label, and confirms a malformed request is
refused with 422. Exit code 0 only if everything passes.

```bash
python smoke_test.py                                   # defaults to localhost:7860
python smoke_test.py https://kioko1-mnist-live.hf.space
```

---

## Layout

```
├── serve.py               FastAPI app: loads the model once at startup, serves /health and /predict
├── preprocessing.py       THE shared transform — imported by both training and serving
├── smoke_test.py          end-to-end check against any running instance
├── sample_digit.json      one real MNIST test digit (index 0, label 7) used as a fixture
├── models/
│   └── final_fc_model.keras
├── static/index.html      draw-a-digit front end, calls /predict with relative URLs
├── Dockerfile             python:3.11-slim, non-root, listens on 7860
├── requirements.txt       exact pins, CPU-only TensorFlow
└── DEPLOYMENT.md          live URL, commit SHA, and verified curl transcripts
```

## Deployment

Hugging Face Spaces, Docker SDK. The container listens on 7860, which matches the
`app_port` declared in the front matter above. `DEPLOYMENT.md` records the live URL,
the deployed commit SHA, and the exact verification commands with their real responses.

## License

MIT
