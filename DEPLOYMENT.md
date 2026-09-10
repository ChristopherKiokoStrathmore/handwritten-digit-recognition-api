# Deployment record

| | |
|---|---|
| **Live URL** | https://mnist-live.onrender.com |
| **Platform** | Render, Docker web service, free instance, region `oregon` |
| **Service ID** | `srv-dah81kp42hec73f6otu0` |
| **Source repo** | https://github.com/ChristopherKiokoStrathmore/handwritten-digit-recognition-api |
| **Branch** | `main` |
| **Deployed commit** | `a5350f8c9eafe156769c906eca7908b032c9361f` (deploy `dep-dah87amq1p3s73b1r780`) |
| **Image** | `python:3.11-slim` base, 423,029,645 bytes (403 MiB) on disk, built and verified locally before deploying |
| **Model in image** | `models/final_fc_model.keras`, 6,459,888 bytes, saved by Keras 3.13.2 |
| **Runtime** | TensorFlow-CPU 2.21.0 / Keras 3.15.1, FastAPI 0.128.0, uvicorn 0.40.0 |
| **Container port** | `$PORT` when the host injects one (Render does), otherwise 7860 |
| **Health check path** | `/health` |
| **Verified at** | 2026-09-10 10:14 UTC |

## Why Render and not Hugging Face Spaces

The original target was `https://kioko1-mnist-live.hf.space`. Hugging Face now returns
**HTTP 402 Payment Required** when creating a Docker Space on a free account:

> Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free
> cpu-basic requires a PRO subscription.

This was confirmed to be a plan limit and not a permissions problem: the same token
creates a **static** Space successfully (HTTP 200) and only `sdk: docker` is refused.
The account reports `isPro: false`.

The Hugging Face front matter and the 7860 default are deliberately left in place, so
if the account is upgraded later this same repo deploys to Spaces unchanged.

## Verification

Every command below was run against the public URL. The raw responses are saved in
`deployment_evidence/`.

### 1. Health check

```bash
curl -i https://mnist-live.onrender.com/health
```

```
HTTP/1.1 200 OK

{"status":"ok","model":"/app/models/final_fc_model.keras"}
```

### 2. Prediction on a real 28x28 digit

The payload is MNIST test-set index 0, whose true label is **7**. The full pixel grid
is in `deployment_evidence/request_digit_used.json` and in `sample_digit.json`.

```bash
# build the request body from the committed fixture
python -c "import json;d=json.load(open('sample_digit.json'));open('digit.json','w').write(json.dumps({'pixels':d['pixels']}))"

curl -X POST https://mnist-live.onrender.com/predict \
     -H "Content-Type: application/json" \
     --data @digit.json
```

```json
{"predicted_digit":7,"confidence":1.0,"route":"auto","all_probabilities":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]}
```

Predicted 7, matching the true label, at full confidence.

### 3. A malformed request is refused rather than guessed at

```bash
curl -X POST https://mnist-live.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"pixels": [[0,0],[0,0]]}'
```

```json
{"detail":"Expected a 28x28 image, but got shape (2, 2). Refusing to guess."}
```

Returned HTTP 422. `preprocessing.py` raises on a wrong shape and the API surfaces
that instead of returning a meaningless prediction.

### 4. Smoke test against the live service

```bash
python smoke_test.py https://mnist-live.onrender.com
```

```
fixture: sample_digit.json  (MNIST test set index 0)  true label = 7
GET  https://mnist-live.onrender.com/health -> 200 {"status": "ok", "model": "/app/models/final_fc_model.keras"}
POST https://mnist-live.onrender.com/predict -> 200
     predicted_digit = 7
     confidence      = 1.0
     route           = auto
POST https://mnist-live.onrender.com/predict (bad 32x32 shape) -> 422 (correctly refused)

SMOKE TEST PASSED - all checks green.
```

### 5. All ten digits

```
All ten digits sent to the live API (first test-set example of each):

  test idx     3  true 0 -> predicted 0  conf 0.9999  auto         OK
  test idx     2  true 1 -> predicted 1  conf 0.9998  auto         OK
  test idx     1  true 2 -> predicted 2  conf 1.0000  auto         OK
  test idx    18  true 3 -> predicted 8  conf 0.5401  human_review MISS
  test idx     4  true 4 -> predicted 4  conf 0.9998  auto         OK
  test idx     8  true 5 -> predicted 5  conf 0.9932  auto         OK
  test idx    11  true 6 -> predicted 6  conf 1.0000  auto         OK
  test idx     0  true 7 -> predicted 7  conf 1.0000  auto         OK
  test idx    61  true 8 -> predicted 8  conf 0.9806  auto         OK
  test idx     7  true 9 -> predicted 9  conf 0.9999  auto         OK

9/10 correct. The single miss falls below the 0.60 confidence threshold
and is routed to human_review rather than asserted, which is the intended behaviour.
```

## Evidence files

| File | What it holds |
|---|---|
| `live_health_response.json`, `live_health_headers.txt` | `/health` body and the 200 status line |
| `live_predict_response.json`, `live_predict_headers.txt` | `/predict` body and headers |
| `live_predict_badshape_response.json` | the 422 refusal |
| `live_smoke_test.txt` | full smoke-test transcript against the public URL |
| `live_all_digits.txt` | all ten digits through the live API |
| `request_digit_used.json` | the exact 28x28 grid that was sent |
| `local_*` | the same checks against the local container, run before deploying |

## Reproducing the deployment

`render.yaml` describes the service, so it can be recreated from this repo.

### Redeploying after a push

The service was created from the public repo URL rather than through Render's
GitHub app, so no webhook is installed and **a push to `main` does not rebuild the
service on its own**, despite `autoDeploy` being set. Either install the Render
GitHub app on the repository to enable it, or trigger a build explicitly:

```bash
curl -X POST \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}' \
  https://api.render.com/v1/services/srv-dah81kp42hec73f6otu0/deploys
```

Or use the **Manual Deploy** button in the Render dashboard.

## Known limitation

The Render free instance sleeps after 15 minutes without traffic. The first request
after a sleep takes roughly 50 seconds while the container restarts and TensorFlow
reloads the model; every request after that is fast. A paid instance removes this.
