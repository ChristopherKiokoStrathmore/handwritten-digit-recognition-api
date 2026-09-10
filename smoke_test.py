"""
smoke_test.py — proves a running instance actually works, end to end.

It loads one real MNIST digit from sample_digit.json, calls /health and
/predict, and checks the response against the true label.

Usage:
    python smoke_test.py                                  # localhost:7860
    python smoke_test.py http://localhost:8000
    python smoke_test.py https://kioko1-mnist-live.hf.space

Exit code is 0 only if every check passes, so CI can gate on it.
Standard library only — nothing to install, nothing extra in the image.
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE = "http://localhost:7860"
TIMEOUT = 60  # a cold container has to load TensorFlow before it can answer

BASE_DIR = Path(__file__).resolve().parent
FIXTURE = BASE_DIR / "sample_digit.json"


def _get(url):
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return r.status, json.loads(r.read().decode())


def _post_json(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status, json.loads(r.read().decode())


def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE).rstrip("/")

    fixture = json.loads(FIXTURE.read_text())
    pixels, true_label = fixture["pixels"], fixture["true_label"]

    # The fixture must match the contract preprocessing.py enforces.
    assert len(pixels) == 28 and all(len(r) == 28 for r in pixels), "fixture is not 28x28"
    print(f"fixture: {FIXTURE.name}  ({fixture.get('source', 'n/a')})  true label = {true_label}")

    failures = []

    # --- /health -------------------------------------------------------
    try:
        status, body = _get(f"{base}/health")
        print(f"GET  {base}/health -> {status} {json.dumps(body)}")
        if status != 200:
            failures.append(f"/health returned {status}, expected 200")
        if body.get("status") != "ok":
            failures.append(f"/health status field was {body.get('status')!r}, expected 'ok'")
    except urllib.error.URLError as e:
        failures.append(f"/health unreachable: {e}")
        print(f"GET  {base}/health -> UNREACHABLE ({e})")

    # --- /predict ------------------------------------------------------
    try:
        status, body = _post_json(f"{base}/predict", {"pixels": pixels})
        print(f"POST {base}/predict -> {status}")
        print(f"     predicted_digit = {body.get('predicted_digit')}")
        print(f"     confidence      = {body.get('confidence')}")
        print(f"     route           = {body.get('route')}")

        if status != 200:
            failures.append(f"/predict returned {status}, expected 200")
        for field in ("predicted_digit", "confidence", "route", "all_probabilities"):
            if field not in body:
                failures.append(f"/predict response missing field {field!r}")
        if len(body.get("all_probabilities", [])) != 10:
            failures.append("all_probabilities should have 10 entries")
        if body.get("predicted_digit") != true_label:
            failures.append(
                f"predicted {body.get('predicted_digit')} but the true label is {true_label}"
            )
    except urllib.error.URLError as e:
        failures.append(f"/predict unreachable: {e}")
        print(f"POST {base}/predict -> UNREACHABLE ({e})")

    # --- a malformed request must be rejected, not guessed at -----------
    try:
        _post_json(f"{base}/predict", {"pixels": [[0] * 32] * 32})
        failures.append("a 32x32 image was accepted; it should have been rejected with 422")
        print("POST /predict (32x32) -> 200  <-- WRONG, expected 422")
    except urllib.error.HTTPError as e:
        print(f"POST {base}/predict (bad 32x32 shape) -> {e.code} (correctly refused)")
        if e.code != 422:
            failures.append(f"bad shape gave {e.code}, expected 422")
    except urllib.error.URLError as e:
        failures.append(f"bad-shape check unreachable: {e}")

    print()
    if failures:
        print("SMOKE TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("SMOKE TEST PASSED — all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
