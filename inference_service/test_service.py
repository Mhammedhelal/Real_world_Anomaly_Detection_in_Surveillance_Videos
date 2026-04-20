"""
inference_service/test_service.py
----------------------------------
Quick verification script — run after docker-compose up to confirm the
service is healthy and inference works end-to-end.

Usage
-----
    python inference_service/test_service.py
    python inference_service/test_service.py --url http://192.168.1.100:8000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.request
import urllib.error
from typing import List

import numpy as np
from PIL import Image


BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(width: int = 224, height: int = 224) -> str:
    """Create a random RGB frame and return it as a base64 JPEG string."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _put(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health() -> bool:
    print("\n1. Health check …", end=" ")
    try:
        data = _get("/health")
        ok = data.get("model_loaded", False)
        print(f"✅ status={data['status']}  device={data['device']}  threshold={data['threshold']}")
        return ok
    except Exception as exc:
        print(f"❌ {exc}")
        return False


def test_predict() -> bool:
    print("2. Prediction …", end=" ")
    try:
        frames = [_make_frame() for _ in range(16)]
        data = _post("/predict", {"frames": frames})
        score = data["anomaly_score"]
        cls   = data["predicted_class"]
        ms    = data["inference_time_ms"]
        loc   = data.get("localisation")
        loc_str = f"  boxes={loc['num_detections']}" if loc else "  no boxes (normal)"
        print(f"✅ score={score:.3f}  class={cls}  {ms:.0f}ms{loc_str}")
        return True
    except Exception as exc:
        print(f"❌ {exc}")
        return False


def test_threshold_update() -> bool:
    print("3. Threshold update …", end=" ")
    try:
        original = _get("/threshold")["threshold"]
        _put("/threshold", {"threshold": 0.42})
        updated = _get("/threshold")["threshold"]
        # restore
        _put("/threshold", {"threshold": original})
        ok = abs(updated - 0.42) < 1e-5
        print(f"✅ set 0.42  →  got {updated:.4f}")
        return ok
    except Exception as exc:
        print(f"❌ {exc}")
        return False


def test_invalid_request() -> bool:
    print("4. Invalid request handling …", end=" ")
    try:
        _post("/predict", {"frames": []})   # should return 422
        print("❌ expected error but got 200")
        return False
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 422):
            print(f"✅ correctly rejected with HTTP {exc.code}")
            return True
        print(f"❌ unexpected HTTP {exc.code}")
        return False
    except Exception as exc:
        print(f"❌ {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BASE_URL, help="Service base URL")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url.rstrip("/")

    print(f"\n{'='*55}")
    print(f" Anomaly Detection Service — Smoke Tests")
    print(f" Target: {BASE_URL}")
    print(f"{'='*55}")

    results = {
        "health":    test_health(),
        "predict":   test_predict(),
        "threshold": test_threshold_update(),
        "invalid":   test_invalid_request(),
    }

    print(f"\n{'='*55}")
    passed = sum(results.values())
    total  = len(results)
    print(f" Results: {passed}/{total} passed")
    for name, ok in results.items():
        print(f"   {'✅' if ok else '❌'}  {name}")
    print(f"{'='*55}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()