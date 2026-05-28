"""
face_detector.py — Face recognition REST service
--------------------------------------------------
Runs on http://localhost:5051

Endpoints
---------
POST /api/match         { embedding: [...] }
                        → { matched, name, score }

POST /api/enroll        { name: str, crop_b64: str }
                        → { ok, name, total_photos }

GET  /api/known         → { names: [...] }

DELETE /api/known/:name → { ok, deleted }

POST /api/enroll-file   { name: str, image_path: str }  (CLI / testing helper)
                        → { ok, name }

Dependencies
------------
    pip install flask opencv-python numpy huggingface_hub
"""

import base64
import json
import sys
import threading
from pathlib import Path

import cv2 as cv
import numpy as np
from flask import Flask, jsonify, request

# face_engine.py must be in the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_engine import FaceEngine

# ── Config ────────────────────────────────────────────────────────────────────

PORT      = 5051
DB_PATH   = Path(__file__).resolve().parent / "face_db.json"
# Maximum stored embeddings per person (oldest are dropped)
MAX_PHOTOS_PER_PERSON = 20

# ── Face DB ───────────────────────────────────────────────────────────────────

_db_lock = threading.Lock()


def load_db() -> list:
    """Return list of { name, photos: [{ embedding, preview_b64? }] }"""
    if DB_PATH.exists():
        try:
            data = json.loads(DB_PATH.read_text())
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def save_db(users: list) -> None:
    DB_PATH.write_text(json.dumps(users, indent=2))


# ── Engine (singleton — not thread-safe for concurrent infer, protected below) ─

_engine_lock = threading.Lock()
print("[face_detector] Loading face models…", flush=True)
engine = FaceEngine()
print("[face_detector] Models ready.", flush=True)

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


# ── /api/match ────────────────────────────────────────────────────────────────

@app.route("/api/match", methods=["POST"])
def match():
    """
    Input:  { "embedding": [128 floats] }
    Output: { "matched": bool, "name": str|null, "score": float|null }
    """
    body = request.get_json(silent=True) or {}
    raw_emb = body.get("embedding")

    if not raw_emb or not isinstance(raw_emb, list):
        return jsonify({"error": "embedding (list of floats) required"}), 400

    probe = engine.embedding_from_list(raw_emb)

    with _db_lock:
        users = load_db()

    if not users:
        return jsonify({"matched": False, "name": None, "score": None})

    with _engine_lock:
        best_user, best_score, matched = engine.match_best(probe, users)

    return jsonify({
        "matched": matched,
        "name":    best_user["name"] if best_user else None,
        "score":   round(float(best_score), 4) if best_score is not None else None,
    })


# ── /api/enroll ───────────────────────────────────────────────────────────────

@app.route("/api/enroll", methods=["POST"])
def enroll():
    """
    Input:  { "name": str, "crop_b64": str }   — JPEG face crop, base64-encoded
    Output: { "ok": true, "name": str, "total_photos": int }
    """
    body = request.get_json(silent=True) or {}
    name     = (body.get("name") or "").strip()
    crop_b64 = body.get("crop_b64") or ""

    if not name:
        return jsonify({"error": "name required"}), 400
    if not crop_b64:
        return jsonify({"error": "crop_b64 required"}), 400

    # Decode image
    try:
        img_bytes = base64.b64decode(crop_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv.imdecode(arr, cv.IMREAD_COLOR)
        if image is None:
            raise ValueError("decode failed")
    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400

    # Extract embedding
    with _engine_lock:
        embedding, face_row = engine.extract_embedding(image)

    if embedding is None:
        return jsonify({"error": "no face detected in crop"}), 422

    emb_list = engine.embedding_to_list(embedding)

    # Store in DB
    with _db_lock:
        users = load_db()
        user  = next((u for u in users if u["name"] == name), None)
        if user is None:
            user = {"name": name, "photos": []}
            users.append(user)

        user["photos"].append({"embedding": emb_list})

        # Keep only the most recent N embeddings per person
        if len(user["photos"]) > MAX_PHOTOS_PER_PERSON:
            user["photos"] = user["photos"][-MAX_PHOTOS_PER_PERSON:]

        save_db(users)
        total = len(user["photos"])

    print(f"[face_detector] Enrolled '{name}' ({total} photos total)", flush=True)
    return jsonify({"ok": True, "name": name, "total_photos": total})


# ── /api/enroll-file (helper for CLI testing) ─────────────────────────────────

@app.route("/api/enroll-file", methods=["POST"])
def enroll_file():
    """
    Input:  { "name": str, "image_path": str }
    Output: { "ok": true, "name": str }
    Useful for bulk-enrolling from disk without the UI.
    """
    body       = request.get_json(silent=True) or {}
    name       = (body.get("name") or "").strip()
    image_path = (body.get("image_path") or "").strip()

    if not name or not image_path:
        return jsonify({"error": "name and image_path required"}), 400

    path = Path(image_path)
    if not path.exists():
        return jsonify({"error": f"file not found: {image_path}"}), 404

    image = cv.imread(str(path))
    if image is None:
        return jsonify({"error": "cannot read image"}), 400

    with _engine_lock:
        embedding, face_row = engine.extract_embedding(image)

    if embedding is None:
        return jsonify({"error": "no face detected"}), 422

    emb_list = engine.embedding_to_list(embedding)

    with _db_lock:
        users = load_db()
        user  = next((u for u in users if u["name"] == name), None)
        if user is None:
            user = {"name": name, "photos": []}
            users.append(user)
        user["photos"].append({"embedding": emb_list})
        if len(user["photos"]) > MAX_PHOTOS_PER_PERSON:
            user["photos"] = user["photos"][-MAX_PHOTOS_PER_PERSON:]
        save_db(users)
        total = len(user["photos"])

    return jsonify({"ok": True, "name": name, "total_photos": total})


# ── /api/known ────────────────────────────────────────────────────────────────

@app.route("/api/known", methods=["GET"])
def known():
    """
    Output: { "names": [ { "name": str, "photos": int } ] }
    """
    with _db_lock:
        users = load_db()
    return jsonify({
        "names": [{"name": u["name"], "photos": len(u.get("photos", []))} for u in users]
    })


@app.route("/api/known/<name>", methods=["DELETE"])
def delete_known(name: str):
    """
    Removes a person and all their embeddings from the DB.
    Output: { "ok": true, "deleted": str }
    """
    with _db_lock:
        users   = load_db()
        before  = len(users)
        users   = [u for u in users if u["name"] != name]
        if len(users) == before:
            return jsonify({"error": f"'{name}' not found"}), 404
        save_db(users)

    print(f"[face_detector] Deleted '{name}' from DB", flush=True)
    return jsonify({"ok": True, "deleted": name})


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    with _db_lock:
        users = load_db()
    return jsonify({"ok": True, "enrolled": len(users)})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Atomo face recognition service")
    parser.add_argument("--port", type=int, default=PORT, help="Port to listen on (default 5051)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1)")
    parser.add_argument("--db",   default=str(DB_PATH),  help="Path to face_db.json")
    args = parser.parse_args()

    DB_PATH = Path(args.db)
    print(f"[face_detector] DB: {DB_PATH}", flush=True)
    print(f"[face_detector] Listening on http://{args.host}:{args.port}", flush=True)

    # threaded=True so multiple cameras can enroll/match concurrently.
    # _engine_lock ensures OpenCV model is only used by one thread at a time.
    app.run(host=args.host, port=args.port, threaded=True)
