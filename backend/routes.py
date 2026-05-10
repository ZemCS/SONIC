import io
import os
import threading
import subprocess
import spotipy
import torch
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from PIL import Image
from datetime import datetime
from pathlib import Path

from bson import ObjectId
from config import config, UI
from database import (
    db, audio_inferences, lyrics_inferences, fused_inferences, 
    image_inferences, feedback_db, recommendation_logs
)
import state
import spotify_utils

routes_bp = Blueprint("routes", __name__)

# We need a way to access SONG_DATABASE and models. 
# In a real app, these would be in a service or shared state.
# I'll create a `state.py` for this.
device_cache = {} # {user_id: {"id": device_id, "timestamp": datetime}}
DEVICE_CACHE_TTL = 60 # seconds


@routes_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict_image_endpoint():
    from pipeline import get_image_transform
    if state.image_model is None:
        return jsonify({"error": "Image model not loaded"}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        import uuid
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Save image locally
        file_ext = os.path.splitext(file.filename)[1] or ".jpg"
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        image_path = os.path.join(config.UPLOADS_DIR, unique_filename)
        image.save(image_path)
        
        # Image transformation for model
        from pipeline import get_image_transform
        input_tensor = get_image_transform()(image).unsqueeze(0).to(config.DEVICE)

        # Model is already on the correct device from initialization
        # No need to move during inference - just ensure input tensor is on same device
        model_device = next(state.image_model.parameters()).device
        input_tensor = input_tensor.to(model_device)

        with torch.no_grad():
            output = state.image_model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            confidence, idx = torch.max(probs, 0)
            predicted_class = config.IMAGE_CLASSES[idx.item()]
            conf_val = float(confidence.item())

        image_inferences.insert_one({
            "filename": file.filename,
            "saved_filename": unique_filename,
            "predicted_mood": predicted_class,
            "confidence": conf_val,
            "timestamp": datetime.now(),
            "user_id": get_jwt_identity()
        })

        # Get server base URL (we can use request.host_url)
        base_url = request.host_url.rstrip('/')
        image_url = f"{base_url}/uploads/{unique_filename}"

        return jsonify({
            "mood": predicted_class, 
            "confidence": conf_val,
            "image_url": image_url,
            "saved_filename": unique_filename
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": "Failed to process image"}), 500

@routes_bp.route("/classes", methods=["GET"])
def get_model_classes():
    return jsonify({
        "image": config.IMAGE_CLASSES,
        "audio": config.AUDIO_CLASSES,
        "lyrics": config.LYRICS_CLASSES,
    })

@routes_bp.route("/correct", methods=["POST"])
@jwt_required()
def correct_mood_endpoint():
    user_id = get_jwt_identity()
    try:
        data = request.get_json()
        feedback_db.insert_one({
            "user_id": user_id,
            "timestamp": datetime.now(),
            "filename": data.get("filename", "unknown"),
            "file_path": data.get("file_path", "unknown"),
            "original_prediction": data.get("original_prediction"),
            "corrected_mood": data.get("corrected_mood"),
        })
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@routes_bp.route("/recommend", methods=["POST"])
@jwt_required()
def recommend_songs_endpoint():
    data = request.get_json()
    if not data or "mood" not in data:
        return jsonify({"error": "Mood not provided"}), 400

    app_mood = data["mood"].lower()
    target_song_mood = app_mood if app_mood not in ["joy", "joyful"] else "joyful"

    recommendations = [s for s in state.SONG_DATABASE if s["mood"] == target_song_mood]
    recommendations.sort(key=lambda x: x["confidence"], reverse=True)
    response_data = [
        {
            "title": s["title"],
            "artist": s["artist"],
            "spotify_uri": s.get("spotify_uri"),
        }
        for s in recommendations
    ]
    return jsonify(response_data)

@routes_bp.route("/spotify/pause", methods=["POST"])
@jwt_required()
def spotify_pause():
    user_id = get_jwt_identity()
    user = db.users.find_one({"_id": ObjectId(user_id)})
    if not user or "spotify_token" not in user:
        return jsonify({"error": "Spotify not connected"}), 400

    try:
        token_info = user["spotify_token"]
        from spotify_utils import check_and_refresh_token
        token_info = check_and_refresh_token(user_id, token_info)
        sp = spotipy.Spotify(auth=token_info["access_token"])
        sp.pause_playback()
        return jsonify({"status": "paused"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@routes_bp.route("/spotify/status", methods=["GET"])
@jwt_required()
def spotify_status():
    user_id = get_jwt_identity()
    user = db.users.find_one({"_id": ObjectId(user_id)})
    if not user or "spotify_token" not in user:
        return jsonify({"is_connected": False})
    return jsonify({"is_connected": True})

@routes_bp.route("/log_recommendation", methods=["POST"])
@jwt_required()
def log_recommendation_endpoint():
    user_id = get_jwt_identity()
    data = request.get_json()
    if not data or "requested_mood" not in data or "recommended_songs" not in data:
        return jsonify({"error": "Missing mood or songs data"}), 400

    try:
        db["recommendation_logs"].insert_one({
            "user_id": user_id,
            "timestamp": datetime.now(),
            "requested_mood": data["requested_mood"],
            "recommended_songs": data["recommended_songs"],
            "image_url": data.get("image_url"), # Link to the trigger image
        })
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@routes_bp.route("/get_recommendation_history", methods=["GET"])
@jwt_required()
def get_recommendation_history():
    user_id = get_jwt_identity()
    try:
        history = list(
            db["recommendation_logs"]
            .find({"user_id": user_id}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(20)
        )
        for item in history:
            if isinstance(item.get("timestamp"), datetime):
                item["timestamp"] = item["timestamp"].isoformat()
        return jsonify(history), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@routes_bp.route("/get_prediction_history", methods=["GET"])
@jwt_required()
def get_prediction_history():
    user_id = get_jwt_identity()
    try:
        history = list(
            image_inferences
            .find({"user_id": user_id}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(20)
        )
        for item in history:
            if isinstance(item.get("timestamp"), datetime):
                item["timestamp"] = item["timestamp"].isoformat()
        return jsonify(history), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Spotify Endpoints
@routes_bp.route("/spotify/login")
@jwt_required()
def spotify_login():
    from spotify_utils import global_sp_oauth
    sonic_user_id = request.args.get("sonic_user_id", "")
    auth_url = global_sp_oauth.get_authorize_url(state=sonic_user_id)
    return jsonify({"auth_url": auth_url})

@routes_bp.route("/spotify/callback")
def spotify_callback():
    from spotify_utils import global_sp_oauth, MongoCacheHandler, resolve_spotify_uris
    code = request.args.get("code")
    sonic_user_id = request.args.get("state")
    if not code:
        return jsonify({"error": "No code provided"}), 400

    try:
        token_info = global_sp_oauth.get_access_token(code)
    except Exception as e:
        return jsonify({"error": "Failed to get access token/Token expired"}), 500

    if not token_info:
        return jsonify({"error": "Failed to get access token"}), 500

    if sonic_user_id:
        try:
            user_cache = MongoCacheHandler(sonic_user_id)
            user_cache.save_token_to_cache(token_info)
            threading.Thread(
                target=resolve_spotify_uris, args=(sonic_user_id,), daemon=True
            ).start()
        except Exception as e:
            print(f"Invalid SONIC user ID: {sonic_user_id} - {e}")

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Authenticating...</title>
        <script>
            window.location.href = "sonicapp://auth-success";
            setTimeout(function() {{ window.close(); }}, 2000);
        </script>
    </head>
    <body>
        <h2>Spotify Connected!</h2>
        <p>Taking you back to SONIC...</p>
    </body>
    </html>
    """

@routes_bp.route("/spotify/history", methods=["GET"])
@jwt_required()
def spotify_history():
    from spotify_utils import get_spotify_oauth
    user_id = get_jwt_identity()
    sp_oauth = get_spotify_oauth(user_id)
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
    if not token_info:
        return jsonify({"error": "Not authenticated with Spotify"}), 401

    sp = spotipy.Spotify(auth=token_info["access_token"])
    try:
        results = sp.current_user_recently_played(limit=20)
        tracks = []
        for item in results["items"]:
            track = item["track"]
            tracks.append({
                "title": track["name"],
                "artist": track["artists"][0]["name"],
                "spotify_uri": track["uri"],
                "played_at": item["played_at"],
            })
        return jsonify(tracks)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@routes_bp.route("/spotify/download_history", methods=["POST"])
@jwt_required()
def spotify_download_history():
    from spotify_utils import get_spotify_oauth, spotify_download_internal
    user_id = get_jwt_identity()
    sp_oauth = get_spotify_oauth(user_id)
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
    if not token_info:
        return jsonify({"error": "Not authenticated with Spotify"}), 401

    threading.Thread(
        target=spotify_download_internal, args=(user_id,), daemon=True
    ).start()
    return jsonify({
        "status": "processing",
        "message": "Downloading history in background.",
    })

@routes_bp.route("/spotify/download_link", methods=["POST"])
@jwt_required()
def spotify_download_link():
    from spotify_utils import spotify_download_link_internal
    user_id = get_jwt_identity()
    data = request.get_json()
    if not data or "link" not in data:
        return jsonify({"error": "No Spotify link provided"}), 400

    url = data["link"]
    threading.Thread(
        target=spotify_download_link_internal, args=(url, user_id), daemon=True
    ).start()
    return jsonify({"status": "processing", "message": "Download started in background."})

@routes_bp.route("/spotify/play", methods=["POST"])
@jwt_required()
def spotify_play():
    from spotify_utils import get_spotify_oauth
    user_id = get_jwt_identity()
    data = request.get_json()
    if data is None:
        return jsonify({"error": "No data provided"}), 400

    sp_oauth = get_spotify_oauth(user_id)
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
    if not token_info:
        return jsonify({"error": "Not authenticated with Spotify"}), 401

    sp = spotipy.Spotify(auth=token_info["access_token"])

    # Resolve the URI: check DB first, then search Spotify
    uri = data.get("spotify_uri")
    should_resume = False
    
    if not uri:
        title = data.get("title")
        artist = data.get("artist")
        if not title and not artist:
            # No track info provided -> signal to resume current context
            should_resume = True
        else:
            from spotify_utils import resolve_track_uri
            uri = resolve_track_uri(sp, title, artist)
            if not uri:
                return jsonify({"error": f"Could not find '{title}' by {artist or 'Unknown'} on Spotify."}), 404

    # Check Cache first
    try:
        global device_cache
        cached = device_cache.get(user_id)
        now = datetime.now()
        
        device_id = None
        if cached and (now - cached["timestamp"]).total_seconds() < DEVICE_CACHE_TTL:
            device_id = cached["id"]
            print(f"Using cached device: {device_id}")

        if not device_id:
            devices = sp.devices()
            if not devices["devices"]:
                return jsonify({"error": "No active Spotify devices found."}), 404
            
            # Prefer active device, fallback to first available
            device_id = next((d["id"] for d in devices["devices"] if d["is_active"]), devices["devices"][0]["id"])
            
            # Update Cache
            device_cache[user_id] = {"id": device_id, "timestamp": now}
            print(f"Cached new device: {device_id}")
        
        if should_resume:
            sp.start_playback(device_id=device_id)
            return jsonify({"status": "resumed", "device": device_id})
        else:
            sp.start_playback(device_id=device_id, uris=[uri])
            return jsonify({"status": "playing", "uri": uri})
    except Exception as e:
        error_msg = str(e)
        if "rate" in error_msg.lower() or "429" in error_msg:
            return jsonify({"error": "Spotify is temporarily rate limited. Please try again in a few minutes."}), 429
        if "Premium required" in error_msg:
            error_msg = "Spotify Premium is required for direct playback."
        return jsonify({"error": error_msg}), 500
