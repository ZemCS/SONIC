import threading
import spotipy
import subprocess
import os
import time
import re
from bson import ObjectId
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException
from config import config, UI
from database import db, fused_inferences
from state import SONG_DATABASE

# ---------------------------------------------------------------------------
# Rate-limiting config
# ---------------------------------------------------------------------------
SPOTIFY_CALL_DELAY = 1.0        # seconds between consecutive API calls
SPOTIFY_MAX_RETRIES = 3         # max retries on 429 (rate-limited)
SPOTIFY_BACKOFF_BASE = 2        # exponential backoff multiplier
SPOTIFY_BACKOFF_CAP = 60        # max wait between retries (seconds)


def _spotify_api_call(func, *args, **kwargs):
    """
    Wrapper that calls a Spotify API function with automatic retry on 429.
    Respects the Retry-After header when present, otherwise uses exponential
    backoff capped at SPOTIFY_BACKOFF_CAP seconds.
    """
    for attempt in range(SPOTIFY_MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers.get("Retry-After", 0)) if e.headers else 0
                wait = max(retry_after, SPOTIFY_BACKOFF_BASE ** (attempt + 1))
                wait = min(wait, SPOTIFY_BACKOFF_CAP)
                print(UI.warning(
                    f"Spotify rate limited (429). "
                    f"Retrying in {wait}s (attempt {attempt + 1}/{SPOTIFY_MAX_RETRIES})..."
                ))
                time.sleep(wait)
            else:
                raise
    # Final attempt — let any exception propagate
    return func(*args, **kwargs)


class MongoCacheHandler(spotipy.cache_handler.CacheHandler):
    def __init__(self, user_id):
        self.user_id = (
            ObjectId(user_id)
            if isinstance(user_id, str) and len(user_id) == 24
            else user_id
        )

    def get_cached_token(self):
        user = db["users"].find_one({"_id": self.user_id})
        return user.get("spotify_token") if user and "spotify_token" in user else None

    def save_token_to_cache(self, token_info):
        db["users"].update_one(
            {"_id": self.user_id}, {"$set": {"spotify_token": token_info}}, upsert=True
        )

def get_spotify_oauth(user_id=None):
    return SpotifyOAuth(
        client_id=config.SPOTIFY_CLIENT_ID,
        client_secret=config.SPOTIFY_CLIENT_SECRET,
        redirect_uri=config.SPOTIFY_REDIRECT_URI,
        scope=config.SPOTIFY_SCOPE,
        cache_handler=MongoCacheHandler(user_id) if user_id else None,
    )

global_sp_oauth = get_spotify_oauth()

def resolve_track_uri(sp, title, artist):
    """
    Search for a track URI on Spotify, but check MongoDB first.
    Returns the URI if found, else None.
    """
    if not title:
        return None

    # 1. Check if we already have this exact title/artist in fused_inferences
    clean_title = title.replace('.mp3', '')
    clean_title_regex = re.compile(f"^{re.escape(clean_title)}$", re.IGNORECASE)
    clean_artist = "" if not artist or artist.lower() == "unknown artist" else artist.strip()
    
    query = {"title": clean_title_regex}
    if clean_artist:
        query["artist"] = re.compile(f"^{re.escape(clean_artist)}$", re.IGNORECASE)

    existing = fused_inferences.find_one(query)
    if existing and existing.get("spotify_uri"):
        print(UI.info(f"  [Cache Hit] URI for {title} found in DB."))
        return existing["spotify_uri"]

    # 2. Not in DB, search Spotify
    try:
        search_query = f"track:{clean_title}"
        if clean_artist and clean_artist.lower() != "unknown artist":
            search_query += f" artist:{clean_artist}"
            
        results = _spotify_api_call(sp.search, q=search_query, type="track", limit=1)
        
        # Loose search if strict fails
        if not results["tracks"]["items"] and clean_artist:
            results = _spotify_api_call(sp.search, q=f"{clean_title} {clean_artist}", type="track", limit=1)

        if results["tracks"]["items"]:
            uri = results["tracks"]["items"][0]["uri"]
            # We don't save to DB here; that's handled by the caller if needed
            return uri
    except Exception as e:
        print(UI.error(f"Spotify search failed for {title}: {e}"))
    
    return None

def resolve_spotify_uris(user_id=None):
    if user_id:
        sp_oauth = get_spotify_oauth(user_id)
        token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
        if not token_info:
            return
        sp = spotipy.Spotify(auth=token_info["access_token"])
    else:
        from spotipy.oauth2 import SpotifyClientCredentials
        sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=config.SPOTIFY_CLIENT_ID,
                client_secret=config.SPOTIFY_CLIENT_SECRET,
            )
        )

    unresolved = [s for s in SONG_DATABASE if not s.get("spotify_uri")]
    total = len(unresolved)
    if total > 0:
        print(UI.info(f"Resolving Spotify URIs for {total}/{len(SONG_DATABASE)} songs..."))

    for i, song in enumerate(unresolved):
        uri = resolve_track_uri(sp, song['title'], song['artist'])
        if uri:
            song["spotify_uri"] = uri
            # Save to MongoDB
            fused_inferences.update_one(
                {"filename": song["filename"]},
                {"$set": {"spotify_uri": uri}}
            )

        # Progress log every 5 songs
        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(UI.info(f"  URI progress: {i + 1}/{total}"))

        # Delay between API calls to stay within rate limits (only if we actually hit the API)
        # Note: resolve_track_uri might return from cache, so this delay could be optimized
        # but 1s is safe regardless.
        if i < total - 1:
            time.sleep(SPOTIFY_CALL_DELAY)

    print(UI.success(f"Spotify URI resolution complete."))

def spotify_download_internal(user_id):
    import sys
    import main
    ytdlp_path = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    sp_oauth = get_spotify_oauth(user_id)
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
    if not token_info:
        return
    sp = spotipy.Spotify(auth=token_info["access_token"])
    try:
        results = _spotify_api_call(sp.current_user_recently_played, limit=10)
    except Exception as e:
        print(f"Background Download Info Error: {e}")
        return

    for item in results["items"]:
        track = item["track"]
        artist_name = track["artists"][0]["name"]
        track_name = track["name"]

        # Strategy: Check if we already have this song in SONG_DATABASE (on disk)
        # OR in fused_inferences (already processed in the past)
        already_downloaded = any(
            s["title"].lower() == track_name.lower()
            and s["artist"].lower() == artist_name.lower()
            for s in SONG_DATABASE
        )
        if already_downloaded:
            continue

        # NEW: Also check fused_inferences directly for title/artist
        # This handles cases where the file was deleted but the inference is still in DB
        # We can reconstruct SONG_DATABASE entry from the DB if it exists
        existing_inference = fused_inferences.find_one({
            "title": {"$regex": f"^{re.escape(track_name)}$", "$options": "i"},
            "artist": {"$regex": f"^{re.escape(artist_name)}$", "$options": "i"}
        })
        if existing_inference:
            # We already know the mood, no need to download again
            # We just need to make sure it's in our in-memory SONG_DATABASE for current sessions
            SONG_DATABASE.append({
                "title": track_name,
                "artist": artist_name,
                "mood": existing_inference["predicted_mood"].lower(),
                "filename": existing_inference["filename"],
                "confidence": existing_inference["confidence"],
                "spotify_uri": track["uri"],
            })
            continue

        search_query = f"ytsearch1:{track_name} {artist_name} audio"
        safe_name = f"{artist_name} - {track_name}".replace("/", "_").replace("\\", "_")
        output_template = os.path.join(config.SONGS_DIR, f"{safe_name}.%(ext)s")

        try:
            subprocess.run(
                [
                    ytdlp_path, "-x", "--audio-format", "mp3",
                    "--postprocessor-args", f"-ar {config.AUDIO_SAMPLE_RATE}",
                    "-o", output_template, search_query,
                ],
                check=True,
            )
            time.sleep(1.5)
            main.build_song_database()
        except Exception as e:
            print(f"Download Error: {e}")

        # Delay between iterations to avoid hammering APIs
        time.sleep(SPOTIFY_CALL_DELAY)

def spotify_download_link_internal(url, user_id):
    import sys
    import main
    ytdlp_path = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    try:
        search_query = url
        output_template = os.path.join(config.SONGS_DIR, "%(title)s.%(ext)s")

        if "spotify.com" in url:
            sp_oauth = get_spotify_oauth(user_id)
            token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
            if token_info:
                sp = spotipy.Spotify(auth=token_info["access_token"])
                track = _spotify_api_call(sp.track, url)
                artist_name = track["artists"][0]["name"]
                track_name = track["name"]

                already_downloaded = any(
                    s["title"].lower() == track_name.lower()
                    and s["artist"].lower() == artist_name.lower()
                    for s in SONG_DATABASE
                )
                if already_downloaded:
                    return

                search_query = f"ytsearch1:{track_name} {artist_name} audio"
                safe_name = f"{artist_name} - {track_name}".replace("/", "_").replace("\\", "_")
                output_template = os.path.join(config.SONGS_DIR, f"{safe_name}.%(ext)s")

        subprocess.run(
            [
                ytdlp_path, "-x", "--audio-format", "mp3",
                "--postprocessor-args", f"-ar {config.AUDIO_SAMPLE_RATE}",
                "-o", output_template, search_query,
            ],
            check=True,
        )
        main.build_song_database()
    except Exception as e:
        print(f"Background YT-DLP Error: {e}")
