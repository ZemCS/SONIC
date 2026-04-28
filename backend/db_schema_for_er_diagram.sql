-- ============================================================================
-- SONIC Application Database Schema (for ER Diagram Generation)
-- ============================================================================
-- Database: sonic_db (MongoDB - represented here as relational tables)
-- This schema represents 8 collections with their fields and relationships.
-- ============================================================================

-- ============================================================================
-- 1. USERS - Core user accounts
-- ============================================================================
CREATE TABLE users (
    _id                 VARCHAR(24)     PRIMARY KEY,        -- MongoDB ObjectId
    email               VARCHAR(255)    NOT NULL UNIQUE,    -- User email (indexed, unique)
    username            VARCHAR(255)    NOT NULL UNIQUE,    -- Display name (indexed, unique)
    password            VARCHAR(255)    NOT NULL,           -- Bcrypt hashed password
    created_at          TIMESTAMP       NOT NULL,           -- Account creation timestamp
    spotify_token       JSON            NULL                -- Cached Spotify OAuth token (access_token, refresh_token, expires_at, etc.)
);

-- ============================================================================
-- 2. OTPS - One-Time Passwords for registration & password reset
-- ============================================================================
CREATE TABLE otps (
    _id                 VARCHAR(24)     PRIMARY KEY,        -- MongoDB ObjectId
    email               VARCHAR(255)    NOT NULL,           -- Associated email (indexed)
    otp                 VARCHAR(6)      NOT NULL,           -- 6-digit OTP code
    created_at          TIMESTAMP       NOT NULL,           -- Creation time (TTL index: expires after 900 seconds)

    FOREIGN KEY (email) REFERENCES users(email)
);

-- ============================================================================
-- 3. AUDIO_INFERENCES - Audio-based mood predictions per song
-- ============================================================================
CREATE TABLE audio_inferences (
    _id                 VARCHAR(24)     PRIMARY KEY,        -- MongoDB ObjectId
    filename            VARCHAR(255)    NOT NULL,           -- Song filename (indexed)
    user_id             VARCHAR(24)     NOT NULL,           -- User who triggered analysis or "system" (indexed)
    mood                VARCHAR(50)     NULL,               -- Predicted mood from audio model (e.g., sad, joyful, angry, relaxed, energetic)
    confidence          FLOAT           NULL,               -- Prediction confidence (0.0 - 1.0)
    valence             FLOAT           NULL,               -- Valence score from Essentia VA model
    arousal             FLOAT           NULL,               -- Arousal score from Essentia VA model
    timestamp           TIMESTAMP       NOT NULL,           -- When inference was created

    FOREIGN KEY (user_id) REFERENCES users(_id)
);

-- ============================================================================
-- 4. LYRICS_INFERENCES - Lyrics-based mood predictions per song
-- ============================================================================
CREATE TABLE lyrics_inferences (
    _id                 VARCHAR(24)     PRIMARY KEY,        -- MongoDB ObjectId
    filename            VARCHAR(255)    NOT NULL,           -- Song filename (indexed)
    user_id             VARCHAR(24)     NOT NULL,           -- User who triggered analysis or "system" (indexed)
    lyrics              TEXT            NULL,               -- Processed lyrics text (translated if applicable)
    original_lyrics     TEXT            NULL,               -- Original lyrics before translation
    translated_lyrics   TEXT            NULL,               -- English translation (NULL if already English)
    language            VARCHAR(10)     NULL,               -- Detected language code (e.g., "en", "es", "none")
    mood                VARCHAR(50)     NULL,               -- Predicted mood from text model
    confidence          FLOAT           NULL,               -- Prediction confidence (0.0 - 1.0)
    source              VARCHAR(50)     NULL,               -- Lyrics source: "Genius API", "Whisper AI", or "None"
    timestamp           TIMESTAMP       NOT NULL,           -- When inference was created

    FOREIGN KEY (user_id) REFERENCES users(_id)
);

-- ============================================================================
-- 5. FUSED_INFERENCES - Final combined (audio + lyrics) mood predictions
-- ============================================================================
CREATE TABLE fused_inferences (
    _id                 VARCHAR(24)     PRIMARY KEY,        -- MongoDB ObjectId
    filename            VARCHAR(255)    NOT NULL,           -- Song filename (indexed)
    user_id             VARCHAR(24)     NOT NULL,           -- User who triggered analysis or "system" (indexed)
    predicted_mood      VARCHAR(50)     NOT NULL,           -- Final fused mood prediction
    confidence          FLOAT           NOT NULL,           -- Fused prediction confidence (0.0 - 1.0)
    spotify_uri         VARCHAR(255)    NULL,               -- Resolved Spotify URI (e.g., "spotify:track:xxxx")
    timestamp           TIMESTAMP       NOT NULL,           -- When inference was created

    FOREIGN KEY (user_id) REFERENCES users(_id)
);

-- ============================================================================
-- 6. IMAGE_INFERENCES - Image-based mood predictions (selfie/photo mood)
-- ============================================================================
CREATE TABLE image_inferences (
    _id                 VARCHAR(24)     PRIMARY KEY,        -- MongoDB ObjectId
    filename            VARCHAR(255)    NOT NULL,           -- Uploaded image filename (indexed)
    user_id             VARCHAR(24)     NOT NULL,           -- User who uploaded the image (indexed)
    predicted_mood      VARCHAR(50)     NOT NULL,           -- Predicted mood from image model
    confidence          FLOAT           NOT NULL,           -- Prediction confidence (0.0 - 1.0)
    timestamp           TIMESTAMP       NOT NULL,           -- When prediction was made

    FOREIGN KEY (user_id) REFERENCES users(_id)
);

-- ============================================================================
-- 7. FEEDBACK - User corrections on mood predictions
-- ============================================================================
CREATE TABLE feedback (
    _id                     VARCHAR(24)     PRIMARY KEY,    -- MongoDB ObjectId
    user_id                 VARCHAR(24)     NOT NULL,       -- User who submitted the correction
    timestamp               TIMESTAMP       NOT NULL,       -- When feedback was submitted
    filename                VARCHAR(255)    NULL,           -- Original filename
    file_path               VARCHAR(500)    NULL,           -- Path to the file
    original_prediction     VARCHAR(50)     NULL,           -- The model's original mood prediction
    corrected_mood          VARCHAR(50)     NULL,           -- The user's corrected mood label

    FOREIGN KEY (user_id) REFERENCES users(_id)
);

-- ============================================================================
-- 8. RECOMMENDATION_LOGS - Tracks song recommendations served to users
-- ============================================================================
CREATE TABLE recommendation_logs (
    _id                 VARCHAR(24)     PRIMARY KEY,        -- MongoDB ObjectId
    user_id             VARCHAR(24)     NOT NULL,           -- User who received recommendations (indexed)
    timestamp           TIMESTAMP       NOT NULL,           -- When recommendation was served (indexed)
    requested_mood      VARCHAR(50)     NOT NULL,           -- Mood the user requested recommendations for
    recommended_songs   JSON            NOT NULL,           -- Array of recommended song objects

    FOREIGN KEY (user_id) REFERENCES users(_id)
);

-- ============================================================================
-- INDEXES (mirroring MongoDB indexes defined in database.py)
-- ============================================================================
CREATE UNIQUE INDEX idx_users_email              ON users(email);
CREATE UNIQUE INDEX idx_users_username           ON users(username);
CREATE INDEX        idx_otps_email               ON otps(email);
CREATE INDEX        idx_audio_inferences_filename ON audio_inferences(filename);
CREATE INDEX        idx_audio_inferences_user_id  ON audio_inferences(user_id);
CREATE INDEX        idx_lyrics_inferences_filename ON lyrics_inferences(filename);
CREATE INDEX        idx_lyrics_inferences_user_id  ON lyrics_inferences(user_id);
CREATE INDEX        idx_fused_inferences_filename  ON fused_inferences(filename);
CREATE INDEX        idx_fused_inferences_user_id   ON fused_inferences(user_id);
CREATE INDEX        idx_image_inferences_filename  ON image_inferences(filename);
CREATE INDEX        idx_image_inferences_user_id   ON image_inferences(user_id);
CREATE INDEX        idx_recommendation_logs_user_id   ON recommendation_logs(user_id);
CREATE INDEX        idx_recommendation_logs_timestamp ON recommendation_logs(timestamp);
