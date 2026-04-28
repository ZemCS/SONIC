from pymongo import MongoClient, ASCENDING, IndexModel
import os
from datetime import datetime
from config import UI

# MongoDB Setup

# MongoDB Setup
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["sonic_db"]

# Collections
audio_inferences = db["audio_inferences"]
lyrics_inferences = db["lyrics_inferences"]
fused_inferences = db["fused_inferences"]
image_inferences = db["image_inferences"]
feedback_db = db["feedback"]
users_db = db["users"]
otps_db = db["otps"]
recommendation_logs = db["recommendation_logs"]

def setup_indexes():
    """Configure MongoDB indexes for performance and security."""
    print(UI.info("Configuring Database Indexes..."))
    try:
        # User indexing
        users_db.create_index("email", unique=True)
        users_db.create_index("username", unique=True)

        # OTP indexing with TTL (Expire after 15 minutes)
        otps_db.create_index("email")
        otps_db.create_index("created_at", expireAfterSeconds=900)

        # Inference indexing for fast lookup by filename and user
        for collection in [audio_inferences, lyrics_inferences, fused_inferences, image_inferences]:
            collection.create_index("filename")
            collection.create_index("user_id")

        # Recommendation log indexing
        recommendation_logs.create_index("user_id")
        recommendation_logs.create_index("timestamp")

        print(UI.success("Database indexes configured."))
    except Exception as e:
        print(UI.error(f"Failed to setup indexes: {e}"))

# Call setup on first import
setup_indexes()
