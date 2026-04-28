import random
import bcrypt
import re
import smtplib
import traceback
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import config, UI
from database import users_db, otps_db
from extensions import limiter

auth_bp = Blueprint("auth", __name__)

def send_email(to_email, otp):
    if not all([config.SMTP_USER, config.SMTP_PASSWORD]):
        print(UI.warning(f"[{datetime.now()}] SMTP credentials not set. Falling back to console log."))
        print(f"\n{'='*50}\nOTP for {to_email}: {otp}\n{'='*50}\n")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = str(config.SMTP_USER)
        msg["To"] = str(to_email)
        msg["Subject"] = "SONIC - Your Verification Code"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #121212; color: #ffffff; padding: 20px;">
            <div style="max-width: 600px; margin: auto; background-color: #1e1e1e; padding: 40px; border-radius: 10px; border: 1px solid #333;">
                <h2 style="color: #1DB954; text-align: center;">SONIC Verification</h2>
                <p style="font-size: 16px;">Hello,</p>
                <p style="font-size: 16px;">Your One-Time Password (OTP) for registration/reset is:</p>
                <div style="font-size: 32px; font-weight: bold; text-align: center; margin: 30px 0; color: #1DB954; letter-spacing: 5px;">
                    {otp}
                </div>
                <p style="font-size: 14px; color: #888;">This code will expire in 10 minutes. If you did not request this, please ignore this email.</p>
                <hr style="border: 0; border-top: 1px solid #333; margin: 30px 0;">
                <p style="font-size: 12px; color: #555; text-align: center;">SONIC AI Music Assistant</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(str(config.SMTP_HOST), int(config.SMTP_PORT)) as server:
            server.starttls()
            server.login(str(config.SMTP_USER), str(config.SMTP_PASSWORD))
            server.send_message(msg)
        
        print(UI.success(f"[{datetime.now()}] OTP sent to {to_email}"))
        return True
    except Exception as e:
        print(UI.error(f"[{datetime.now()}] Failed to send email to {to_email}: {e}"))
        traceback.print_exc()
        return False

def validate_input(email=None, username=None, password=None):
    if email:
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(email_regex, email):
            return False, "Invalid email format"
    if username:
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"
    if password:
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"
        if not any(not c.isalnum() for c in password):
            return False, "Password must contain at least one special character"
    return True, ""

# Rate Limit will be applied to this route
@auth_bp.route("/send-otp", methods=["POST"])
@limiter.limit("3 per hour")
def send_otp():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request data"}), 400
        
    email = data.get("email", "").lower()
    username = data.get("username", "")
    purpose = data.get("purpose", "register")  # 'register' or 'reset'
    
    # Check format first
    is_valid, err = validate_input(email=email, username=username if purpose == "register" else None)
    if not is_valid:
        return jsonify({"error": err}), 400

    if purpose == "reset":
        # For password reset, the email MUST already exist
        if not users_db.find_one({"email": email}):
            return jsonify({"error": "No account found with this email"}), 404
    else:
        # For registration, the email must NOT already exist
        if email and users_db.find_one({"email": email}):
            return jsonify({"error": "Email already registered"}), 400
        if username and users_db.find_one({"username": username}):
            return jsonify({"error": "Username already taken"}), 400

    otp = str(random.randint(100000, 999999))
    otps_db.update_one(
        {"email": email},
        {"$set": {"otp": otp, "created_at": datetime.now()}},
        upsert=True,
    )

    send_email(email, otp)
    return jsonify({"message": "OTP sent successfully"}), 200

@auth_bp.route("/register", methods=["POST"])
def register_user():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request data"}), 400
        
    email = data.get("email", "").lower()
    username = data.get("username", "")
    password = data.get("password", "")
    otp = str(data.get("otp", ""))

    if not all([email, username, password, otp]):
        return jsonify({"error": "All fields are required"}), 400

    is_valid, err = validate_input(email=email, username=username, password=password)
    if not is_valid:
        return jsonify({"error": err}), 400

    if users_db.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 400

    otp_record = otps_db.find_one({"email": email, "otp": otp})
    if not otp_record:
        return jsonify({"error": "Invalid or expired OTP"}), 400

    if datetime.now() - otp_record["created_at"] > timedelta(minutes=10):
        return jsonify({"error": "OTP has expired"}), 400

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    result = users_db.insert_one(
        {
            "email": email,
            "username": username,
            "password": hashed_pw,
            "created_at": datetime.now(),
        }
    )
    otps_db.delete_one({"email": email})

    # Create JWT token
    access_token = create_access_token(identity=str(result.inserted_id))

    return jsonify({
        "message": "User registered successfully",
        "access_token": access_token,
        "user": {
            "id": str(result.inserted_id),
            "username": username,
            "email": email
        }
    }), 201

@auth_bp.route("/login", methods=["POST"])
def login_user():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request data"}), 400
        
    email = data.get("email", "").lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = users_db.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        return jsonify({"error": "Invalid password"}), 401

    # Create JWT token
    access_token = create_access_token(identity=str(user["_id"]))

    return jsonify({
        "message": "Login successful",
        "access_token": access_token,
        "user": {
            "id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"]
        }
    }), 200

@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json()
    email = data.get("email", "").lower()
    otp = str(data.get("otp", ""))
    new_password = data.get("new_password", "")

    if not all([email, otp, new_password]):
        return jsonify({"error": "Email, OTP, and new password are required"}), 400

    is_valid, err = validate_input(password=new_password)
    if not is_valid:
        return jsonify({"error": err}), 400

    user = users_db.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404

    otp_record = otps_db.find_one({"email": email, "otp": otp})
    if not otp_record:
        return jsonify({"error": "Invalid or expired OTP"}), 400

    if datetime.now() - otp_record["created_at"] > timedelta(minutes=10):
        return jsonify({"error": "OTP has expired"}), 400

    hashed_pw = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    users_db.update_one({"email": email}, {"$set": {"password": hashed_pw}})
    otps_db.delete_one({"email": email})

    return jsonify({"message": "Password reset successfully"}), 200
