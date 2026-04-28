# SONIC

This project is a comprehensive mood detection application that analyzes user inputs (audio, images, and lyrics) to determine emotional states. It integrates with Spotify for music recommendations based on detected moods. The application consists of a Python backend for multimodal AI model inference and a Flutter frontend for the user interface.

## Features

- **Audio Mood Detection**: Uses machine learning models to analyze audio files and detect moods.
- **Image Mood Detection**: Analyzes images to infer emotional content.
- **Lyrics Mood Detection**: Processes song lyrics to determine mood.
- **Spotify Integration**: Connects to Spotify for personalized music recommendations.
- **User Authentication**: Secure login and registration system.
- **Cross-Platform UI**: Flutter-based frontend that works on mobile and web.

## Directory Structure

```
.
├── backend/                          # Python backend application
│   ├── main.py                       # Main Flask application entry point
│   ├── requirements.txt              # Python dependencies
│   ├── config.py                     # Configuration settings
│   ├── database.py                   # Database connection and operations
│   ├── auth.py                       # Authentication logic
│   ├── routes.py                     # API routes
│   ├── spotify_utils.py              # Spotify API utilities
│   ├── pipeline.py                   # Mood detection pipeline
│   ├── state.py                      # Application state management
│   ├── extensions.py                 # Flask extensions
│   ├── start_ngrok.py                # Ngrok tunneling setup
│   ├── db_schema_for_er_diagram.sql  # Database schema
│   ├── models/                       # AI models and training data (git ignored)
│   │   ├── deam-msd-musicnn-2.pb     # Audio model 1
│   │   ├── msd-musicnn-1.pb          # Audio model 2
│   │   ├── audio_mood_model/         # Audio mood detection model
│   │   ├── image_mood_model/         # Image mood detection model
│   │   └── lyrics_mood_model/        # Lyrics mood detection model
│   ├── music_library/                # Music library storage (git ignored)
│   └── testing/                      # Test files (git ignored)
│       └── test_auth.py
└── frontend/                         # Flutter frontend application
    ├── pubspec.yaml                  # Flutter dependencies
    ├── lib/                          # Dart source code
    │   ├── main.dart                 # Main Flutter app entry point
    │   ├── home_page.dart            # Home page
    │   ├── login_page.dart           # Login page
    │   ├── register_page.dart        # Registration page
    │   ├── forgot_password_page.dart # Password recovery
    │   ├── account_page.dart         # User account page
    │   ├── camera_page.dart          # Camera interface
    │   ├── gallery_page.dart         # Gallery interface
    │   ├── predict_page.dart         # Mood prediction page
    │   ├── results_page.dart         # Results display
    │   ├── recommendations_page.dart # Music recommendations
    │   ├── spotify_connect_page.dart # Spotify connection
    │   └── welcome_page.dart         # Welcome/onboarding page
    ├── android/                      # Android-specific files
    ├── ios/                          # iOS-specific files
    ├── web/                          # Web-specific files
    ├── linux/                        # Linux-specific files
    ├── macos/                        # macOS-specific files
    ├── windows/                      # Windows-specific files
    └── analysis_options.yaml         # Dart analysis options
```

## Prerequisites

### Backend (Linux/WSL)
- Python 3.12
- pip (Python package installer)
- Virtual environment tools (venv)
- Ngrok account (for tunneling, if needed)

### Frontend
- Flutter SDK (version 3.0 or higher)
- Dart SDK (comes with Flutter)
- Android Studio or Xcode (for mobile development, optional)

## Setup Instructions

### Backend Setup (Linux/WSL)

1. **Navigate to the backend directory:**
   ```bash
   cd backend/
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**
   Create a `.env` file in the backend directory with necessary configurations (e.g., database URL, Spotify API keys, etc.). Example:
   ```
   DATABASE_URL=your_database_url
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
   SECRET_KEY=your_secret_key
   ```

6. **Set up the database:**
   Run the database schema:
   ```bash
   # Assuming you have PostgreSQL or similar set up
   psql -d your_database < db_schema_for_er_diagram.sql
   ```

7. **For ngrok tunneling (if needed):**
   - Install ngrok if not already installed
   - Run `python start_ngrok.py` to start tunneling

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend/
   ```

2. **Install Flutter dependencies:**
   ```bash
   flutter pub get
   ```

3. **Set up for your platform:**
   - **Android:** Ensure Android SDK is installed and configured
   - **iOS:** Ensure Xcode is installed (macOS only)
   - **Web:** No additional setup needed
   - **Desktop:** Follow Flutter's desktop setup guide

## Running the Application

### Backend (Linux/WSL)

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run the Flask application:**
   ```bash
   python main.py
   ```

   The backend should start on `http://localhost:5000` (or configured port).

3. **For development with ngrok:**
   ```bash
   python start_ngrok.py
   ```
   This will provide a public URL for the backend.

### Frontend

1. **Run the Flutter app:**
   ```bash
   flutter run
   ```

   Choose your target platform (Android, iOS, Web, etc.).

2. **For web development:**
   ```bash
   flutter run -d web
   ```

3. **For specific devices:**
   ```bash
   flutter devices  # List available devices
   flutter run -d <device_id>
   ```

## API Endpoints

The backend provides the following main endpoints (assuming default Flask setup):

- `POST /login` - User authentication
- `POST /register` - User registration
- `POST /predict/audio` - Audio mood prediction
- `POST /predict/image` - Image mood prediction
- `POST /predict/lyrics` - Lyrics mood prediction
- `GET /recommendations` - Get Spotify recommendations

Refer to `routes.py` for complete API documentation.

## Testing

### Backend Tests
```bash
cd backend/
source venv/bin/activate
python -m pytest testing/
```

### Frontend Tests
```bash
cd frontend/
flutter test
```

## Deployment

### Backend
- Use Gunicorn or similar WSGI server for production
- Set up proper environment variables
- Configure a production database

### Frontend
- Build for production:
  ```bash
  flutter build web  # For web
  flutter build apk  # For Android
  flutter build ios  # For iOS
  ```

## Troubleshooting

- **Backend issues:** Check Python version, virtual environment activation, and dependency installation
- **Frontend issues:** Ensure Flutter SDK is properly installed and PATH is set
- **Database issues:** Verify database connection and schema setup
- **Spotify integration:** Ensure API keys are correctly set in environment variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
