import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pyngrok import ngrok, conf

BASE_DIR = Path(__file__).resolve().parent


def main():
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        raise RuntimeError("NGROK_AUTH_TOKEN is missing from .env")

    conf.get_default().auth_token = auth_token
    
    # Kill any existing ngrok tunnels to ensure a fresh connection
    ngrok.kill()
    
    url = ngrok.connect(5000).public_url

    out_path = BASE_DIR / "ngrok_url.txt"
    out_path.write_text(url)
    print(f"Ngrok running at: {url}")

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        out_path = BASE_DIR / "ngrok_url.txt"
        out_path.write_text(f"ERROR: {e}")
        print(f"Ngrok startup failed: {e}")
        raise
