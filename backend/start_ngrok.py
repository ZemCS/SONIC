import time
from pyngrok import ngrok, conf

def main():
    try:
        # Authenticate with the user's provided token
        conf.get_default().auth_token = "3BFvk32AiCeP0bapTaMSNv9mXI6_6EsbNSZPMP9fKectbiRwD"
        url = ngrok.connect(5000).public_url
        
        with open("/home/abdullah/Documents/SONIC/backend/ngrok_url.txt", "w") as f:
            f.write(url)
            
        print(f"Ngrok running at: {url}")
        
        # Keep process alive
        while True:
            time.sleep(3600)
    except Exception as e:
        with open("/home/abdullah/Documents/SONIC/backend/ngrok_url.txt", "w") as f:
            f.write(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()
