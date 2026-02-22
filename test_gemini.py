import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# 1. Grab the Gemini key safely
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# 2. HIDE the Cloud key from the environment
if "GOOGLE_API_KEY" in os.environ:
    del os.environ["GOOGLE_API_KEY"]

# 3. Initialize Gemini
client = genai.Client(api_key=GEMINI_KEY)

print("Sending request to Gemini...")

try:
    response = client.models.generate_content(
        model="gemini-3.1-flash",  # Updated to the current active model
        contents="You are a test assistant. Respond with exactly: 'Connection successful!'"
    )
    print(f"\n[Success] {response.text}")
except Exception as e:
    print(f"\n[Error] {e}")