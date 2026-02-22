import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Hide Cloud Key
if "GOOGLE_API_KEY" in os.environ:
    del os.environ["GOOGLE_API_KEY"]

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY)

print("Fetching available models for your API key...\n")

try:
    for model in client.models.list():
        print(model.name)
except Exception as e:
    print(f"[Error] {e}")