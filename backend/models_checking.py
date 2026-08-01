from google import genai
from dotenv import load_dotenv
import os

# 1. Load environment variables from .env file
load_dotenv()

# 2. Strict API Key Validation
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

for model in client.models.list():
    print(model.name)