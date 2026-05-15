from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
print("Available flash models:")
for m in client.models.list():
    if 'flash' in m.name:
        print(m.name)
