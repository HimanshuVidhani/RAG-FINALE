from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
print("Available embedding models:")
try:
    for m in client.models.list():
        if 'embed' in m.name.lower():
            print(m.name)
except Exception as e:
    print("Error:", e)
