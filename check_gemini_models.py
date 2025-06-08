# check_gemini_models.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set in .env")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Available Gemini Models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name} (Supports generateContent)")
        else:
            print(f"  - {m.name} (Does NOT support generateContent for your account)")