# Configuration settings for the application
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Google Gemini Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "models/gemini-1.5-flash-latest"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please add it to your .env file.")

# --- Ollama Configuration (for local LLM) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

# --- RAG Configuration ---
CHUNK_SIZE = 500  # Number of characters per chunk
CHUNK_OVERLAP = 100  # Number of characters to overlap between chunks

# --- CORS Configuration ---
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add any other origins where your frontend might be running
]

# --- File Upload Configuration ---
UPLOAD_DIR_NAME = "uploads"
