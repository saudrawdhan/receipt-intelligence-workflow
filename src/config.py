"""Configuration: loads the API key and model name from the environment."""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()


def require_api_key() -> str:
    """Return the API key or raise a clear error telling the user how to get one."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_key_here":
        raise RuntimeError(
            "No GEMINI_API_KEY found. Copy .env.example to .env and add a free key "
            "from https://aistudio.google.com/apikey (no credit card needed). "
            "Or run with --mock to test the pipeline without calling the model."
        )
    return GEMINI_API_KEY
