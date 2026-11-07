import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://lessons:lessons@localhost:5432/lessons")

# Speech-to-text. whisper-large-v3 is ~2.8x the price but a bit more accurate.
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
# Default language for Whisper when a lesson doesn't set one ('it', 'en', ...). Empty = auto-detect.
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "")

# Agent 1 step 2 (cheap, fast, runs every window) and Agent 2 (runs once per lesson).
LISTENER_MODEL = os.getenv("LISTENER_MODEL", "openai/gpt-oss-20b")
ANALYST_MODEL = os.getenv("ANALYST_MODEL", "openai/gpt-oss-120b")

# The Listener runs once at least this many seconds of un-annotated transcript have piled up.
# With 30 s chunks and a 30 s window it runs on every chunk; with 15 s chunks it runs every 2nd.
LISTENER_WINDOW_S = int(os.getenv("LISTENER_WINDOW_S", "30"))

# A student "didn't understand" a topic if their understanding rating is <= this.
LOW_UNDERSTANDING = int(os.getenv("LOW_UNDERSTANDING", "2"))
