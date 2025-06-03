import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "files.db"

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")  # optional
