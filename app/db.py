import sqlite3, datetime as dt
from .settings import DB_PATH

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur  = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY,
  filename      TEXT,
  upload_date   TEXT,
  course_name   TEXT,
  course_number TEXT,
  topics        TEXT
)
""")
conn.commit()

def now_iso() -> str:
    return dt.datetime.utcnow().isoformat(" ", "seconds")
