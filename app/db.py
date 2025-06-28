import sqlite3
import datetime as dt
from pathlib import Path

# ------------------------------------------------------------------
# Database connection (unchanged)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "files.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur  = conn.cursor()

cur.execute(
    """
    CREATE TABLE IF NOT EXISTS files(
      id INTEGER PRIMARY KEY,
      filename      TEXT,
      upload_date   TEXT,      -- YYYY-MM-DD
      course_name   TEXT,
      course_number TEXT,
      topics        TEXT
    )
"""
)
conn.commit()


def now_iso() -> str:
    return dt.datetime.utcnow().isoformat(" ", "seconds")


# ------------------------------------------------------------------
# NEW: helper to look up filenames that match metadata filters
# ------------------------------------------------------------------
def filenames_by_meta(
    course_name: str | None = None,
    course_number: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[str]:
    """
    Return list of filenames that satisfy all given metadata filters.
    Each arg may be None (ignored).  Dates are inclusive and expected
    as 'YYYY-MM-DD'.
    """
    clauses, params = [], []

    if course_name:
        clauses.append("course_name LIKE ?")
        params.append(f"%{course_name}%")

    if course_number:
        clauses.append("course_number = ?")
        params.append(course_number)

    if date_from:
        clauses.append("upload_date >= ?")
        params.append(date_from)

    if date_to:
        clauses.append("upload_date <= ?")
        params.append(date_to)

    sql = "SELECT filename FROM files"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    return [row[0] for row in cur.execute(sql, params).fetchall()]
