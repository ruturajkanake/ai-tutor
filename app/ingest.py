import fitz                       # PyMuPDF
from .db import cur, conn
from .models import vector_store

# ------------ helpers ------------ #
def chunk_text(text: str, size: int = 500, overlap: int = 50):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return chunks

# ------------ main ingest ------------ #
def ingest_pdf(
    data: bytes,
    filename: str,
    upload_date: str,
    course_name: str,
    course_number: str,
    topics: str
):
    # 1) save metadata
    cur.execute(
        """INSERT INTO files(filename, upload_date, course_name,
                             course_number, topics)
           VALUES (?,?,?,?,?)""",
        (filename, upload_date, course_name, course_number, topics)
    )
    conn.commit()

    # 2) extract text
    doc = fitz.open(stream=data, filetype="pdf")
    fulltext = "".join(page.get_text() for page in doc)

    # 3) chunk & embed
    chunks     = chunk_text(fulltext)
    metadatas  = [{"filename": filename, "chunk_idx": i}
                  for i in range(len(chunks))]
    vector_store.add_texts(texts=chunks, metadatas=metadatas)

    return {"ingested_chunks": len(chunks)}
