import os
import time
import sqlite3

import torch
import pynvml
import nest_asyncio
import uvicorn

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional

from .db import cur, conn
from .ingest import ingest_pdf
from .models import (
    evaluate_model,
    evaluate_both_models
)

app = FastAPI(title="PDF Ingestion & RAG Service")


# ================================
# 1.  Pydantic schema for queries
# ================================
class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None   # "deepseek" or "llama"; if omitted, /query/auto is used


# ============================
# 2.  Ingest endpoint (/ingest/)
# ============================
@app.post("/ingest/")
async def ingest_endpoint(
    file: UploadFile = File(...),
    upload_date: str   = Form(...),    # format: "YYYY-MM-DD"
    course_name: str   = Form(...),
    course_number: str = Form(...),
    topics: str        = Form(...)
):
    """
    Ingest a PDF:
      - Save metadata (filename, upload_date, course_name, course_number, topics) into SQLite.
      - Extract text from the PDF, chunk it, embed the chunks, and index them.
    Returns {"status":"success","ingested_chunks":<int>}.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    data = await file.read()
    try:
        result = ingest_pdf(
            data,
            file.filename,
            upload_date,
            course_name,
            course_number,
            topics
        )
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================
# 3.  Single-model query (/query/)
# =============================
@app.post("/query/")
def query_endpoint( model: str = Form(...),
    query: str = Form(...)):
    """
    Accepts {"query":"...","model":"deepseek"} or {"query":"...","model":"llama"}.
    Runs the specified model (with auto metadata filtering) and returns metrics + answer.
    """
    if model == "deepseek":
        result = evaluate_model(query, "deepseek")
    elif model == "llama":
        result = evaluate_model(query, "llama")
    else:
        raise HTTPException(status_code=400, detail="Model must be 'deepseek' or 'llama'")
    return {"status": "success", "result": result}


# =====================================
# 4.  Dual-model query (/query/auto)
# =====================================
@app.post("/query/auto")
def query_auto(query: str = Form(...)):
    """
    Accepts {"query":"..."} (ignore req.model).
    Runs both DeepSeek-7B and LLaMA-3.1-8B on the same query (with metadata filters),
    logs both rows to metrics.csv, picks the lower perplexity, and returns:
      {
        "status": "success",
        "chosen": "deepseek" or "llama",
        "deepseek": {...metrics+answer...},
        "llama":   {...metrics+answer...}
      }
    """
    try:
        result = evaluate_both_models(query)
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================
# 5.  Health endpoint
# ==================
@app.get("/health")
def health_check():
    return {"ok": True}


# ======================================================
# 6.  If running directly, start Uvicorn (no ngrok)
# ======================================================
if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8081")),
        log_level="info"
    )
