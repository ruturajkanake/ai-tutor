from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional

from .ingest import ingest_pdf
from .models  import (
    run_deepseek, run_llama,
    select_best_model, evaluate_model
)

app = FastAPI(title="PDF Ingestion & RAG Service")

# ---------- Pydantic schema ---------- #
class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None  # deepseek | llama (None → auto)

# ---------- endpoints ---------- #
@app.post("/ingest/")
async def ingest_endpoint(
    file: UploadFile = File(...),
    upload_date: str   = Form(...),
    course_name: str   = Form(...),
    course_number: str = Form(...),
    topics: str        = Form(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    data = await file.read()
    result = ingest_pdf(
        data, file.filename, upload_date,
        course_name, course_number, topics
    )
    return {"status": "success", **result}

@app.post("/query/")
def query_endpoint(req: QueryRequest):
    if req.model == "deepseek":
        res = run_deepseek(req.query)
    elif req.model == "llama":
        res = run_llama(req.query)
    else:
        raise HTTPException(400, "model must be 'deepseek' or 'llama'")
    return {"status": "success", "answer": res}

@app.post("/query/auto")
def query_auto(req: QueryRequest):
    chosen = select_best_model(req.query)
    result = evaluate_model(req.query, chosen)
    return {"status": "success", "selected_model": chosen, "result": result}

@app.get("/health")
def health(): return {"ok": True}
