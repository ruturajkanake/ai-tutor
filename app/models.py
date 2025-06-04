import csv
import time
import re
import datetime as dt
from pathlib import Path
from typing import Tuple, Dict

import torch
import pynvml
from dateutil import parser as dateparse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.sklearn import SKLearnVectorStore

from .db import filenames_by_meta, cur, BASE_DIR

# ==========================================================================
# 0.  Utility: safe vector-store length
# ==========================================================================
def _vs_count(vs: SKLearnVectorStore) -> int:
    """
    Return the number of indexed chunks in the SKLearnVectorStore `vs`.
    Covers both older versions (with _embeddings) and newer (with index_to_docstore_id).
    """
    if hasattr(vs, "index_to_docstore_id"):
        return len(vs.index_to_docstore_id)
    if hasattr(vs, "_embeddings"):
        return len(vs._embeddings)
    return 0


# ==========================================================================
# 1.  Regex + lookup for metadata filtering
# ==========================================================================
COURSE_NUMBERS = [row[0] for row in cur.execute("SELECT DISTINCT course_number FROM files")]
COURSE_NAMES   = [row[0] for row in cur.execute("SELECT DISTINCT course_name   FROM files")]

COURSE_RE = re.compile(r"\b([A-Z]{2,4}\s?\d{2,3})\b")   # e.g. CS210 or EE 260
DATE_RE   = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")      # strict ISO format (YYYY-MM-DD)


def extract_filters(text: str) -> Tuple[str | None, str | None, str | None, str | None]:
    """
    Parse a natural-language query and return:
      (course_name, course_number, date_from, date_to), each may be None.
    """
    # 1) Course number via regex
    number = None
    m = COURSE_RE.search(text)
    if m:
        normalized = m.group(1).replace(" ", "")
        if normalized in COURSE_NUMBERS:
            number = normalized

    # 2) Course name via simple substring (case-insensitive)
    name = next((c for c in COURSE_NAMES if c.lower() in text.lower()), None)

    # 3) ISO date (YYYY-MM-DD)
    m = DATE_RE.search(text)
    if m:
        d = m.group(1)
        return name, number, d, d

    # 4) Natural-language dates (e.g. "May 29 2025")
    dates = []
    for tok in text.split():
        try:
            parsed = dateparse.parse(tok, fuzzy=False, default=dt.datetime(1900, 1, 1))
            if parsed.year > 1901:
                dates.append(parsed.date().isoformat())
        except Exception:
            pass

    if len(dates) == 1:
        return name, number, dates[0], dates[0]
    if len(dates) >= 2:
        return name, number, min(dates), max(dates)

    return name, number, None, None


# ==========================================================================
# 2.  Embeddings + vector store
# ==========================================================================
embedder     = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = SKLearnVectorStore(embedding=embedder)


# ==========================================================================
# 3.  Load 8-bit quantized LLMs
# ==========================================================================
def _init_pipe(repo: str):
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        repo,
        quantization_config=bnb_cfg,
        device_map="auto",
    ).eval()
    pipe = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        device_map="auto",
    )
    return pipe, tok, mdl


pipe_ds, tok_ds, mdl_ds = _init_pipe("deepseek-ai/deepseek-llm-7b-base")
pipe_ll, tok_ll, mdl_ll = _init_pipe("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")


# ==========================================================================
# 4.  GPU monitoring
# ==========================================================================
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


# ==========================================================================
# 5.  Retrieve RAG context (with optional metadata filters)
# ==========================================================================
def get_context(
    query: str,
    *,
    k: int = 5,
    course_name:   str | None = None,
    course_number: str | None = None,
    date_from:     str | None = None,
    date_to:       str | None = None,
) -> str:
    """
    Return the top-k most similar text chunks for `query`.
    If any filter is provided, restrict to filenames matching those metadata constraints.
    """
    total = _vs_count(vector_store)
    if total == 0:
        return ""

    restrict = filenames_by_meta(course_name, course_number, date_from, date_to)

    if restrict:
        docs_all = vector_store.similarity_search(query, k=total)
        docs = [d for d in docs_all if d.metadata["filename"] in restrict][:k]
    else:
        docs = vector_store.similarity_search(query, k=min(k, total))

    return "\n\n".join(d.page_content for d in docs)


# ==========================================================================
# 6.  Compute perplexity for given prompt, model, tokenizer
# ==========================================================================
def compute_perplexity(prompt: str, model, tokenizer) -> float:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()


# ==========================================================================
# 7.  Generate text via pipeline
# ==========================================================================
def _generate(pipe, prompt: str) -> str:
    return pipe(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        return_full_text=False,
    )[0]["generated_text"]


# ==========================================================================
# 8.  Metrics CSV setup
# ==========================================================================
CSV_PATH = BASE_DIR / "metrics.csv"
CSV_FIELDS = [
    "timestamp",
    "prompt",
    "model_used",
    "tokens",
    "time_s",
    "toks_per_s",
    "peak_mem_MiB",
    "energy_J",
    "similarity",
    "perplexity",
]

if not CSV_PATH.exists():
    with CSV_PATH.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def _append_csv(row: Dict):
    with CSV_PATH.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


# ==========================================================================
# 9.  Single-model evaluation + logging
# ==========================================================================
def _run_one(
    query: str,
    *,
    pipe,
    tokenizer,
    model,
    model_name: str,
    course_name:   str | None,
    course_number: str | None,
    date_from:     str | None,
    date_to:       str | None,
) -> Dict:
    """
    Run exactly one model (pipe+tokenizer+model_name) on 'query' with auto filters:
      - build RAG context
      - generate answer
      - compute similarity between answer & context
      - compute perplexity on the prompt
      - measure GPU memory, energy, throughput
      - log a row to metrics.csv
      - return a dict including all metrics + 'answer'
    """
    # Reset GPU stats
    torch.cuda.reset_peak_memory_stats()
    p0 = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000
    t0 = time.time()

    # Build RAG context (with metadata)
    context = get_context(
        query,
        course_name=course_name,
        course_number=course_number,
        date_from=date_from,
        date_to=date_to,
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = _generate(pipe, prompt)
    n_tokens = len(tokenizer.encode(answer))

    # Similarity: answer vs context
    similarity = 0.0
    if context.strip():
        vecs = embedder.embed_documents([answer, context])
        similarity = float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])

    # Perplexity of the prompt
    perplexity = compute_perplexity(prompt, model, tokenizer)

    # Final GPU stats
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    p1 = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000
    energy = ((p0 + p1) / 2) * elapsed

    # Log to CSV
    row = {
        "timestamp": dt.datetime.utcnow().isoformat(),
        "prompt": query,
        "model_used": model_name,
        "tokens": n_tokens,
        "time_s": elapsed,
        "toks_per_s": n_tokens / elapsed if elapsed else None,
        "peak_mem_MiB": peak_mem,
        "energy_J": energy,
        "similarity": similarity,
        "perplexity": perplexity,
    }
    _append_csv(row)

    return {**row, "answer": answer}


# ==========================================================================
# 10.  Public evaluation functions
# ==========================================================================
def evaluate_model(query: str, model_choice: str) -> Dict:
    """
    Single-model endpoint:
      - If model_choice == "deepseek", run DeepSeek-7B
      - If model_choice == "llama", run LLaMA-3.1-8B
    Returns all metrics + answer.
    """
    cname, cnum, d_from, d_to = extract_filters(query)

    if model_choice == "deepseek":
        return _run_one(
            query,
            pipe=pipe_ds,
            tokenizer=tok_ds,
            model=mdl_ds,
            model_name="DeepSeek-7B",
            course_name=cname,
            course_number=cnum,
            date_from=d_from,
            date_to=d_to,
        )
    elif model_choice == "llama":
        return _run_one(
            query,
            pipe=pipe_ll,
            tokenizer=tok_ll,
            model=mdl_ll,
            model_name="LLaMA-3.1-8B",
            course_name=cname,
            course_number=cnum,
            date_from=d_from,
            date_to=d_to,
        )
    else:
        raise ValueError("model must be 'deepseek' or 'llama'")


def evaluate_both_models(query: str) -> Dict:
    """
    Dual-model endpoint:
      - Runs both DeepSeek-7B and LLaMA-3.1-8B on the same query (with metadata filters).
      - Logs two rows in metrics.csv (one per model).
      - Returns a dict containing both results and 'chosen' based on lower perplexity.
    """
    cname, cnum, d_from, d_to = extract_filters(query)

    res_ds = _run_one(
        query,
        pipe=pipe_ds,
        tokenizer=tok_ds,
        model=mdl_ds,
        model_name="DeepSeek-7B",
        course_name=cname,
        course_number=cnum,
        date_from=d_from,
        date_to=d_to,
    )

    res_ll = _run_one(
        query,
        pipe=pipe_ll,
        tokenizer=tok_ll,
        model=mdl_ll,
        model_name="LLaMA-3.1-8B",
        course_name=cname,
        course_number=cnum,
        date_from=d_from,
        date_to=d_to,
    )

    chosen = "deepseek" if res_ds["perplexity"] < res_ll["perplexity"] else "llama"
    return {
        "chosen": chosen,
        "deepseek": res_ds,
        "llama": res_ll,
    }
