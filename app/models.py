import time, torch, pynvml
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    pipeline, AutoTokenizer, AutoModelForCausalLM
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.sklearn import SKLearnVectorStore
from transformers import BitsAndBytesConfig  

# ---------- embeddings & vector store ---------- #
embedder     = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = SKLearnVectorStore(embedding=embedder)

# ---------- load language models (8‑bit) ---------- #
def _init_pipe(repo: str):
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True) 
    mdl = AutoModelForCausalLM.from_pretrained(
        repo,
        quantization_config=bnb_cfg,
        device_map="auto"
    ).eval()
    pipe = pipeline("text-generation",
                    model=mdl,
                    tokenizer=tok,
                    device_map="auto")
    return pipe, tok, mdl

pipe_ds, tok_ds, mdl_ds = _init_pipe("deepseek-ai/deepseek-llm-7b-base")
pipe_ll, tok_ll, mdl_ll = _init_pipe(
    "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
)

# ---------- GPU power stats ---------- #
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# ---------- RAG helpers ---------- #
def get_context(query: str, k: int = 5) -> str:
    total = len(vector_store._embeddings)
    if total == 0:
        return ""
    docs = vector_store.similarity_search(query, k=min(k, total))
    return "\n\n".join(d.page_content for d in docs)

def compute_perplexity(prompt: str, model, tok) -> float:
    inputs = tok(prompt, return_tensors="pt",
                 truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()

def _generate(pipe, prompt: str) -> str:
    return pipe(prompt, max_new_tokens=256,
                do_sample=False, return_full_text=False)[0]["generated_text"]

def select_best_model(query: str) -> str:
    context = get_context(query)
    prompt  = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    ppl_ds = compute_perplexity(prompt, mdl_ds, tok_ds)
    ppl_ll = compute_perplexity(prompt, mdl_ll, tok_ll)
    return "deepseek" if ppl_ds < ppl_ll else "llama"

def evaluate_model(query: str, model: str) -> dict:
    if model == "deepseek":
        pipe, tok, mdl = pipe_ds, tok_ds, mdl_ds
        name = "DeepSeek‑7B"
    elif model == "llama":
        pipe, tok, mdl = pipe_ll, tok_ll, mdl_ll
        name = "LLaMA‑3.1‑8B"
    else:
        raise ValueError("model must be 'deepseek' or 'llama'")

    torch.cuda.reset_peak_memory_stats()
    p0   = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000
    t0   = time.time()

    prompt  = f"Context:\n{get_context(query)}\n\nQuestion: {query}\nAnswer:"
    answer  = _generate(pipe, prompt)
    ntokens = len(tok.encode(answer))

    torch.cuda.synchronize()
    elapsed  = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    p1       = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000
    energy   = ((p0 + p1) / 2) * elapsed

    return dict(model_used=name, answer=answer,
                tokens=ntokens, time_s=elapsed,
                toks_per_s=ntokens/elapsed if elapsed else None,
                peak_mem_MiB=peak_mem, energy_J=energy)

# convenience wrappers
def run_deepseek(prompt: str) -> str: return _generate(pipe_ds, prompt)
def run_llama(prompt: str)    -> str: return _generate(pipe_ll, prompt)
