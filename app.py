# app.py
import json
import time
import traceback
from typing import List, Dict, Any, Annotated

from fastapi import FastAPI, HTTPException, Form, Request
from openai import RateLimitError, APIConnectionError
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from definitions import ROOT_DIR, client
from llm_embeddings.scripts.faiss_index import load_faiss_index, search_index

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

load_dotenv()  # Load environment variables

FAISS_INDEX_PATH = f"{ROOT_DIR}/data/faiss/templ.index"
EMBEDDINGS_JSON_PATH = f"{ROOT_DIR}/data/embeddings/templ_embeddings.json"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536  # For text-embedding-ada-002

# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    use_chat: bool = False  # If True, we'll call ChatCompletion

class QueryResponse(BaseModel):
    query: str
    matched_chunks: List[Dict[str, Any]]
    answer: str = ""  # only if use_chat = True

# ------------------------------------------------------------------------------
# Initialize FastAPI
# ------------------------------------------------------------------------------

app = FastAPI(
    title="Templ Docs Query API",
    description="Queries the Faiss index with Templ doc embeddings",
    version="1.0.0",
)

# ------------------------------------------------------------------------------
# Load FAISS index & Embeddings once on startup
# ------------------------------------------------------------------------------

faiss_index = None
embeddings_data = None

@app.on_event("startup")
def load_resources():
    global faiss_index, embeddings_data
    # 1. Load FAISS index
    try:
        faiss_index = load_faiss_index(FAISS_INDEX_PATH)
        print(f"FAISS index loaded with {faiss_index.ntotal} embeddings.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        raise e

    # 2. Load embeddings JSON
    try:
        with open(EMBEDDINGS_JSON_PATH, "r", encoding="utf-8") as f:
            embeddings_data = json.load(f)
        print(f"Loaded {len(embeddings_data)} embeddings from JSON.")
    except Exception as e:
        print(f"Error loading embeddings JSON: {e}")
        raise e

# ------------------------------------------------------------------------------
# Helper: Create an embedding from a user query
# ------------------------------------------------------------------------------

def create_query_embedding(query: str, max_retries: int = 5, base_delay: float = 1.0) -> List[float]:
    backoff = base_delay
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query
            )
            embedding = response.data[0].embedding
            return embedding
        except RateLimitError as e:
            print(f"[RateLimitError] Attempt {attempt+1}/{max_retries}. Waiting {backoff} seconds and retrying...")
            time.sleep(backoff)
            backoff *= 2
        except APIConnectionError as e:
            print(f"[APIConnectionError] Attempt {attempt+1}/{max_retries}. Waiting {backoff} seconds and retrying...")
            time.sleep(backoff)
            backoff *= 2
        except Exception as e:
            print(f"[Error] {e}")
            raise e
    raise Exception(f"Failed to create query embedding after {max_retries} retries.")

# ------------------------------------------------------------------------------
# Helper: Retrieve matched chunks from FAISS index
# ------------------------------------------------------------------------------

def retrieve_matched_chunks(query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    # 1. Search the index
    distances, indices = search_index(faiss_index, query_embedding, top_k=top_k)

    # 2. Convert results to a list of matched chunks
    matched_chunks = []
    for rank, idx in enumerate(indices[0]):
        # idx is the row in embeddings_data
        if idx < 0 or idx >= len(embeddings_data):
            continue
        chunk_data = embeddings_data[idx]
        matched_chunks.append({
            "rank": rank,
            "score": float(distances[0][rank]),
            # "source": chunk_data["chunk"].get("source", ""),
            "text": chunk_data["chunk"]
        })
    return matched_chunks

# ------------------------------------------------------------------------------
# Optional: Summarize or Answer with ChatCompletion
# ------------------------------------------------------------------------------

def answer_with_chat(query: str, matched_chunks: List[Dict[str, Any]]) -> str:
    """
    Pass the matched chunks + user query into ChatCompletion to get a summarized or direct answer.
    """
    # Build a prompt or system content
    context_text = "\n".join(
        f"CHUNK {c['rank']} (score: {c['score']}):\n{c['text']}"
        for c in matched_chunks
    )

    system_prompt = f"""You are a helpful AI with knowledge of Templ (a Go templating library).
Use the following context to answer the user's question:
{context_text}

If the answer is not in the context, say you don't have enough information.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    try:
        chat_response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7)
        return chat_response.choices[0].message.content
    except Exception as e:
        print(f"[Error in ChatCompletion] {e}")
        traceback.print_exc()

    return "Sorry, I ran into an error while generating an answer."


@app.post("/query-augmented", response_model=QueryResponse)
def augment_response_with_docs(request: QueryRequest):
    # 1. Embed user query
    query_vector = create_query_embedding(request.query)

    # 2. Vector search in Faiss
    distances, indices = search_index(faiss_index, query_vector, top_k=request.top_k)

# 2. Convert results to a list of matched chunks
    context_snippets = []
    matched_chunks = retrieve_matched_chunks(query_vector, request.top_k)
    for c in matched_chunks:
        context_snippets.append(c["text"])

# 4. Build final prompt
    system_text = f"""You are a helpful AI assistant. Here are some doc snippets that might be relevant:
    {context_snippets}

    You can also use your general knowledge. 
    If the user question is about Templ, rely on the doc context plus your broader knowledge.
    """
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": request.query}
    ]

    # 5. Call OpenAI ChatCompletion
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return QueryResponse(
        query=request.query,
        matched_chunks=matched_chunks,
        answer=response.choices[0].message.content
    )
# ------------------------------------------------------------------------------
# API Endpoint: /query
# ------------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query_docs(req: QueryRequest):
    """
    Query the Faiss index with a user-provided query, retrieve top_k chunks,
    optionally call ChatCompletion for an LLM-based answer.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Create embedding for the user query
    try:
        query_embedding = create_query_embedding(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 2. Retrieve matched chunks
    matched_chunks = retrieve_matched_chunks(query_embedding, req.top_k)

    # 3. If user wants an LLM-based answer, call ChatCompletion
    answer_text = ""
    if req.use_chat:
        answer_text = answer_with_chat(req.query, matched_chunks)

    # 4. Return results
    return QueryResponse(
        query=req.query,
        matched_chunks=matched_chunks,
        answer=answer_text
    )


templates = Jinja2Templates(directory="templates")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def form_post(request: Request) :
    return templates.TemplateResponse('home.html', context={'request': request})



@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str):
    query = QueryRequest(
        query=q,
        use_chat=True,
        top_k=5,
    )
    query_response = query_docs(query)
    return templates.TemplateResponse(
        request=request, name="search-result.html", context={"query": q, "query_response": query_response}
    )