import os
from contextlib import asynccontextmanager
from pathlib import Path

import requests as _requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import chat

FAISS_INDEX_PATH = Path(__file__).parent.parent / "faiss_index"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not FAISS_INDEX_PATH.exists():
        print("FAISS index not found — running ingestion on startup…")
        from rag.ingest import ingest_documents
        ingest_documents()
        print("Ingestion complete.")
    else:
        print("FAISS index found — skipping ingestion.")

    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
    try:
        _requests.get(f"{mcp_url}/health", timeout=2)
        print(f"MCP Tool Server reachable at {mcp_url} (OK)")
    except Exception:
        print(
            f"\n! WARNING: MCP Tool Server is NOT reachable at {mcp_url}.\n"
            "   Operations tools (booking, availability, specials, loyalty) will fail.\n"
            "   Start the MCP server in a separate terminal:\n"
            "   -> python tools/mcp_server.py\n"
        )

    yield


app = FastAPI(
    title="NovaBite AI Assistant",
    description="Multi-agent RAG system for NovaBite restaurant chain",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    agent_used: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "NovaBite AI Assistant", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id must not be empty")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    result = chat(request.session_id, request.message)
    return ChatResponse(**result)
