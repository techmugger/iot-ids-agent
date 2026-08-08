"""
FastAPI entrypoint. Run with:

    uvicorn src.agent.app:app --reload --port 8000

Then open http://localhost:8000 in a browser for the chat UI, or:
    curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \\
         -d '{"message": "Is my device secure right now?"}'
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.agent.triage import handle_user_input

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="IoT IDS Agentic Chatbot")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    path: str
    payload: dict


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = handle_user_input(req.message, session_id=req.session_id)
    return ChatResponse(
        session_id=result["session_id"],
        path=result["path"],
        payload=result,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


# Mount remaining static assets (if any css/js/images get added later)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
