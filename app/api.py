from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid

from .agent import run_agent


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):

    # Create a new conversation ID if this is a new conversation
    conversation_id = request.conversation_id or str(uuid.uuid4())

    result = run_agent(
        user_question=request.message,
        conversation_id=conversation_id,
    )

    return {
        "answer": result["answer"],
        "conversation_id": conversation_id,
        "tool_calls": result["tool_calls"],
        "execution_time": result["execution_time"],
        "trace": result["trace"],
    }