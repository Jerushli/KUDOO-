from typing import Optional
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from .agent import run_agent


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="KUDOO AI Data Analyst",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("message must not be empty")

        return value


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    # --------------------------------------------------
    # Create conversation ID if needed
    # --------------------------------------------------

    conversation_id = (
        request.conversation_id
        or str(uuid.uuid4())
    )

    # --------------------------------------------------
    # Run KUDOO
    # --------------------------------------------------

    result = run_agent(
        user_question=request.message,
        conversation_id=conversation_id,
    )

    # --------------------------------------------------
    # Return structured response
    # --------------------------------------------------

    return {
        "answer": result["answer"],
        "conversation_id": result["conversation_id"],
        "tool_calls": result["tool_calls"],
        "execution_time": result["execution_time"],
        "conversation": result["conversation"],
        "trace": result["trace"],
    }