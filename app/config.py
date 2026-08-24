import os

from dotenv import load_dotenv


load_dotenv()


APP_NAME = os.getenv("APP_NAME", "KUDOO")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT_SECONDS = float(
	os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")
)