import os

from dotenv import load_dotenv


load_dotenv()


APP_NAME = os.getenv("APP_NAME", "KUDOO")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()