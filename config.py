# config.py
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8888")
RECOMMENDER_PORT = int(os.getenv("RECOMMENDER_PORT", 8000))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
