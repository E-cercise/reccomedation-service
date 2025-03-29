from fastapi import FastAPI
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from utils.vector_cache import load_vector_cache
from fastapi.middleware.cors import CORSMiddleware
from api.route import router
from config import ALLOWED_ORIGINS, DEBUG


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Loading model and vector cache...")
    app.state.model = SentenceTransformer("all-MiniLM-L6-v2")
    app.state.vector_data = load_vector_cache()
    print("✅ Loaded.")
    yield
    print("🔻 Shutting down...")

app = FastAPI(debug=DEBUG)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)