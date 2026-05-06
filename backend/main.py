from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.core.config import settings
from backend.core.inference import get_engine
from backend.api.routes import health, predict

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager runs code before the app starts accepting requests.
    We use this to pre-load the ML model so the first request doesn't timeout.
    """
    get_engine()  # Initializes the singleton and loads the TF model
    yield
    # Cleanup code goes here (if needed)

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan
)

# Configure CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Inference"])

if __name__ == "__main__":
    # Start the server locally when running the file directly
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
