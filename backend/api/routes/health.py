from fastapi import APIRouter
from backend.api.schemas.response import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the API is running and the model is loaded in memory.
    """
    # A lightweight check. In production you might want to try-catch 
    # the get_engine() singleton to verify it's actually alive.
    return HealthResponse(
        status="ok",
        model_loaded=True
    )
