from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/health", summary="Health check")
def health() -> HealthResponse:
    """Answers without touching the database or OpenAI."""
    return HealthResponse(status="ok")
