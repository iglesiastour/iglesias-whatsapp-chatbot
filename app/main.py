from fastapi import FastAPI

from app.config import settings
from app.routes.messages import router as messages_router
from app.routes.operator_handoffs import router as operator_handoffs_router


app = FastAPI(
    title="Iglesias WhatsApp Chatbot",
    version="0.1.0",
    description="Local API foundation for the Iglesias Tour Turkey chatbot.",
)

app.include_router(messages_router, prefix="/api/v1")
app.include_router(operator_handoffs_router, prefix="/api/v1")


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Report whether the API process is healthy."""
    return {"status": "ok", "service": settings.service_name}
