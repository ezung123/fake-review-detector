from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.app.config import settings
from api.app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
)
from api.app.services.analyzer import analyze_review, detect_runtime_mode

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for independent Authorship and Deception analysis of reviews.",
)

# Enable CORS for the Next.js web application
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check returning operational status and current model mode
    ('stub', 'classical', or 'transformer').
    """
    mode = detect_runtime_mode()
    return HealthResponse(status="ok", mode=mode)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze review text across both Authorship and Deception dimensions.
    Returns probabilities, labels, combined verdict, and token-level explanations.
    """
    text = payload.text.strip()
    if len(text) < settings.MIN_TEXT_LENGTH or len(text) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Text length must be between {settings.MIN_TEXT_LENGTH} and {settings.MAX_TEXT_LENGTH} characters.",
        )
    return analyze_review(text)

