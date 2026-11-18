from typing import Literal
from pydantic import BaseModel, Field
from api.app.config import settings

class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: Literal["ok"] = "ok"
    mode: Literal["stub", "classical", "transformer"]

class AnalyzeRequest(BaseModel):
    """Analyze endpoint request payload with strict length validation."""
    text: str = Field(
        ...,
        min_length=settings.MIN_TEXT_LENGTH,
        max_length=settings.MAX_TEXT_LENGTH,
        description=f"Review text to analyze ({settings.MIN_TEXT_LENGTH}-{settings.MAX_TEXT_LENGTH} chars)",
        examples=["This was an unforgettable stay! The staff was incredibly welcoming and our room had breathtaking views."]
    )

class AuthorshipResult(BaseModel):
    """Authorship classification outcome."""
    ai_probability: float = Field(..., ge=0.0, le=1.0)
    label: Literal["human", "ai"]

class DeceptionResult(BaseModel):
    """Deception classification outcome."""
    deceptive_probability: float = Field(..., ge=0.0, le=1.0)
    label: Literal["genuine", "deceptive"]

class TokenExplanation(BaseModel):
    """Token attribution weight. Positive weight pushes towards AI or deceptive."""
    token: str
    weight: float

class Explanations(BaseModel):
    """Feature attributions for both classification dimensions."""
    authorship: list[TokenExplanation]
    deception: list[TokenExplanation]

class ModelInfo(BaseModel):
    """Active model inference metadata."""
    mode: Literal["stub", "classical", "transformer"] | str

class AnalyzeResponse(BaseModel):
    """Full analysis response combining independent predictions, verdict, and explanations."""
    authorship: AuthorshipResult
    deception: DeceptionResult
    verdict: Literal["genuine_human", "deceptive_human", "ai_genuine", "ai_deceptive"]
    explanations: Explanations
    model_info: ModelInfo
    warnings: list[str] = Field(default_factory=list)

