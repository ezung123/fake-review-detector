from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    """Application configuration settings."""
    PROJECT_NAME: str = "Fake and AI-Generated Review Detector"
    VERSION: str = "1.0.0"
    
    # Classification decision thresholds (0.0 to 1.0)
    AUTHORSHIP_THRESHOLD: float = 0.5
    DECEPTION_THRESHOLD: float = 0.5
    
    # Text length validation constraints
    MIN_TEXT_LENGTH: int = 20
    MAX_TEXT_LENGTH: int = 3000
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Allowed origins for CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "*",  # Permissive for development
    ]

settings = Settings()

