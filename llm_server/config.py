"""Configuration for the LLM server."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Server configuration settings."""

    host: str = "127.0.0.1"
    port: int = 8777
    model_name: Optional[str] = None  # None = use llm library default (gpt-4o-mini)
    debug: bool = False
    pidfile: Optional[str] = None
    logfile: Optional[str] = None

    class Config:
        env_prefix = "LLM_SERVER_"


settings = Settings()


def is_gemini_model(model_name: str) -> bool:
    """Check if the model is a Gemini/Vertex model."""
    if not model_name:
        return False
    return model_name.startswith("gemini/") or model_name.startswith("vertex/")
