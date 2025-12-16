"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY")
    
    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    
    # LLM API Keys
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    
    # Application
    environment: Literal["development", "production"] = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_upload_size_mb: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")
    
    # Embedding
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # LLM
    default_llm_model: str = Field(default="llama-3.3-70b-versatile", env="DEFAULT_LLM_MODEL")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    
    # Retrieval
    basic_top_k: int = Field(default=5, env="BASIC_TOP_K")
    advanced_top_k: int = Field(default=10, env="ADVANCED_TOP_K")
    
    # Storage
    storage_path: str = Field(default="./storage", env="STORAGE_PATH")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
