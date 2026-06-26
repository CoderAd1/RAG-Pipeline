"""Embedding service — uses fastembed (ONNX) when available, falls back to sentence-transformers."""

from typing import List
import numpy as np
from loguru import logger
from app.core.config import settings

try:
    from fastembed import TextEmbedding as FastEmbed
    _BACKEND = "fastembed"
except ImportError:
    FastEmbed = None
    _BACKEND = None

if _BACKEND is None:
    try:
        from sentence_transformers import SentenceTransformer
        _BACKEND = "sentence_transformers"
    except ImportError:
        SentenceTransformer = None

if _BACKEND is None:
    raise ImportError(
        "No embedding backend found. Install fastembed (recommended) or sentence-transformers."
    )

logger.info(f"Embedding backend: {_BACKEND}")


class EmbeddingService:
    """Singleton embedding service supporting fastembed and sentence-transformers."""

    _instance: "EmbeddingService" = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            if _BACKEND == "fastembed":
                self._model = FastEmbed(settings.embedding_model)
            else:
                self._model = SentenceTransformer(settings.embedding_model)
            logger.success("Embedding model loaded successfully")

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * settings.embedding_dimension

        if _BACKEND == "fastembed":
            return list(self._model.embed([text]))[0].tolist()
        else:
            return self._model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        valid_texts, valid_indices = [], []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            logger.warning("No valid texts provided for batch embedding")
            return [[0.0] * settings.embedding_dimension] * len(texts)

        logger.debug(f"Generating embeddings for {len(valid_texts)} texts")

        if _BACKEND == "fastembed":
            embeddings = [e.tolist() for e in self._model.embed(valid_texts)]
        else:
            embeddings = self._model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(valid_texts) > 100,
            ).tolist()

        result = [[0.0] * settings.embedding_dimension for _ in range(len(texts))]
        for idx, embedding in zip(valid_indices, embeddings):
            result[idx] = embedding

        logger.debug(f"Generated {len(result)} embeddings")
        return result

    def get_embedding_dimension(self) -> int:
        return settings.embedding_dimension

    def get_model_name(self) -> str:
        return settings.embedding_model


# Global instance
embedding_service = EmbeddingService()
