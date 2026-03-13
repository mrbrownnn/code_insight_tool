"""
Code Embedder — generates embeddings using UniXcoder.
Specialized for code understanding across 6 programming languages.
"""

from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class CodeEmbedder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        logger.info(f"Model loaded successfully ({self.device})")

    def embed(self, text: str) -> List[float]:
        self._load_model()
        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**tokens)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :]

        return embedding.squeeze().cpu().tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of code/text strings.

        Returns:
            List of embedding vectors.
        """
        self._load_model()

        tokens = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings.cpu().tolist()

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return settings.embedding_dim
