"""
Embedding Module for Research Synthesis Engine.
Handles text embedding generation using Google's Gemini embedding model.
"""

from typing import List
import time
import logging
from google import genai

logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embedding generation using Google Gemini."""
    
    def __init__(self, api_key: str):
        """
        Initialize the embedding manager.
        
        Args:
            api_key: Google Gemini API key
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-embedding-001"
    
    def embed_texts(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            task_type: Task type for embedding optimization
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = self._embed_batch_with_retry(batch, task_type=task_type)
            for emb in result.embeddings:
                embeddings.append(emb.values)

        return embeddings

    def _embed_batch_with_retry(self, batch: List[str], task_type: str = "RETRIEVAL_DOCUMENT", retries: int = 3):
        """Call the embedding API with simple exponential-backoff retry.

        Raises a descriptive exception if all retries fail.
        """
        backoff = 1.0
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                return self.client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=batch,
                    config={"task_type": task_type},
                )
            except Exception as e:
                last_exc = e
                logging.warning("Embed attempt %s failed: %s", attempt, e)
                if attempt < retries:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise RuntimeError(f"Embedding API failed after {retries} attempts: {e}")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector
        """
        try:
            result = self.client.models.embed_content(
                model="gemini-embedding-2",
                contents=query,
                config={"task_type": "RETRIEVAL_QUERY"},
            )
            return result.embeddings[0].values
        except Exception as e:
            logging.exception("Query embedding failed")
            raise RuntimeError(f"Query embedding failed: {e}")
