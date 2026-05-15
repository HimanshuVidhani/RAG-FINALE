"""
Embedding Module for Research Synthesis Engine.
Handles text embedding generation using Google's Gemini embedding model.
"""

import time
import logging
from typing import List, Optional
from google import genai
from google.genai import errors as genai_errors

logger = logging.getLogger(__name__)

# Gemini embedding model limits
MAX_BATCH_SIZE = 20          # Safe batch size (API allows up to 100 but smaller = more stable)
MAX_CHARS_PER_TEXT = 9000    # ~2048 tokens; Gemini embedding input limit
MAX_RETRIES = 3
RETRY_DELAY = 2.0            # seconds between retries


class EmbeddingManager:
    """Manages text embedding generation using Google Gemini."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        """
        Initialize the embedding manager.

        Args:
            api_key:    Google Gemini API key
            model_name: Embedding model to use.
                        Recommended: "text-embedding-004" (768-dim, stable)
        """
        if not api_key:
            raise ValueError("API key must not be empty.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts:     List of text strings to embed.
            task_type: Task type for embedding optimisation.
                       One of: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY |
                               SEMANTIC_SIMILARITY | CLASSIFICATION | CLUSTERING

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []

        cleaned = [self._truncate(t) for t in texts]
        embeddings: List[List[float]] = []

        for i in range(0, len(cleaned), MAX_BATCH_SIZE):
            batch = cleaned[i : i + MAX_BATCH_SIZE]
            batch_embeddings = self._embed_batch_with_retry(batch, task_type)
            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a single query string.

        Args:
            query: Query string.

        Returns:
            Embedding vector as a list of floats.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        result = self._embed_batch_with_retry(
            [self._truncate(query)], task_type="RETRIEVAL_QUERY"
        )
        return result[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_batch_with_retry(
        self,
        batch: List[str],
        task_type: str,
        retries: int = MAX_RETRIES,
    ) -> List[List[float]]:
        """Call the API with exponential-backoff retry on transient errors."""
        last_error: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config={"task_type": task_type},
                )
                return [emb.values for emb in result.embeddings]

            except genai_errors.ClientError as e:
                status = getattr(e, "status_code", None)

                # 400 Bad Request — bad input, retrying won't help
                if status == 400:
                    raise RuntimeError(
                        f"Bad request (400) — check model name or input text.\n"
                        f"Model: {self.model_name}\n"
                        f"First text in batch (truncated): {batch[0][:200]!r}\n"
                        f"Original error: {e}"
                    ) from e

                # 401 / 403 — API key / permissions issue
                if status in (401, 403):
                    raise RuntimeError(
                        f"Authentication error ({status}) — check your GOOGLE_API_KEY "
                        f"in Streamlit secrets.\nOriginal error: {e}"
                    ) from e

                # 429 Rate limit — back off and retry
                if status == 429:
                    wait = RETRY_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Rate limited (429). Waiting %.1fs before retry %d/%d.",
                        wait, attempt, retries,
                    )
                    time.sleep(wait)
                    last_error = e
                    continue

                # 5xx Server errors — retry with backoff
                if status and status >= 500:
                    wait = RETRY_DELAY * attempt
                    logger.warning(
                        "Server error (%s). Waiting %.1fs before retry %d/%d.",
                        status, wait, attempt, retries,
                    )
                    time.sleep(wait)
                    last_error = e
                    continue

                # Any other ClientError — raise immediately with full detail
                raise RuntimeError(
                    f"Gemini API ClientError (status={status}):\n{e}"
                ) from e

            except Exception as e:
                # Unexpected error — surface it immediately
                raise RuntimeError(
                    f"Unexpected error during embedding: {type(e).__name__}: {e}"
                ) from e

        raise RuntimeError(
            f"Embedding failed after {retries} retries. Last error: {last_error}"
        )

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate text to the model's safe character limit."""
        text = text.strip()
        if len(text) > MAX_CHARS_PER_TEXT:
            logger.warning(
                "Text truncated from %d to %d chars.", len(text), MAX_CHARS_PER_TEXT
            )
            return text[:MAX_CHARS_PER_TEXT]
        return text
