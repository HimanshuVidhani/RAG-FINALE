"""
Embedding Module for Research Synthesis Engine.
Handles text embedding generation using Google's Gemini embedding model.
"""

from typing import List
from google import genai


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
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config={"task_type": task_type},
            )
            for emb in result.embeddings:
                embeddings.append(emb.values)
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector
        """
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=query,
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        return result.embeddings[0].values
