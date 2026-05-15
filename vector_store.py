"""
Vector Store Module for Research Synthesis Engine.
Manages ChromaDB collections for multi-paper vector storage and retrieval.
"""

import os
import chromadb
from typing import List, Dict, Optional, Tuple
from chromadb.config import Settings


class VectorStoreManager:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory for persisting ChromaDB data
        """
        self.persist_directory = persist_directory
        # Use EphemeralClient for Streamlit Cloud compatibility
        self.client = chromadb.EphemeralClient(Settings(
            anonymized_telemetry=False,
        ))
        self.collection = None
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create or get the research papers collection."""
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> int:
        """
        Add document chunks with their embeddings to the vector store.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'
            embeddings: Corresponding embedding vectors
            
        Returns:
            Number of chunks added
        """
        if not chunks or not embeddings:
            return 0
        
        ids = []
        documents = []
        metadatas = []
        
        existing_count = self.collection.count()
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{existing_count + i}_{chunk['metadata']['paper_id']}_{chunk['metadata']['page_number']}_{chunk['metadata']['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            # ChromaDB requires metadata values to be str, int, float, or bool
            clean_meta = {}
            for k, v in chunk["metadata"].items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        return len(ids)
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        paper_filter: Optional[str] = None,
    ) -> Dict:
        """
        Query the vector store for relevant chunks.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            paper_filter: Optional paper_id to filter results
            
        Returns:
            Dict with documents, metadatas, and distances
        """
        where_filter = None
        if paper_filter:
            where_filter = {"paper_id": paper_filter}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        return results
    
    def query_multiple_papers(
        self,
        query_embedding: List[float],
        paper_ids: List[str],
        n_per_paper: int = 3,
    ) -> Dict[str, List[Dict]]:
        """
        Query and return results grouped by paper.
        
        Args:
            query_embedding: Query vector
            paper_ids: List of paper IDs to search across
            n_per_paper: Results per paper
            
        Returns:
            Dict mapping paper_id to list of result dicts
        """
        grouped_results = {}
        
        for paper_id in paper_ids:
            results = self.query(
                query_embedding=query_embedding,
                n_results=n_per_paper,
                paper_filter=paper_id,
            )
            
            paper_results = []
            if results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    paper_results.append({
                        "text": doc,
                        "metadata": meta,
                        "relevance_score": 1 - dist,  # Convert distance to similarity
                    })
            
            grouped_results[paper_id] = paper_results
        
        return grouped_results
    
    def get_all_paper_ids(self) -> List[str]:
        """Get all unique paper IDs in the collection."""
        if self.collection.count() == 0:
            return []
        
        # Get all metadatas
        all_data = self.collection.get(include=["metadatas"])
        paper_ids = set()
        for meta in all_data["metadatas"]:
            if "paper_id" in meta:
                paper_ids.add(meta["paper_id"])
        
        return sorted(list(paper_ids))
    
    def get_paper_info(self) -> List[Dict]:
        """Get metadata about all papers in the store."""
        if self.collection.count() == 0:
            return []
        
        all_data = self.collection.get(include=["metadatas"])
        papers = {}
        
        for meta in all_data["metadatas"]:
            pid = meta.get("paper_id", "unknown")
            if pid not in papers:
                papers[pid] = {
                    "paper_id": pid,
                    "title": meta.get("paper_title", "Unknown"),
                    "author": meta.get("paper_author", "Unknown"),
                    "filename": meta.get("filename", "Unknown"),
                    "total_pages": meta.get("total_pages", 0),
                    "chunk_count": 0,
                }
            papers[pid]["chunk_count"] += 1
        
        return list(papers.values())
    
    def clear_collection(self):
        """Delete and recreate the collection."""
        self.client.delete_collection("research_papers")
        self._ensure_collection()
    
    def remove_paper(self, paper_id: str):
        """Remove all chunks belonging to a specific paper."""
        all_data = self.collection.get(
            where={"paper_id": paper_id},
            include=["metadatas"],
        )
        if all_data["ids"]:
            self.collection.delete(ids=all_data["ids"])
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks in the collection."""
        return self.collection.count()
