from openai import OpenAI
import numpy as np
from loguru import logger
from typing import List, Dict, Optional
import os
import pickle
import json
from .network_utils import network_retry

class GroundTruthRAG:
    """
    Dense (embedding-based) RAG system for ground truth retrieval.
    
    For a given entity, builds an index from:
    - Wikipedia article paragraphs (original_content)
    - All web document paragraphs (from full web documents)
    
    Retrieves top-k passages using cosine similarity between query embedding
    and passage embeddings.
    """
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: OpenAI embedding model to use (default: text-embedding-3-small)
        """
        self.embedding_model = embedding_model
        self.client = OpenAI()
        self.passages = []
        self.embeddings = None
        
    def build_index(self, wiki_data: Dict) -> None:
        """
        Build passage index from ground truth data.
        
        Creates passages from:
        - Wikipedia article: each paragraph becomes a passage
        - Web documents: each paragraph becomes a passage with title prefix
        
        Args:
            wiki_data: Dictionary containing ground truth with keys:
                - original_content: full Wikipedia article text
                - web_full_documents: list of web document dicts with 'content' and 'title'
        """
        passages = []
        
        wikipedia_text = wiki_data.get('original_content', '')
        if wikipedia_text:
            paragraphs = wikipedia_text.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if len(para) > 50:
                    passages.append({
                        'content': para,
                        'source': 'wikipedia',
                        'metadata': {'type': 'article_paragraph'}
                    })
        
        web_docs = wiki_data.get('web_full_documents', [])
        for doc in web_docs:
            content = doc.get('content', '')
            title = doc.get('title', '')
            if content:
                # Split web document into paragraphs, just like Wikipedia
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        passages.append({
                            'content': f"{title}: {para}",
                            'source': 'web',
                            'metadata': {'type': 'document_paragraph', 'url': doc.get('url', ''), 'title': title}
                        })
        
        self.passages = passages
        self.embeddings = None
        
    def retrieve_top_k(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve top-k most relevant passages using dense retrieval.
        
        Uses embedding-based similarity (cosine) between query and passages.
        Falls back to keyword overlap if embeddings are not cached.
        
        Args:
            query: The query string (typically an RDF triple in format "(subject, predicate, object)")
            k: Number of passages to retrieve (default: 5)
            
        Returns:
            List of top-k passage dicts with keys: 'content', 'source', 'metadata'
        """
        if not self.passages:
            return []
        
        query_embedding = self._get_embedding(query)
        
        scores = []
        for i, passage in enumerate(self.passages):
            if self.embeddings is not None and i < len(self.embeddings):
                score = self._cosine_similarity(query_embedding, self.embeddings[i])
            else:
                score = self._keyword_overlap_score(query, passage['content'])
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:k]]
        
        return [self.passages[i] for i in top_indices]
    
    @network_retry(max_retries=6, max_delay=120)
    def _get_embedding(self, text: str, max_length: int = 500) -> List[float]:
        """
        Get embedding for text using OpenAI embeddings API.
        
        Args:
            text: Text to embed
            max_length: Maximum character length for text (default: 500)
            
        Returns:
            List of embedding values
        """
        if len(text) > max_length:
            text = text[:max_length]
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        
        Args:
            a: First embedding vector
            b: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    def _keyword_overlap_score(self, query: str, passage: str) -> float:
        """
        Fallback scoring method using keyword overlap.
        
        Computes Jaccard-like similarity between query and passage terms.
        
        Args:
            query: Query text
            passage: Passage text
            
        Returns:
            Keyword overlap score (0 to 1)
        """
        query_terms = set(query.lower().split())
        passage_terms = set(passage.lower().split())
        if not query_terms:
            return 0.0
        return len(query_terms & passage_terms) / len(query_terms)
    
    def cache_embeddings(self, batch_size: int = 100, max_chars: int = 500) -> None:
        """
        Pre-compute and cache embeddings for all passages.
        
        Improves retrieval efficiency by avoiding per-query embedding generation.
        
        Args:
            batch_size: Number of passages to embed per API call (default: 100)
            max_chars: Maximum character length per passage (default: 500, ~125 tokens)
        """
        if not self.passages:
            return
        
        self.embeddings = []
        for i in range(0, len(self.passages), batch_size):
            batch = []
            for p in self.passages[i:i+batch_size]:
                content = p['content']
                if len(content) > max_chars:
                    content = content[:max_chars]
                batch.append(content)
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                self.embeddings.extend([d.embedding for d in response.data])
            except Exception as e:
                logger.error(f"Error caching embeddings batch {i}: {e}")
                self.embeddings.extend([None] * len(batch))

    def save_to_disk(self, cache_path: str) -> None:
        """
        Save the RAG index (passages and embeddings) to disk.
        
        Args:
            cache_path: Directory path where the index will be saved.
        """
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        
        passages_path = os.path.join(cache_path, "passages.pkl")
        embeddings_path = os.path.join(cache_path, "embeddings.npy")
        metadata_path = os.path.join(cache_path, "metadata.json")
        
        with open(passages_path, "wb") as f:
            pickle.dump(self.passages, f)
        
        if self.embeddings is not None:
            np.save(embeddings_path, np.array(self.embeddings))
        
        metadata = {
            "embedding_model": self.embedding_model,
            "num_passages": len(self.passages),
            "num_embeddings": len(self.embeddings) if self.embeddings else 0
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        logger.info(f"RAG index saved to {cache_path}: {len(self.passages)} passages, {len(self.embeddings) if self.embeddings else 0} embeddings")

    @classmethod
    def load_from_disk(cls, cache_path: str) -> Optional['GroundTruthRAG']:
        """
        Load a RAG index from disk.
        
        Args:
            cache_path: Directory path where the index is stored.
            
        Returns:
            GroundTruthRAG instance with loaded index, or None if not found.
        """
        passages_path = os.path.join(cache_path, "passages.pkl")
        embeddings_path = os.path.join(cache_path, "embeddings.npy")
        metadata_path = os.path.join(cache_path, "metadata.json")
        
        if not os.path.exists(passages_path) or not os.path.exists(metadata_path):
            logger.debug(f"No cached RAG index found at {cache_path}")
            return None
        
        try:
            with open(passages_path, "rb") as f:
                passages = pickle.load(f)
            
            embeddings = None
            if os.path.exists(embeddings_path):
                embeddings = np.load(embeddings_path).tolist()
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            instance = cls(embedding_model=metadata.get("embedding_model", "text-embedding-3-small"))
            instance.passages = passages
            instance.embeddings = embeddings
            
            logger.info(f"RAG index loaded from {cache_path}: {len(passages)} passages, {len(embeddings) if embeddings else 0} embeddings")
            return instance
        except Exception as e:
            logger.error(f"Error loading RAG index from {cache_path}: {e}")
            return None

    @staticmethod
    def get_cache_path_for_entity(entities_cache_dir: str, entity_name: str) -> str:
        """
        Get the cache directory path for a specific entity.
        
        Args:
            entities_cache_dir: Base directory for entity caches.
            entity_name: Name of the entity.
            
        Returns:
            Full path to the entity's cache directory.
        """
        safe_name = entity_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return os.path.join(entities_cache_dir, safe_name)
