"""
Hybrid Retrieval combining dense and sparse search (Fully Compatible)
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict
from loguru import logger

from .vector_store import VectorStore
from ..processing.embeddings import MultimodalEmbedder


class HybridRetriever:
    """Hybrid retriever combining semantic and keyword search"""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedder: MultimodalEmbedder,
                 alpha: float = 0.5):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: Vector database instance
            embedder: Multimodal embedder
            alpha: Weight for dense retrieval (0-1)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = alpha
        self.bm25_index = None
        self.corpus = []
        self.corpus_metadata = []
        self.corpus_ids = []
    
    def index_corpus(self, texts: List[str], metadata: List[Dict[str, Any]], ids: List[str]):
        """Index corpus for BM25 search"""
        self.corpus = texts
        self.corpus_metadata = metadata
        self.corpus_ids = ids
        tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        logger.info(f"Indexed {len(texts)} documents for BM25")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25"""
        import re
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def retrieve(self, 
                 query: str,
                 filter_conditions: Optional[Dict] = None,
                 top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval with fallback mechanisms
        
        Args:
            query: User query
            filter_conditions: Metadata filters
            top_k: Number of results
        
        Returns:
            List of retrieved documents with scores
        """
        try:
            # Get dense search results
            query_embedding = self.embedder.embed_text(query)[0]
            
            try:
                dense_results = self.vector_store.search(
                    query_vector=query_embedding,
                    filter_conditions=filter_conditions,
                    top_k=top_k * 2
                )
            except Exception as e:
                logger.error(f"Dense search failed: {e}")
                dense_results = []
            
            # Get sparse (BM25) results
            sparse_results = self._bm25_search(query, top_k * 2)
            
            # Combine results
            combined = self._combine_results(dense_results, sparse_results, top_k)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            # Fallback: try dense only with different method
            try:
                query_embedding = self.embedder.embed_text(query)[0]
                return self.vector_store.search(
                    query_vector=query_embedding,
                    filter_conditions=filter_conditions,
                    top_k=top_k
                )
            except:
                # Ultimate fallback: return empty results
                return []
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform BM25 search"""
        if self.bm25_index is None or not self.corpus:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'id': self.corpus_ids[idx] if idx < len(self.corpus_ids) else str(idx),
                    'score': float(scores[idx]),
                    'payload': self.corpus_metadata[idx] if idx < len(self.corpus_metadata) else {},
                    'text': self.corpus[idx] if idx < len(self.corpus) else '',
                    'rank': len(results) + 1
                })
        
        return results
    
    def _combine_results(self, 
                         dense_results: List[Dict],
                         sparse_results: List[Dict],
                         top_k: int) -> List[Dict[str, Any]]:
        """Combine dense and sparse results"""
        if not dense_results and not sparse_results:
            return []
        
        if not dense_results:
            return sparse_results[:top_k]
        if not sparse_results:
            return dense_results[:top_k]
        
        # Normalize scores
        dense_scores = self._normalize_scores([r.get('score', 0) for r in dense_results])
        sparse_scores = self._normalize_scores([r.get('score', 0) for r in sparse_results])
        
        # Combine
        combined_scores = defaultdict(float)
        result_map = {}
        
        for idx, result in enumerate(dense_results):
            doc_id = result.get('id', str(idx))
            combined_scores[doc_id] += self.alpha * (dense_scores[idx] if idx < len(dense_scores) else 0)
            result_map[doc_id] = result
        
        for idx, result in enumerate(sparse_results):
            doc_id = result.get('id', str(idx))
            combined_scores[doc_id] += (1 - self.alpha) * (sparse_scores[idx] if idx < len(sparse_scores) else 0)
            if doc_id not in result_map:
                result_map[doc_id] = result
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        final_results = []
        for doc_id, score in sorted_results:
            result = result_map.get(doc_id, {})
            result['combined_score'] = score
            final_results.append(result)
        
        return final_results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1]"""
        if not scores:
            return []
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        if max_score == min_score:
            return [0.5] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]