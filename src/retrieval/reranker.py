"""
Re-ranking Module using BGE Reranker (Open Source)
"""
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from loguru import logger


class Reranker:
    """Re-rank retrieved documents using cross-encoder"""
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-reranker-large",
                 device: str = "cpu"):
        """
        Initialize reranker
        
        Args:
            model_name: Cross-encoder model name
            device: 'cpu' or 'cuda'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            logger.info(f"Loaded reranker: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}. Using fallback ranking.")
            self.model = None
            self.tokenizer = None
    
    def rerank(self,
               query: str,
               documents: List[Dict[str, Any]],
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on relevance to query
        
        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to keep
        
        Returns:
            Re-ranked documents
        """
        if not documents:
            return []
        
        if self.model is None:
            # Fallback: return top-k based on original score
            sorted_docs = sorted(documents, key=lambda x: x.get('score', 0), reverse=True)
            return sorted_docs[:top_k]
        
        # Prepare pairs for cross-encoder
        pairs = []
        for doc in documents:
            doc_text = self._extract_document_text(doc)
            pairs.append((query, doc_text))
        
        # Compute relevance scores
        scores = self._compute_scores(pairs)
        
        # Add reranking scores
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by reranking score
        sorted_docs = sorted(documents, key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return sorted_docs[:top_k]
    
    def _compute_scores(self, pairs: List[tuple]) -> List[float]:
        """Compute relevance scores using cross-encoder"""
        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            scores = self.model(**features).logits.squeeze(-1)
        
        return scores.cpu().tolist()
    
    def _extract_document_text(self, doc: Dict[str, Any]) -> str:
        """Extract text from document"""
        payload = doc.get('payload', {})
        
        # Check for various text fields
        if 'text' in payload:
            return payload['text']
        elif 'content' in payload:
            return payload['content']
        elif 'description' in payload:
            return payload['description']
        else:
            # Try to concatenate all string values
            text_parts = []
            for key, value in payload.items():
                if isinstance(value, str):
                    text_parts.append(value)
            return ' '.join(text_parts) if text_parts else ''