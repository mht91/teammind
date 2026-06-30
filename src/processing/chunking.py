"""
Intelligent Chunking Module using semantic segmentation
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger


@dataclass
class SemanticChunk:
    """Represents a semantically meaningful chunk"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    type: str = "text"  # text, image, meeting, slide


class SemanticChunker:
    """Intelligent chunking based on semantic similarity"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", 
                 max_chunk_size: int = 512,
                 min_chunk_size: int = 100,
                 similarity_threshold: float = 0.7):
        """
        Initialize semantic chunker
        
        Args:
            model_name: Sentence transformer model
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk
            similarity_threshold: Threshold for splitting chunks
        """
        self.model = SentenceTransformer(model_name)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"Loaded chunking model: {model_name}")
    
    def chunk_meeting_transcript(self, segments: List[Dict[str, Any]], 
                                 metadata: Dict) -> List[SemanticChunk]:
        """
        Chunk meeting transcript by semantic topics
        
        Args:
            segments: List of meeting segments with speaker, text, timestamps
            metadata: Meeting metadata
        
        Returns:
            List of semantic chunks
        """
        chunks = []
        current_chunk = []
        current_text = []
        current_embedding = None
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            if not text:
                continue
            
            # Get embedding for segment
            seg_embedding = self.model.encode([text])[0]
            
            # Check if this segment should start a new chunk
            if current_embedding is not None:
                similarity = self._cosine_similarity(current_embedding, seg_embedding)
                
                # If similarity is low, start new chunk
                if similarity < self.similarity_threshold or len(' '.join(current_text)) > self.max_chunk_size:
                    if current_text:
                        chunk = self._create_chunk(current_text, current_chunk, metadata)
                        chunks.append(chunk)
                        current_text = []
                        current_chunk = []
            
            current_text.append(text)
            current_chunk.append(segment)
            current_embedding = seg_embedding
        
        # Add final chunk
        if current_text:
            chunk = self._create_chunk(current_text, current_chunk, metadata)
            chunks.append(chunk)
        
        return chunks
    
    def chunk_document(self, content: str, metadata: Dict) -> List[SemanticChunk]:
        """
        Chunk document content by semantic paragraphs
        
        Args:
            content: Document text
            metadata: Document metadata
        
        Returns:
            List of semantic chunks
        """
        # Split by paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_text = []
        current_embedding = None
        
        for para in paragraphs:
            para_embedding = self.model.encode([para])[0]
            
            if current_embedding is not None:
                similarity = self._cosine_similarity(current_embedding, para_embedding)
                
                # Check if we should start a new chunk
                if similarity < self.similarity_threshold or len(' '.join(current_text)) > self.max_chunk_size:
                    if current_text:
                        chunk = self._create_chunk_from_text(current_text, metadata)
                        chunks.append(chunk)
                        current_text = []
            
            current_text.append(para)
            current_embedding = para_embedding
        
        # Add final chunk
        if current_text:
            chunk = self._create_chunk_from_text(current_text, metadata)
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, segments: List[Dict], segment_data: List[Dict], 
                     metadata: Dict) -> SemanticChunk:
        """Create a semantic chunk from meeting segments"""
        text = ' '.join([s.get('text', '') for s in segments])
        
        chunk_metadata = metadata.copy()
        chunk_metadata['start_time'] = segment_data[0].get('start', 0)
        chunk_metadata['end_time'] = segment_data[-1].get('end', 0)
        chunk_metadata['speakers'] = list(set([s.get('speaker', 'Unknown') for s in segment_data]))
        chunk_metadata['segment_count'] = len(segments)
        
        embedding = self.model.encode([text])[0]
        
        return SemanticChunk(
            content=text,
            metadata=chunk_metadata,
            embedding=embedding,
            type="meeting"
        )
    
    def _create_chunk_from_text(self, texts: List[str], metadata: Dict) -> SemanticChunk:
        """Create a semantic chunk from document text"""
        content = '\n'.join(texts)
        embedding = self.model.encode([content])[0]
        
        return SemanticChunk(
            content=content,
            metadata=metadata.copy(),
            embedding=embedding,
            type="document"
        )
    
    @staticmethod
    def _cosine_similarity(a, b):
        """Compute cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))