"""
Multimodal Embedding Generation
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import clip
from PIL import Image
from loguru import logger


class MultimodalEmbedder:
    """Generate embeddings for different modalities"""
    
    def __init__(self, 
                 text_model: str = "BAAI/bge-large-en-v1.5",
                 image_model: str = "ViT-B/32",
                 device: str = "cpu"):
        """
        Initialize multimodal embedder
        
        Args:
            text_model: Sentence transformer model for text
            image_model: CLIP model for images
            device: 'cpu' or 'cuda'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Text embeddings
        self.text_model = SentenceTransformer(text_model, device=self.device)
        logger.info(f"Loaded text model: {text_model}")
        
        # Image embeddings (CLIP)
        try:
            self.image_model, self.image_preprocess = clip.load(image_model, device=self.device)
            logger.info(f"Loaded image model: {image_model}")
        except Exception as e:
            logger.warning(f"Could not load CLIP: {e}. Image embeddings will use text model fallback.")
            self.image_model = None
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate text embeddings
        
        Args:
            text: Text string or list of strings
        
        Returns:
            Embedding array
        """
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.text_model.encode(text, normalize_embeddings=True)
        return embeddings
    
    def embed_image(self, image_path: Path) -> np.ndarray:
        """
        Generate image embeddings using CLIP
        
        Args:
            image_path: Path to image file
        
        Returns:
            Embedding array
        """
        if self.image_model is None:
            # Fallback: use text description if available
            return np.zeros((512,))
        
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.image_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.image_model.encode_image(image_input)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {e}")
            return np.zeros((512,))
    
    def embed_meeting(self, text: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Embed meeting transcript with context
        
        Args:
            text: Meeting text
            metadata: Meeting metadata
        
        Returns:
            Embedding array
        """
        # Combine text with relevant metadata
        context_parts = [text]
        
        if metadata:
            if 'speaker' in metadata:
                context_parts.append(f"Speaker: {metadata['speaker']}")
            if 'department' in metadata:
                context_parts.append(f"Department: {metadata['department']}")
            if 'topic' in metadata:
                context_parts.append(f"Topic: {metadata['topic']}")
        
        context = ' | '.join(context_parts)
        embedding = self.text_model.encode([context], normalize_embeddings=True)
        return embedding[0]
    
    def embed_slide(self, text: str, image_path: Optional[Path] = None) -> np.ndarray:
        """
        Embed presentation slide combining text and image
        
        Args:
            text: Slide text
            image_path: Optional path to slide image
        
        Returns:
            Combined embedding
        """
        # Get text embedding
        text_embedding = self.embed_text(text)[0]
        
        # Get image embedding if available
        if image_path and self.image_model:
            image_embedding = self.embed_image(image_path)
            # Combine text and image embeddings
            combined = np.concatenate([text_embedding, image_embedding])
            # Normalize
            combined = combined / np.linalg.norm(combined)
            return combined
        
        return text_embedding