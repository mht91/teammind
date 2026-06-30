"""
Simplified Image Processing using only transformers
"""
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from loguru import logger


class ImageProcessor:
    """Simple image processor using CLIP from transformers"""
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu"):
        """
        Initialize CLIP model
        
        Args:
            model_name: CLIP model variant
            device: 'cpu' or 'cuda'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info(f"Loading CLIP model: {model_name}")
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            logger.info("✅ CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            self.model = None
            self.processor = None
    
    def process_image(self, image_path: Path, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process image and extract features
        
        Args:
            image_path: Path to image file
            metadata: Additional metadata
        
        Returns:
            Dict containing image description and metadata
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            if self.model is None:
                return {
                    'description': "Image processor not available",
                    'features': [],
                    'metadata': metadata or {},
                    'source_file': str(image_path)
                }
            
            # Extract features
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
            
            return {
                'description': self._generate_description(image),
                'features': features.cpu().numpy().flatten().tolist(),
                'metadata': metadata or {},
                'source_file': str(image_path),
                'image_size': image.size,
                'format': image.format
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'description': f"Error: {str(e)}",
                'features': [],
                'error': str(e),
                'metadata': metadata or {},
                'source_file': str(image_path)
            }
    
    def _generate_description(self, image: Image.Image) -> str:
        """Generate a simple description (CLIP doesn't do captions)"""
        # This is a placeholder - CLIP doesn't generate captions
        return "Image processed with CLIP (captioning not available)"
    
    def extract_visual_features(self, image_path: Path) -> np.ndarray:
        """Extract visual features"""
        try:
            result = self.process_image(image_path)
            features = result.get('features', [])
            if features:
                return np.array(features)
            return np.zeros((512,))
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return np.zeros((512,))
        