"""
CLIP Encoder - Layer 2: Representation

Encapsulates CLIP model for extracting semantically aligned visual embeddings
from hand gesture images.

Architecture Role:
- Transforms perception outputs (hand crops) into semantic representations
- Provides 512-dim embeddings in CLIP's joint vision-language space
- Enables zero-shot semantic matching via text alignment

Design Principles:
- Centralized model lifecycle management
- Clean input/output contracts (NumPy arrays, not raw tensors)
- Deterministic configuration
- No ML framework leakage to higher layers
"""

import logging
from typing import Optional, Tuple
import numpy as np
import torch
from PIL import Image
import open_clip

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """
    CLIP-based visual encoder for gesture semantics.
    
    Transforms hand gesture images into 512-dimensional embeddings
    aligned with CLIP's text embedding space, enabling semantic
    similarity matching.
    
    Architecture Boundary:
    - Input: NumPy array (H, W, 3) RGB uint8 image
    - Output: NumPy array (512,) float32 L2-normalized embedding
    
    Example:
        >>> encoder = CLIPEncoder(model_name='ViT-B-32', device='cpu')
        >>> hand_image = np.zeros((224, 224, 3), dtype=np.uint8)
        >>> embedding = encoder.encode_image(hand_image)
        >>> embedding.shape
        (512,)
        >>> np.isclose(np.linalg.norm(embedding), 1.0)
        True
    """
    
    def __init__(
        self,
        model_name: str = 'ViT-B-32',
        pretrained: str = 'openai',
        device: Optional[str] = None
    ):
        """
        Initialize CLIP encoder with deterministic configuration.
        
        Args:
            model_name: CLIP model architecture
                - 'ViT-B-32': Base model, 512-dim, ~30ms inference (default)
                - 'ViT-L-14': Large model, 768-dim, ~80ms inference
            pretrained: Pretrained weights source
                - 'openai': Official OpenAI CLIP weights (default)
            device: Compute device ('cpu', 'cuda', 'mps', or None for auto)
        
        Raises:
            RuntimeError: If model loading fails
            ValueError: If model_name is not supported
        """
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        
        logger.info(
            f"Initializing CLIPEncoder: model={model_name}, "
            f"pretrained={pretrained}, device={device}"
        )
        
        try:
            # Load CLIP model and preprocessing pipeline
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
                device=device
            )
            self.model.eval()  # Set to evaluation mode
            
            # Determine embedding dimension from model
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224, device=device)
                dummy_output = self.model.encode_image(dummy_input)
                self.embedding_dim = dummy_output.shape[1]
            
            logger.info(
                f"CLIPEncoder initialized successfully: "
                f"embedding_dim={self.embedding_dim}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize CLIPEncoder: {e}")
            raise RuntimeError(
                f"CLIP model initialization failed for {model_name} "
                f"with {pretrained} weights: {e}"
            ) from e
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode hand gesture image to semantic embedding.
        
        Transforms RGB image into CLIP's joint vision-language embedding
        space, enabling semantic similarity matching with text descriptions.
        
        Args:
            image: RGB image as NumPy array with shape:
                - (H, W, 3) for single image
                - Values: uint8 [0, 255] or float32 [0.0, 1.0]
        
        Returns:
            L2-normalized embedding vector as NumPy array:
                - Shape: (embedding_dim,) typically (512,) for ViT-B/32
                - Dtype: float32
                - Normalized: ||embedding|| = 1.0
        
        Raises:
            ValueError: If image has invalid shape or dtype
            RuntimeError: If encoding fails
        
        Example:
            >>> encoder = CLIPEncoder()
            >>> image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            >>> embedding = encoder.encode_image(image)
            >>> assert embedding.shape == (512,)
            >>> assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-5
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Expected NumPy array, got {type(image).__name__}"
            )
        
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) RGB image, got shape {image.shape}"
            )
        
        # Convert to PIL Image for preprocessing
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image, mode='RGB')
        elif image.dtype in (np.float32, np.float64):
            # Assume [0, 1] range
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_uint8, mode='RGB')
        else:
            raise ValueError(
                f"Unsupported dtype: {image.dtype}. "
                f"Expected uint8 or float32/float64"
            )
        
        try:
            # Preprocess and encode
            with torch.no_grad():
                # Apply CLIP preprocessing (resize, normalize, etc.)
                image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                
                # Extract visual embedding
                embedding_tensor = self.model.encode_image(image_tensor)
                
                # L2 normalize (CLIP embeddings should be normalized for cosine similarity)
                embedding_tensor = embedding_tensor / embedding_tensor.norm(dim=-1, keepdim=True)
                
                # Convert to NumPy
                embedding = embedding_tensor.cpu().numpy().squeeze()
            
            # Final validation
            if embedding.shape != (self.embedding_dim,):
                raise RuntimeError(f"Unexpected embedding shape: {embedding.shape}")
            if not np.isfinite(embedding).all():
                raise RuntimeError("Embedding contains NaN or Inf values")
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            raise RuntimeError(f"Failed to encode image: {e}") from e
    
    def encode_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Encode batch of hand gesture images to semantic embeddings.
        
        More efficient than calling encode_image() repeatedly.
        
        Args:
            images: Batch of RGB images as NumPy array:
                - Shape: (N, H, W, 3)
                - Values: uint8 [0, 255] or float32 [0.0, 1.0]
        
        Returns:
            L2-normalized embedding matrix:
                - Shape: (N, embedding_dim)
                - Dtype: float32
                - Each row normalized: ||embedding[i]|| = 1.0
        
        Raises:
            ValueError: If images have invalid shape or dtype
            RuntimeError: If encoding fails
        
        Example:
            >>> encoder = CLIPEncoder()
            >>> images = np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8)
            >>> embeddings = encoder.encode_batch(images)
            >>> assert embeddings.shape == (5, 512)
        """
        # Validate input
        if not isinstance(images, np.ndarray):
            raise ValueError(
                f"Expected NumPy array, got {type(images).__name__}"
            )
        
        if images.ndim != 4 or images.shape[3] != 3:
            raise ValueError(
                f"Expected (N, H, W, 3) batch of RGB images, got shape {images.shape}"
            )
        
        batch_size = images.shape[0]
        
        try:
            # Convert batch to PIL images and preprocess
            pil_images = []
            for i in range(batch_size):
                image = images[i]
                
                if image.dtype == np.uint8:
                    pil_image = Image.fromarray(image, mode='RGB')
                elif image.dtype in (np.float32, np.float64):
                    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                    pil_image = Image.fromarray(image_uint8, mode='RGB')
                else:
                    raise ValueError(
                        f"Unsupported dtype: {image.dtype}. "
                        f"Expected uint8 or float32/float64"
                    )
                
                pil_images.append(pil_image)
            
            # Batch encode
            with torch.no_grad():
                # Stack preprocessed images
                image_tensors = torch.stack([
                    self.preprocess(img) for img in pil_images
                ]).to(self.device)
                
                # Extract visual embeddings
                embedding_tensors = self.model.encode_image(image_tensors)
                
                # L2 normalize
                embedding_tensors = embedding_tensors / embedding_tensors.norm(dim=-1, keepdim=True)
                
                # Convert to NumPy
                embeddings = embedding_tensors.cpu().numpy()
            
            # Final validation
            assert embeddings.shape == (batch_size, self.embedding_dim), \
                f"Unexpected embeddings shape: {embeddings.shape}"
            assert np.isfinite(embeddings).all(), \
                "Embeddings contain NaN or Inf values"
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            raise RuntimeError(f"Failed to encode batch: {e}") from e
    
    def get_embedding_dim(self) -> int:
        """
        Get embedding dimension of the encoder.
        
        Returns:
            Embedding dimension (e.g., 512 for ViT-B/32, 768 for ViT-L/14)
        """
        return self.embedding_dim
    
    def get_config(self) -> dict:
        """
        Get encoder configuration for serialization/logging.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'model_name': self.model_name,
            'pretrained': self.pretrained,
            'device': self.device,
            'embedding_dim': self.embedding_dim
        }
    
    def __repr__(self) -> str:
        return (
            f"CLIPEncoder(model_name='{self.model_name}', "
            f"pretrained='{self.pretrained}', device='{self.device}', "
            f"embedding_dim={self.embedding_dim})"
        )
