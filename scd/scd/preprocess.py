"""
Image Preprocessing Module
=========================

This module provides preprocessing utilities for skin lesion images in this project.
It handles image loading, transformation, and preparation for model inference.

The preprocessing pipeline includes:
- Image loading from file paths
- RGB conversion for consistent color channels
- Standardised transformations (resize, normalisation)
- Tensor conversion with batch dimension addition
- Error handling for image processing

Functions:
---------
- preprocess_input(img_path): Preprocess input images for model inference

Dependencies:
------------
- numpy: For array operations
- PIL (Pillow): For image loading and conversion
- utils.common: For standardised test transformations

Usage:
-----
    from src.preprocess import preprocess_input
    
    # Preprocess an image for inference
    processed_tensor = preprocess_input('path/to/image.jpg')
"""

import numpy as np
from PIL import Image
import torch
from scd.utils.common import get_test_transforms

# Function to preprocess input data
def preprocess_input(img_path: str, resize: tuple = (224, 224)) -> torch.Tensor:
    """
    Preprocess the input image for inference.
    Parameters
    ----------
        img_path : str or Path
            Path to the input image file.
        resize : tuple, optional
            Desired size to resize the image to (default is (224, 224)).
    Returns
    -------
        torch.Tensor
            The preprocessed image tensor ready for inference.
    """
    transformations = get_test_transforms(resize=resize)

    try:
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        processed_image = transformations(image=image_np)['image']

        return processed_image.unsqueeze(0)  # Add batch dimension
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {img_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")