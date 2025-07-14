"""
Model Loading and Management Module
==================================

This module provides utilities for loading and managing pre-trained deep learning models
for skin lesion classification. It handles model initialisation, state dictionary loading,
and preparation for inference.

The module supports:
- Loading pre-trained Xception models from saved state dictionaries
- Model configuration for binary classification (malignant vs benign)
- Error handling for model loading
- CPU/GPU device mapping for flexible deployment

Functions:
---------
- load_model(model_path, num_classes): Load a pre-trained model from file

Dependencies:
------------
- timm: For creating pre-trained model architectures
- torch: For PyTorch model operations and state dictionary handling

Usage:
-----
    from src.model import load_model
    
    # Load a pre-trained model
    model = load_model('path/to/model.pth', num_classes=2)
"""

import timm
import torch

# Load the model
def load_model(model_path, num_classes):
    """
    Load the pre-trained model from the specified path.
    Parameters
    ----------
        model_path : str
            Path to the pre-trained model file.
        num_classes : int
            Number of output classes for the model.
    Returns
    -------
        torch.nn.Module
            The loaded model ready for inference.
    Raises
    ------
        FileNotFoundError : If the model file does not exist.
        RuntimeError : If there is an error loading the model.
    """
    model = timm.create_model('legacy_xception', pretrained=False, num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")