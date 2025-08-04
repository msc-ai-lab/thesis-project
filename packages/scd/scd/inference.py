"""
Model Inference Module
=====================

This module provides inference capabilities for skin lesion classification models.
It handles model prediction, probability calculation, and result interpretation
for binary classification tasks (malignant vs benign).

The inference pipeline includes:
- Forward pass through the model with gradient computation disabled
- Logits extraction from model outputs (supporting both standard and transformer models)
- Probability calculation using softmax activation
- Class prediction based on argmax of logits
- Result formatting with class names and confidence scores

Functions:
---------
- predict(model, input_tensor): Perform inference on preprocessed input tensors

Dependencies:
------------
- torch: For PyTorch model operations and tensor handling
- torch.nn: For neural network modules
- torch.nn.functional: For softmax probability calculations

Usage:
-----
    from src.inference import predict
    
    # Perform inference on a preprocessed image
    (pred_idx, pred_name), probabilities = predict(model, input_tensor)
    print(f"Prediction: {pred_name} (confidence: {probabilities[pred_name]:.2f})")
"""

# Third-party imports
import torch
from torch import nn
from torch.nn.functional import softmax

# Function to predict using the model
def predict(model: nn.Module, input_tensor: torch.Tensor):
    """
    Predict the class of the input tensor using the model.
    Parameters
    ----------
        model : torch.nn.Module
            The model to use for prediction.
        input_tensor : torch.Tensor
            The input tensor for which to predict the class.
    Returns
    -------
        tuple
            A tuple containing the predicted class index and the class name.
        dict
            A dictionary with probabilities for each class.
    """
    try:
        with torch.no_grad():
            outputs, _ = model(input_tensor)
            
            # Get class prediction
            pred_idx = torch.argmax(outputs, dim=1).item()
            pred = "Benign" if pred_idx == 0 else "Malignant"
            
            # Get probability values
            probabilities = softmax(outputs, dim=1).squeeze()
            benign_prob = probabilities[0].item()
            malignant_prob = probabilities[1].item()

            return (pred_idx, pred), {"Benign": benign_prob, "Malignant": malignant_prob}
    except Exception as e:
        raise RuntimeError(f"Error during prediction")
