"""
Explainable AI Module
====================

This module provides explainable AI capabilities for skin lesion classification models
using Grad-CAM (Gradient-weighted Class Activation Mapping) visualisation techniques.
It helps interpret model decisions by highlighting important regions in input images.

The explainability pipeline includes:
- Grad-CAM attribution computation using Captum library
- Layer-specific gradient analysis (targeting model.act4 layer)
- Heatmap generation and interpolation to match input image size
- Visualisation overlay combining original image with activation heatmap
- Transparent overlay rendering for clear interpretation

Functions:
---------
- grad_cam(model, input_tensor, input_image, predicted_class_index): Generate Grad-CAM visualisation

Dependencies:
------------
- torch: For PyTorch model operations and tensor handling
- torch.nn: For neural network modules
- torch.nn.functional: For interpolation and tensor operations
- captum.attr: For LayerGradCam attribution computation
- matplotlib.pyplot: For visualisation and plotting

Usage:
-----
    from src.explainer import grad_cam
    
    # Generate Grad-CAM visualisation
    grad_cam(model, input_tensor, original_image, predicted_class_idx)
"""

import torch
from torch import nn
from torch.nn import functional as F
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt

def grad_cam(model: nn.Module, input_tensor: torch.Tensor, input_image: torch.Tensor, predicted_class_index: int):
  grad_cam_layer = model.vit.embeddings.patch_embeddings.projection
  layer_gc = LayerGradCam(model, grad_cam_layer)
  attribution_gc = layer_gc.attribute(input_tensor, target=predicted_class_index)

  # To save the clean heatmap overlay as requested, we'll build it manually.
  heatmap = F.interpolate(attribution_gc, size=input_image.size, mode='bilinear', align_corners=False)
  heatmap = heatmap.squeeze().cpu().detach().numpy()

  # Plotting and saving the Grad-CAM image
  _, ax = plt.subplots(figsize=(8, 8)) # Create a figure with a specific size
  ax.imshow(input_image)
  ax.imshow(heatmap, cmap='jet', alpha=0.5) # Overlay heatmap with transparency
  ax.axis('off')

  plt.show()
