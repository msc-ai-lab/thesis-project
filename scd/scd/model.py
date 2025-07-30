"""
Neural Network Models for Skin Cancer Detection
===============================================

This module implements neural network architectures for binary skin cancer classification
(malignant vs benign). It includes a custom CNN with attention mechanisms and Low-Rank
Adaptation (LoRA) for efficient fine-tuning on medical imaging data.

Classes:
-------
- LoRALayer: Low-Rank Adaptation layer for efficient parameter adaptation
- SkinCancerCNN: Complete CNN architecture with ResNet backbone, attention, and LoRA

The SkinCancerCNN model combines:
- Pre-trained ResNet34 backbone for robust feature extraction
- Attention mechanism for interpretability and explainability
- LoRA layers for parameter-efficient fine-tuning
- Binary classification head for malignant/benign prediction

Dependencies:
------------
- torch: PyTorch framework for neural network operations
- torch.nn: Neural network modules and layers
- torchvision.models: Pre-trained model architectures
"""

import torch
from torch import nn
from torchvision import models

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) Layer for Parameter-Efficient Fine-tuning.
    
    LoRA decomposes weight updates into low-rank matrices, significantly reducing
    the number of trainable parameters while maintaining model performance. This
    is particularly useful for fine-tuning large pre-trained models on specific
    domains like medical imaging.
    
    The layer implements: output = x @ (W + A @ B)
    where W is the frozen base weight, and A, B are trainable low-rank matrices.
    
    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int 
        Number of output features
    rank : int, default=4
        Rank of the low-rank decomposition. Lower rank means fewer parameters
        but potentially reduced expressiveness.
        
    Attributes
    ----------
    kernel : nn.Parameter
        Base weight matrix (frozen during LoRA training)
    lora_A : nn.Parameter
        First low-rank matrix (in_features x rank)
    lora_B : nn.Parameter
        Second low-rank matrix (rank x out_features)
    """
    
    def __init__(self, in_features, out_features, rank=4):
        super(LoRALayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Base weight
        self.kernel = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.kernel)

        # LoRA weights
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, out_features) * 0.01)

    def forward(self, x):
        """
        Forward pass through the LoRA layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features)
            Combined result of base transformation and low-rank adaptation
        """
        base_output = x @ self.kernel
        lora_output = x @ self.lora_A @ self.lora_B
        return base_output + lora_output

class SkinCancerCNN(nn.Module):
    """
    Complete CNN Architecture for Skin Cancer Binary Classification.
    
    This model combines a pre-trained ResNet34 backbone with attention mechanisms
    and Low-Rank Adaptation (LoRA) for efficient and interpretable skin lesion
    classification. The architecture is designed for binary classification of
    skin lesions as malignant or benign.
    
    Parameters
    ----------
    num_classes : int, default=2
        Number of output classes (2 for binary classification: benign/malignant)
        
    Attributes
    ----------
    features : nn.Sequential
        ResNet34 feature extraction layers (conv layers without final pooling/fc)
    attention : nn.Sequential
        Spatial attention module for generating attention maps
    pool : nn.AdaptiveAvgPool2d
        Global average pooling layer
    lora_dense : LoRALayer
        Low-rank adaptation layer for efficient fine-tuning
    classifier : nn.Sequential
        Final classification layers with regularisation
    """
    
    def __init__(self, num_classes=2):
        super(SkinCancerCNN, self).__init__()

        # Feature Extractor: Pretrained ResNet backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Attention Module for Explainability
        self.attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        # Global Pooling + Fully Connected Layers
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lora_dense = LoRALayer(512, 512)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the complete skin cancer classification network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input batch of skin lesion images
            Shape: (batch_size, 3, height, width)
            
        Returns
        -------
        tuple of torch.Tensor
            x : Classification logits of shape (batch_size, num_classes)
                Raw scores for each class (before softmax)
            attn_map : Attention map of shape (batch_size, 1, H, W)
                Spatial attention weights highlighting important regions
                Values are sigmoid-activated (range 0-1)
                
        Notes
        -----
        The attention map can be used for:
        - Model interpretability and explainability
        - Visualisation of regions the model focuses on
        - Integration with Grad-CAM or other XAI techniques
        - Medical professional validation of model attention
        """
        x = self.features(x)
        # Apply attention mask
        attn_map = self.attention(x)
        attn_map = torch.sigmoid(attn_map)
        x = x * attn_map
        x = self.pool(x)
        x = x.flatten(1)  # Flatten for LoRA layer
        x = self.lora_dense(x)
        x = self.classifier(x)
        return x, attn_map