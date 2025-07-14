"""
Utility Functions for the Thesis Project
========================================

This module provides utility functions for various tasks in the project, including:

- Dataset loading and preprocessing (load_datasets)
- Data transformations for testing (get_test_transforms)
- Model initialisation and configuration (get_model)

Available Functions:
------------------
- load_datasets(path): Load train, validation, and test datasets from saved PyTorch tensors
- get_test_transforms(size): Get standardised testing transformations with resize and normalisation
- get_model(model_name, num_classes): Initialise pre-trained models (Xception or ViT) for binary classification

These utilities serve as the foundation for model implementation, training, evaluation, and interpretability analysis throughout the project.
"""