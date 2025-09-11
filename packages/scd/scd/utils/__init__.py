"""
Utility Functions for the Thesis Project
========================================

This module provides utility functions for various tasks in the project, including:

- Dataset loading and preprocessing (load_datasets)
- Data transformations for testing (get_test_transforms)
- Model initialisation and configuration (load_model)

Available Functions:
------------------
- load_datasets(path): Load train, validation, and test datasets from saved PyTorch tensors
- get_test_transforms(resize): Get standardised testing transformations with resize and normalisation
- load_model(model_path): Initialises SkinCancerCNN model with pre-trained weights from a specified path

These utilities serve as the foundation for model implementation, training, evaluation, and interpretability analysis throughout the project.
"""