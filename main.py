"""
Main Application Module
======================

This module serves as the main entry point for the skin lesion classification application.
It orchestrates the complete pipeline from model loading to inference and explainability
for skin lesion analysis.

The application workflow includes:
- Device detection and model loading with GPU/CPU support
- Interactive user input for image path specification
- Input validation and error handling
- Image preprocessing and tensor preparation
- Model inference with probability calculation
- Grad-CAM visualisation generation for model interpretability
- Comprehensive error handling

Functions:
---------
- main(): Main application entry point that orchestrates the complete pipeline

Dependencies:
------------
- torch: For PyTorch device handling and tensor operations
- model: For pre-trained model loading functionality
- preprocess: For image preprocessing and transformation
- inference: For model prediction and probability calculation
- explainer: For Grad-CAM visualisation generation
- pathlib: For robust file path handling

Usage:
-----
    python main.py
    
    # The application will prompt for an image path and provide:
    # - Classification result (malignant/benign)
    # - Prediction probabilities
    # - Grad-CAM visualisation overlay
"""


import torch
from pathlib import Path

from scd.preprocess import preprocess_input
from scd.inference import predict
from scd.utils.common import load_model, load_datasets
from xaiLLM.explainer import grad_cam, shap, calculate_influence

def main():
    try:
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define paths
        ROOT_DIR = Path.cwd()
        DATASET_PATH = ROOT_DIR / 'data' / 'processed'
        MODEL_PATH = ROOT_DIR / 'models' / 'ResNet_skin_cancer_classification.pth'
        IMAGE_RESIZE = (384, 384)

        # Load the model
        model = load_model(MODEL_PATH)

        # Ask user for image path
        image_path = input("Please enter the path to your image: ")
        input_path = Path(image_path)

        # Validate the path
        if not input_path.exists():
            print(f"Error: The file {input_path} does not exist.")
            exit(1)

        # Preprocess the input image
        image_tensor = preprocess_input(input_path, resize=IMAGE_RESIZE).to(device)

        # Predict the class and get probabilities
        (pred_idx, output), probs = predict(model, image_tensor)
        print(f"Inference result: {output}")
        print(f"Probabilities: {probs}")

        # Generate Grad-CAM visualisation
        print('Generating Grad-CAM visualisation...')
        gradcam_viz = grad_cam(model, image_tensor, input_path, predicted_class_index=pred_idx)

        # Generate SHAP visualisation
        print('Generating SHAP visualisation...')
        shap_viz = shap(model, image_tensor, input_path, predicted_class_index=pred_idx)

        # Influence Function
        print('Calculating influence...')
        dataset, filenames = load_datasets(DATASET_PATH, only_train_dataset_with_filenames=True)
        influencers = calculate_influence(model, image_tensor, pred_idx, dataset, filenames)
        print(influencers.head(5))
    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()