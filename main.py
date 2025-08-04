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
from scd.utils.common import load_model
from xaiLLM.run import run_xaiLLM
from xaiLLM.utils.helpers import show_title, show_image

def main():
    try:
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define paths
        ROOT_DIR = Path.cwd()
        DATASET_PATH = ROOT_DIR / 'data' / 'processed' / 'train_dataset.pt'
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
        show_title('Running Inference...')
        print(f"Inference result: {output}")
        print(f"Probabilities:")
        print(f"\tBenign: {probs['Benign'] * 100:.2f}%")
        print(f"\tMalignant: {probs['Malignant'] * 100:.2f}%")


        gradcam_viz, shap_viz, influencers, llm_output = run_xaiLLM(
            model,
            image_tensor,
            input_path,
            pred_idx,
            dataset_path=DATASET_PATH,
            probabilities=probs,
        )

        # Show top 5 influencers
        print("\nTop 5 Influencers:")
        influencers.to_csv(ROOT_DIR / 'results' / 'influencers.csv', index=False)
        print(influencers.head(5))

        # Print LLM interpretation
        print("\nLLM Interpretation:")
        print(llm_output)
    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()