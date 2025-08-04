import os
import torch
import pandas as pd
from pathlib import Path

from scd.preprocess import preprocess_input
from scd.inference import predict
from scd.utils.common import load_model
from xaiLLM.run import run_xaiLLM
from xaiLLM.utils.helpers import show_title
from xaiLLM.interpreter.parser import Parser

def run_quantitative_analysis():
    results = []

    try:        
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define paths
        ROOT_DIR = Path.cwd()
        DATASET_PATH = ROOT_DIR / 'data' / 'processed' / 'train_dataset.pt'
        MODEL_PATH = ROOT_DIR / 'models' / 'ResNet_skin_cancer_classification.pth'
        IMAGE_RESIZE = (384, 384)
        TEST_DATASET_PATH = ROOT_DIR / 'data' / 'quantitative_analysis_images'

        # Ensure the test dataset directory exists
        if not os.path.exists(TEST_DATASET_PATH):
            os.makedirs(TEST_DATASET_PATH)
        
        # Load the model
        model = load_model(MODEL_PATH)

        # Load images
        images = []
        for image_file in TEST_DATASET_PATH.glob('*.jpg'):
            images.append(image_file)
        
        if len(images) == 0:
            raise ValueError("No images found in 'data/quantitative_analysis_images' directory.")

        # Run inference on each image
        for input_path in images:
            # Preprocess the input image
            image_tensor = preprocess_input(input_path, resize=IMAGE_RESIZE).to(device)

            # Predict the class and get probabilities
            (pred_idx, output), probs = predict(model, image_tensor)
            show_title('Running Inference...')
            print(f"Inference result: {output}")
            print(f"Probabilities:")
            print(f"\tBenign: {probs['Benign'] * 100:.2f}%")
            print(f"\tMalignant: {probs['Malignant'] * 100:.2f}%")

            # Run xaiLLM for explanation
            _, _, influencers, llm_output = run_xaiLLM(
                model,
                image_tensor,
                input_path,
                pred_idx,
                dataset_path=DATASET_PATH,
                probabilities=probs,
                show_images=False
            )

            # Calculate model influential percentage where model predictions match ground truth
            model_influential_percentage = (sum(influencers['ground_truth'] == output) / len(influencers['ground_truth'])) * 100

            # Parse the LLM output
            parser = Parser()
            parsed_output = parser.parse(llm_output)

            result = {
                'model_prediction': output,
                'model_confidence': round(probs[output] * 100, 2),
                'model_influential_percentage': model_influential_percentage,
                'llm_parsed_prediction': parsed_output['prediction'],
                'llm_parsed_borderline': parsed_output['borderline'],
                'llm_parsed_confidence': parsed_output['confidence'],
                'llm_parsed_influential_percentage': parsed_output['influential_cases_percentage']
            }
            print(result)

            # Store results
            results.append(result)

    except Exception as e:
        print(f"An error occurred during inference: {e}")

    finally:
        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv('results/quantitative_analysis.csv')

if __name__ == "__main__":
    run_quantitative_analysis()