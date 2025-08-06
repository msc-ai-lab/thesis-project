from pathlib import Path
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from xaiLLM.utils.config import OPENAI_API_KEY
import importlib.resources
import pandas as pd
import numpy as np


class LLMInterpreter:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        with importlib.resources.open_text('xaiLLM.constants', 'llm_instructions.md', encoding='utf-8') as f:
            self.instructions = f.read()

    def influence_functions_stats(self, probs, influencers):
        
        # Establish the CNN-predicted class
        sample_probs = pd.read_csv(probs)
        predicted_class = str(sample_probs.loc[sample_probs['confidence'].idxmax(), 'class'])

        # Read in the influence functions data
        influence_data = pd.read_csv(influencers)

        # Filter for influential training cases that share ground truth with predicted class
        alligned_groundtruth = influence_data[influence_data['ground_truth'] == predicted_class]

        # Set default values
        groundtruth_alignment_percentage, groundtruth_misalignment_percentage, misclassified_percentage = 0, 100, None

        # Check for the count of alligned cases
        if len(alligned_groundtruth) > 0:
            # Calculate the percentage of influential training cases that share ground truth with predicted class
            groundtruth_alignment_percentage = (len(alligned_groundtruth) / len(influence_data['ground_truth'])) * 100

            # Calculate the percentage of the aligned cases that were misclassified during training
            misclassified_percentage = round((len(alligned_groundtruth[alligned_groundtruth["ground_truth"] != alligned_groundtruth["prediction"]]) / len(alligned_groundtruth)) * 100, 2)

        # Calculate the percentage of influential training cases whose ground truth does NOT match the predicted class
        groundtruth_misalignment_percentage = 100 - groundtruth_alignment_percentage

        return groundtruth_alignment_percentage, groundtruth_misalignment_percentage, misclassified_percentage
    
    
    def inference(self, probs, influence_stats, xai_gradcam_enc, xai_shap_enc, input_image_enc):
        try:
            response = self.client.responses.create(
                model="gpt-4.1-2025-04-14",
                input=[
                    {
                        "role": "developer",
                        "content" : self.instructions
                    },
                    {
                        "role": "user",
                        "content": [
                            { 
                                "type": "input_text",
                                "text": str(probs)}, # prediction probabilities
                            { 
                                "type": "input_text",
                                "text": str(influence_stats)}, # influence function statistics
                                
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{xai_gradcam_enc}", # GradCAM output, base64-encoded PNG file
                                "detail": "auto"
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{xai_shap_enc}", # SHAP output, base64-encoded PNG file
                                "detail": "auto"
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpg;base64,{input_image_enc}", # Original user sample, base64-encoded JPG file
                                "detail": "auto"
                            },

                        ],
                    }
                ],
                temperature=0.0
            )

            # Print the LLM interpretation
            return response.output_text

        # Handle potential errors
        except TimeoutError as e:
            raise Exception(f"The LLM API call encountered TimeoutError: {e}")

        except APIError as e:
            raise Exception(f"OpenAI API error: {e}")

        except APIConnectionError as e:
            raise Exception(f"Connection error: {e}")

        except RateLimitError as e:
            raise Exception(f"Rate limit exceeded: {e}")

        except Exception as e:
            raise Exception(f"Unexpected error in LLM API call: {e}")

