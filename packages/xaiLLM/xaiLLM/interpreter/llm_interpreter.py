"""
LLM interpretation orchestration.

Builds the prompt with instructions and evidence (visualisations and stats)
and calls the OpenAI Responses API to obtain a narrative interpretation.
"""

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
    
    
    def inference(self, probabilities: dict, influence_stats: tuple, xai_gradcam_enc: str, xai_shap_enc: str, input_image_enc: str) -> str:
        """
        Generate an interpretation of the model's prediction using an LLM.

        Parameters
        ----------
        probabilities : dict
            The prediction probabilities for the input image.
        influence_stats : tuple
            A tuple containing statistics about the influence functions.
        xai_gradcam_enc : str
            Base64-encoded Grad-CAM visualisation.
        xai_shap_enc : str
            Base64-encoded SHAP visualisation.
        input_image_enc : str
            Base64-encoded original input image.
        
        Returns
        -------
        str
            The LLM interpretation output.
        """
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
                                "text": str(probabilities)}, # prediction probabilities
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

