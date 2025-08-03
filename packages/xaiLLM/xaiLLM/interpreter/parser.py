from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from pydantic import BaseModel, Field
from typing import Literal, Annotated
from xaiLLM.utils.config import OPENAI_API_KEY

class TextFormat(BaseModel):
    prediction: Literal['Benign', 'Malignant'] = Field(
        description="'Benign' for when the AI analysis suggests low or moderately low concern for malignancy; 'Malignant' for when the AI analysis indicates high or moderately high concern for malignancy."
        )
    borderline: Literal[True, False] = Field(
        description="True if the model considers this to be a borderline case, False otherwise. Must be exactly True or False."
    )
    confidence: Annotated[float, Field(
        description="Model confidence, as indicated in Confidence Level section, to 2 decimal places."
        )]
    influential_cases_percentage: Annotated[float, Field(
        description="Influence Functions: What percentage of the most influential training cases share the same ground truth label as the predicted class. Output float with 2 decimal places."
    )]


class Parser(BaseModel):
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.text_format = TextFormat()

    def parse(self, response: str) -> dict:
        """
        Parse the LLM response to extract structured data.

        Parameters
        ----------
        response : str
            The unstructured text response from the LLM.

        Returns
        -------
        dict
            A dictionary containing the parsed structured data according to the TextFormat model.
        """
        try:
            extractor = self.client.responses.parse(
                model="gpt-4o-2024-08-06",
                input=[
                    {
                        "role": "system",
                        "content": "You are an expert at structured data extraction. You will be given unstructured text from AI analysis and you should convert it into the given structure.",
                    },
                    {
                        "role": "user", 
                        "content": response.output_text
                        },
                ],
                text_format=self.text_format,
            )

            return extractor.output_parsed.model_dump()

        # Handle potential errors
        except TimeoutError as e:
            print(f"The LLM API call encountered TimeoutError: {e}")

        except APIError as e:
            print(f"OpenAI API error: {e}")    

        except APIConnectionError as e:
            print(f"Connection error: {e}")
            
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            
        except Exception as e:
            print(f"Unexpected error: {e}")