from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from pydantic import BaseModel, Field
from typing import Literal, Annotated, List
from xaiLLM.utils.config import OPENAI_API_KEY

class TextFormat(BaseModel):
    prediction: Literal['Benign', 'Malignant'] = Field(
        description="'Benign' for when the AI analysis suggests low or moderately low concern for malignancy; 'Malignant' for when the AI analysis indicates high or moderately high concern for malignancy."
        )
    # borderline: Literal[True, False] = Field(
    #     description="True if the model considers this to be a borderline case, False otherwise. Must be exactly True or False."
    # )
    confidence: Annotated[float, Field(
        description="Model confidence, as indicated in Confidence Level section, to 2 decimal places."
        )]
    influential_cases_percentage: Annotated[int, Field(
        description="Influence Functions: What percentage of the most influential training cases share the same ground truth label as the predicted class."
    )]


class Parser:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.text_format = TextFormat

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
            A dictionary containing the parsed structured data according to the TextFormat model 
            with "borderline" status added in a subsequent step.
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
                        "content": response
                        },
                ],
                text_format=self.text_format,
            )

            extracted_data = extractor.output_parsed.model_dump()

            # Define custom function for extracting the borderline prediction status
            def borderline_parser(llm_output : str, key_words: List[str]) -> bool:
                """
                Parse the LLM output and search for the key words to confirm if the given prediction 
                was interpreted as "borderline". 
                
                Arguments:
                llm_output (str): original LLM interpretation of XAI methods in the skin cancer prediction
                key_words (List[str]): key words to match against the LLM output
                
                Returns:
                A bool value for the presence or absence of the "borderline" prediction status.
                """
                
                # Narrow-down parsing focus if exact heading is present in the LLM output
                if '**Confidence Level**' in llm_output:
                    start_indx = 0
                    end_indx = llm_output.find("**Confidence Level**")
                    llm_output = llm_output[start_indx : end_indx].lower()
                else:
                    llm_output.lower()
                    
                borderline = False
                key_words = [word.lower() for word in key_words]
                
                # Parse in search of the key words (with lowered case)
                for word in key_words:
                    if word in llm_output:
                        borderline = True
                        break

                return borderline

            # Implementat the function and update extracted_data with the function's finding
            key_words = ["borderline", "no clear decision", "uncertain", "indeterminate", "ambiguous", "unclear", "equivocal" "between benign and malignant"]
            borderline_status = borderline_parser(llm_output=response, 
                                                key_words=key_words)
            extracted_data["borderline"] = borderline_status

            return extracted_data


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
