import os
from dotenv import load_dotenv

load_dotenv()

# Configuration for xaiLLM package
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')