"""
Configuration for xaiLLM.

Loads environment variables (e.g., OPENAI_API_KEY) used across the package.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Configuration for xaiLLM package
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')