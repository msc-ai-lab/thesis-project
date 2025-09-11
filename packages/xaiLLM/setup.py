from setuptools import setup, find_packages

setup(
  name='llm_interpretated_xai_outputs',
  version='0.1.1',
  description='An LLM-enhanced explainable AI.',
  packages=find_packages(include=['xaiLLM', 'xaiLLM.*']),
  install_requires=[
    'torch',
    'captum',
    'matplotlib',
    'openai',
    'tqdm',
    'Pillow',
    'pandas',
    'dotenv',
    'scikit-learn',
  ],
  python_requires='>=3.10',
)