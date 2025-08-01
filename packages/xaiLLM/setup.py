from setuptools import setup, find_packages

setup(
  name='llm_interpretated_xai_outputs',
  version='0.1.0',
  description='An LLM-enhanced explainable AI.',
  packages=find_packages(include=['xaiLLM', 'xaiLLM.*']),
  install_requires=[
    'torch',
    'captum',
    'matplotlib',
    'openai',
    'tqdm',
    'Pillow',
    'numpy',
    'pandas',
  ],
  python_requires='>=3.8',
)