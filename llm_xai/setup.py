from setuptools import setup, find_packages

setup(
  name='llm_interpretated_xai_outputs',
  version='0.1.0',
  description='An LLM-enhanced explainable AI.',
  packages=find_packages(include=['llm_xai', 'llm_xai.*']),
  install_requires=[
    'torch',
    'captum',
    'matplotlib',
    'openai',
  ],
  python_requires='>=3.8',
)