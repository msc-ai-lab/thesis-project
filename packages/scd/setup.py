from setuptools import setup, find_packages

setup(
  name='skin-cancer-detection',
  version='0.1.0',
  description='An LLM-enhanced skin cancer detection application using convolutional neural networks and explainable AI.',
  packages=find_packages(include=['scd', 'scd.*']),
  install_requires=[
    'albumentations',
    'matplotlib',
    'Pillow',
    'torch',
    'torchvision',
  ],
  python_requires='>=3.10',
)