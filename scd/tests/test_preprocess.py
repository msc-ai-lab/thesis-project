import pytest
import torch
from pathlib import Path
from scd.preprocess import preprocess_input

def test_preprocess_input_with_valid_image():
    img_path = Path(__file__).parent.parent.parent / 'data' / 'raw_dataset' / 'images' / '000001.png'
    processed_tensor = preprocess_input(img_path, (224, 224))
    
    assert isinstance(processed_tensor, torch.Tensor)
    assert processed_tensor.dim() == 4  # Should have batch dimension
    assert processed_tensor.shape[1:] == (3, 224, 224)  # Check for RGB and resize


def test_preprocess_input_with_invalid_image():
    img_path = Path(__file__).parent.parent.parent / 'data' / 'raw_dataset' / 'images' / 'non_existent_image.png'
    
    with pytest.raises(FileNotFoundError):
        preprocess_input(img_path, (224, 224))