import pytest
import torch
import torch.nn as nn
from scd.inference import predict


class MockModel(nn.Module):
    """Mock model for testing inference functionality."""
    
    def __init__(self, return_benign=True):
        super().__init__()
        self.return_benign = return_benign
        
    def forward(self, x):
        # Return mock outputs and logits for testing
        batch_size = x.shape[0]
        if self.return_benign:
            # Higher probability for benign (class 0)
            outputs = torch.tensor([[0.8, 0.2]] * batch_size)
        else:
            # Higher probability for malignant (class 1) 
            outputs = torch.tensor([[0.3, 0.7]] * batch_size)

        return outputs, outputs  # Return same tensor for outputs and logits


def test_predict():
    """Test the predict function with mock models and input tensors."""
    
    # Test case 1: Benign prediction
    benign_model = MockModel(return_benign=True)
    input_tensor = torch.randn(1, 3, 224, 224)  # Mock image tensor
    
    (pred_idx, pred_name), probabilities = predict(benign_model, input_tensor)
    
    # Assertions for benign prediction
    assert pred_idx == 0, f"Expected class index 0 (benign), got {pred_idx}"
    assert pred_name == "Benign", f"Expected 'Benign', got '{pred_name}'"
    assert "Benign" in probabilities, "Probabilities dict should contain 'Benign' key"
    assert "Malignant" in probabilities, "Probabilities dict should contain 'Malignant' key"
    assert round(probabilities["Benign"], 2) == 0.8, f"Expected benign probability 0.8, got {probabilities['Benign']}"
    assert round(probabilities["Malignant"], 2) == 0.2, f"Expected malignant probability 0.2, got {probabilities['Malignant']}"
    
    # Test case 2: Malignant prediction
    malignant_model = MockModel(return_benign=False)
    
    (pred_idx, pred_name), probabilities = predict(malignant_model, input_tensor)
    
    # Assertions for malignant prediction
    assert pred_idx == 1, f"Expected class index 1 (malignant), got {pred_idx}"
    assert pred_name == "Malignant", f"Expected 'Malignant', got '{pred_name}'"
    assert round(probabilities["Benign"], 2) == 0.3, f"Expected benign probability 0.3, got {probabilities['Benign']}"
    assert round(probabilities["Malignant"], 2) == 0.7, f"Expected malignant probability 0.7, got {probabilities['Malignant']}"
    
    # Test case 3: Test with different input tensor shapes
    larger_input = torch.randn(2, 3, 299, 299)  # Different batch size and image size
    with pytest.raises(RuntimeError):
        predict(benign_model, larger_input)


def test_predict_output_types():
    """Test that predict function returns correct output types."""
    
    model = MockModel(return_benign=True)
    input_tensor = torch.randn(1, 3, 224, 224)
    
    result = predict(model, input_tensor)
    
    # Test return structure
    assert isinstance(result, tuple), "Function should return a tuple"
    assert len(result) == 2, "Function should return a tuple of length 2"
    
    prediction_tuple, probabilities = result
    
    # Test prediction tuple
    assert isinstance(prediction_tuple, tuple), "First element should be a tuple"
    assert len(prediction_tuple) == 2, "Prediction tuple should have length 2"
    
    pred_idx, pred_name = prediction_tuple
    assert isinstance(pred_idx, int), "Prediction index should be an integer"
    assert isinstance(pred_name, str), "Prediction name should be a string"
    
    # Test probabilities dictionary
    assert isinstance(probabilities, dict), "Second element should be a dictionary"
    assert all(isinstance(k, str) for k in probabilities.keys()), "All probability keys should be strings"
    assert all(isinstance(v, float) for v in probabilities.values()), "All probability values should be floats"
