import torch
from scd.model import SkinCancerCNN

def test_model_initialization():
    model = SkinCancerCNN(num_classes=2)
    assert isinstance(model, SkinCancerCNN)

def test_forward_pass():
    model = SkinCancerCNN()
    model.eval()  # Set model to evaluation mode
    # Create sample input with appropriate shape
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        outputs, attention_map = model(input_tensor)
    
    # Check output shape
    assert outputs.shape == (batch_size, 2), f"Expected output shape (2, 2), got {outputs.shape}"
    
    # Check attention map shape
    assert attention_map.shape[0] == batch_size, "Batch size mismatch in attention map"
    assert attention_map.shape[1] == 1, "Attention map should have 1 channel"
    
    # Check if outputs are valid probabilities (softmax output)
    assert torch.all(outputs >= 0) and torch.all(outputs <= 1), "Outputs should be probabilities between 0 and 1"
    assert torch.allclose(outputs.sum(dim=1), torch.ones(batch_size)), "Probability distribution should sum to 1"