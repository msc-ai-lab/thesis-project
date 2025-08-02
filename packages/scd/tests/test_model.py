import torch
from scd.model import SkinCancerCNN

def test_model_initialization():
    model = SkinCancerCNN(num_classes=2)
    assert isinstance(model, SkinCancerCNN)

def test_forward_pass():
    model = SkinCancerCNN()
    model.eval()  # Set model to evaluation mode
    # Create dummy input tensor (batch_size=2, channels=3, height=384, width=384)
    x = torch.randn(2, 3, 384, 384)
    
    # Forward pass
    logits, attention_map = model(x)
    
    # Test output shapes
    assert logits.shape == (2, 2), f"Expected logits shape (2, 2), got {logits.shape}"
    assert attention_map.shape[0] == 2, f"Expected batch size 2, got {attention_map.shape[0]}"
    assert attention_map.shape[1] == 1, f"Expected 1 attention channel, got {attention_map.shape[1]}"
    
    # Test output types
    assert isinstance(logits, torch.Tensor), "Logits should be a torch.Tensor"
    assert isinstance(attention_map, torch.Tensor), "Attention map should be a torch.Tensor"
    
    # Test attention map values (should be between 0 and 1 after sigmoid)
    assert torch.all(attention_map >= 0) and torch.all(attention_map <= 1), "Attention values should be in range [0,1]"