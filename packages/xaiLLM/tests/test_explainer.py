"""
Test cases for xaiLLM explainer module
=====================================

This test suite covers all functions in the xaiLLM.explainer module including:
- WrapperModel class functionality
- grad_cam visualization generation
- shap visualization generation  
- get_gradient_and_prediction calculations
- calculate_real_influence scoring
- calculate_influence DataFrame generation

The tests use mock models, synthetic data, and assertions to validate
correct behavior, output formats, and error handling.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from unittest.mock import patch
import tempfile
import os

# Import the functions to test
from xaiLLM.explainer import (
    WrapperModel, 
    grad_cam, 
    shap, 
    get_gradient_and_prediction,
    calculate_real_influence,
    calculate_influence
)


class MockSkinCancerModel(nn.Module):
    """Mock model that mimics the SkinCancerCNN structure for testing."""
    
    def __init__(self, return_tuple=True, num_classes=2):
        super().__init__()
        self.return_tuple = return_tuple
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        if self.return_tuple:
            # Return tuple like SkinCancerCNN (logits, attention_map)
            attention_map = torch.rand(x.shape[0], 1, x.shape[2]//8, x.shape[3]//8)
            return logits, attention_map
        return logits


class TestWrapperModel:
    """Test cases for WrapperModel class."""
    
    def test_wrapper_model_init(self):
        """Test WrapperModel initialization."""
        base_model = MockSkinCancerModel()
        wrapper = WrapperModel(base_model)
        assert wrapper.model is base_model
        
    def test_get_logits_with_tuple_output(self):
        """Test get_logits with model that returns tuple."""
        base_model = MockSkinCancerModel(return_tuple=True)
        wrapper = WrapperModel(base_model)
        
        input_tensor = torch.randn(1, 3, 224, 224)
        logits = wrapper.get_logits(input_tensor)
        
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (1, 2)  # batch_size=1, num_classes=2
        
    def test_get_logits_with_single_output(self):
        """Test get_logits with model that returns single tensor."""
        base_model = MockSkinCancerModel(return_tuple=False)
        wrapper = WrapperModel(base_model)
        
        input_tensor = torch.randn(1, 3, 224, 224)
        logits = wrapper.get_logits(input_tensor)
        
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (1, 2)


class TestGradCAM:
    """Test cases for grad_cam function."""
    
    @pytest.fixture
    def setup_grad_cam_test(self):
        """Setup common test fixtures for Grad-CAM tests."""
        model = MockSkinCancerModel()
        input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        
        # Create a temporary image file
        temp_image = Image.new('RGB', (224, 224), color='red')
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_image.save(temp_file.name)
        
        return model, input_tensor, temp_file.name, 1  # predicted_class_index=1
        
    def test_grad_cam_execution(self, setup_grad_cam_test):
        """Test that grad_cam executes without errors."""
        model, input_tensor, temp_image_path, pred_idx = setup_grad_cam_test
        
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.close'):
            try:
                result = grad_cam(model, input_tensor, temp_image_path, pred_idx)
                assert isinstance(result, Image.Image)
            finally:
                os.unlink(temp_image_path)  # Clean up temp file
                
    def test_grad_cam_requires_grad(self, setup_grad_cam_test):
        """Test that grad_cam handles tensors without gradients."""
        model, input_tensor, temp_image_path, pred_idx = setup_grad_cam_test
        
        # Create tensor without gradients
        input_no_grad = torch.randn(1, 3, 224, 224, requires_grad=False)
        
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.close'):
            try:
                result = grad_cam(model, input_no_grad, temp_image_path, pred_idx)
                assert isinstance(result, Image.Image)
                # Should have enabled gradients internally
                assert input_no_grad.requires_grad is True
            finally:
                os.unlink(temp_image_path)
                
    def test_grad_cam_model_eval_mode(self, setup_grad_cam_test):
        """Test that grad_cam puts model in eval mode."""
        model, input_tensor, temp_image_path, pred_idx = setup_grad_cam_test
        
        model.train()  # Set to train mode initially
        assert model.training is True
        
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.close'):
            try:
                grad_cam(model, input_tensor, temp_image_path, pred_idx)
                assert model.training is False  # Should be in eval mode
            finally:
                os.unlink(temp_image_path)


class TestSHAP:
    """Test cases for shap function."""
    
    @pytest.fixture
    def setup_shap_test(self):
        """Setup common test fixtures for SHAP tests."""
        model = MockSkinCancerModel()
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Create a temporary image file
        temp_image = Image.new('RGB', (224, 224), color='blue')
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_image.save(temp_file.name)
        
        return model, input_tensor, temp_file.name, 0  # predicted_class_index=0
        
    def test_shap_execution(self, setup_shap_test):
        """Test that shap executes without errors."""
        model, input_tensor, temp_image_path, pred_idx = setup_shap_test
        
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.close'):
            try:
                result = shap(model, input_tensor, temp_image_path, pred_idx)
                assert isinstance(result, Image.Image)
            finally:
                os.unlink(temp_image_path)
                
    def test_shap_model_eval_mode(self, setup_shap_test):
        """Test that shap puts model in eval mode."""
        model, input_tensor, temp_image_path, pred_idx = setup_shap_test
        
        model.train()  # Set to train mode initially
        
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.close'):
            try:
                shap(model, input_tensor, temp_image_path, pred_idx)
                assert model.training is False
            finally:
                os.unlink(temp_image_path)


class TestGradientAndPrediction:
    """Test cases for get_gradient_and_prediction function."""
    
    def test_get_gradient_and_prediction_basic(self):
        """Test basic functionality of get_gradient_and_prediction."""
        model = MockSkinCancerModel()
        data_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        target = torch.tensor([1])
        loss_fn = nn.CrossEntropyLoss()
        
        # Clear any existing gradients
        model.zero_grad()
        
        grad, pred_idx = get_gradient_and_prediction(model, data_tensor, target, loss_fn)
        
        assert isinstance(grad, torch.Tensor)
        assert isinstance(pred_idx, int)
        assert pred_idx in [0, 1]  # Binary classification
        assert grad.dim() == 1  # Should be flattened
        
    def test_gradient_computation(self):
        """Test that gradients are actually computed."""
        model = MockSkinCancerModel()
        data_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        target = torch.tensor([0])
        loss_fn = nn.CrossEntropyLoss()
        
        # Ensure no gradients initially
        model.zero_grad()
        assert model.classifier[-1].weight.grad is None
        
        grad, _ = get_gradient_and_prediction(model, data_tensor, target, loss_fn)
        
        # Gradients should now exist
        assert model.classifier[-1].weight.grad is not None
        assert torch.any(grad != 0)  # Some gradients should be non-zero
        
    def test_model_eval_mode(self):
        """Test that model is set to eval mode."""
        model = MockSkinCancerModel()
        model.train()  # Start in train mode
        
        data_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        target = torch.tensor([1])
        loss_fn = nn.CrossEntropyLoss()
        
        get_gradient_and_prediction(model, data_tensor, target, loss_fn)
        
        assert model.training is False


class TestCalculateRealInfluence:
    """Test cases for calculate_real_influence function."""
    
    @pytest.fixture
    def setup_influence_test(self):
        """Setup test fixtures for influence calculation."""
        model = MockSkinCancerModel()
        device = torch.device('cpu')
        
        # Create synthetic training data
        train_images = torch.randn(10, 3, 224, 224)
        train_labels = torch.randint(0, 2, (10,))
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=4)
        
        filenames = [f'image_{i:03d}.jpg' for i in range(10)]
        
        test_tensor = torch.randn(1, 3, 224, 224)
        test_target = 1
        
        return model, train_loader, filenames, test_tensor, test_target, device
        
    def test_calculate_real_influence_basic(self, setup_influence_test):
        """Test basic functionality of calculate_real_influence."""
        model, train_loader, filenames, test_tensor, test_target, device = setup_influence_test
        
        with patch('builtins.print'):  # Suppress print statements
            results = calculate_real_influence(
                model, train_loader, filenames, test_tensor, test_target, device
            )
        
        assert isinstance(results, list)
        assert len(results) == 10  # Should have results for all training examples
        
        # Check result structure
        for result in results:
            assert 'score' in result
            assert 'prediction' in result
            assert 'filename' in result
            assert isinstance(result['score'], float)
            assert isinstance(result['prediction'], int)
            assert isinstance(result['filename'], str)
            
    def test_influence_scores_are_numeric(self, setup_influence_test):
        """Test that influence scores are proper numeric values."""
        model, train_loader, filenames, test_tensor, test_target, device = setup_influence_test
        
        with patch('builtins.print'):
            results = calculate_real_influence(
                model, train_loader, filenames, test_tensor, test_target, device
            )
        
        scores = [r['score'] for r in results]
        assert all(isinstance(score, (int, float)) for score in scores)
        assert all(not np.isnan(score) for score in scores)


class TestCalculateInfluence:
    """Test cases for calculate_influence function."""
    
    @pytest.fixture
    def setup_calculate_influence_test(self):
        """Setup test fixtures for calculate_influence."""
        model = MockSkinCancerModel()
        input_tensor = torch.randn(1, 3, 224, 224)
        predicted_class_index = 1
        
        # Create synthetic training dataset
        train_images = torch.randn(20, 3, 224, 224)
        train_labels = torch.randint(0, 2, (20,))
        training_dataset = TensorDataset(train_images, train_labels)
        
        filenames = [f'train_image_{i:03d}.jpg' for i in range(20)]
        
        return model, input_tensor, predicted_class_index, training_dataset, filenames
        
    def test_calculate_influence_returns_dataframe(self, setup_calculate_influence_test):
        """Test that calculate_influence returns a proper DataFrame."""
        args = setup_calculate_influence_test
        model, input_tensor, pred_idx, training_dataset, filenames = args
        
        with patch('builtins.print'):  # Suppress print statements
            result = calculate_influence(*args)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 100  # Should return top 100 or fewer
        
        # Check required columns
        expected_columns = ['case_id', 'influence_score', 'ground_truth', 'prediction']
        for col in expected_columns:
            assert col in result.columns
            
    def test_influence_dataframe_sorting(self, setup_calculate_influence_test):
        """Test that influence results are sorted by absolute influence score."""
        args = setup_calculate_influence_test
        
        with patch('builtins.print'):
            result = calculate_influence(*args)
        
        # Should be sorted by absolute influence score in descending order
        abs_scores = result['influence_score'].abs()
        assert all(abs_scores.iloc[i] >= abs_scores.iloc[i+1] 
                  for i in range(len(abs_scores)-1))
                  
    def test_influence_label_mapping(self, setup_calculate_influence_test):
        """Test that labels are properly mapped to strings."""
        args = setup_calculate_influence_test
        
        with patch('builtins.print'):
            result = calculate_influence(*args)
        
        # Check that ground truth and prediction are mapped to strings
        valid_labels = {'Benign', 'Malignant', 'Unknown'}
        assert all(gt in valid_labels for gt in result['ground_truth'])
        assert all(pred in valid_labels for pred in result['prediction'])


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_end_to_end_workflow(self):
        """Test a complete XAI workflow with all functions."""
        # Setup
        model = MockSkinCancerModel()
        input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        
        # Create temporary image
        temp_image = Image.new('RGB', (224, 224), color='green')
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_image.save(temp_file.name)
        
        # Training data for influence calculation
        train_images = torch.randn(5, 3, 224, 224)
        train_labels = torch.randint(0, 2, (5,))
        training_dataset = TensorDataset(train_images, train_labels)
        filenames = [f'train_{i}.jpg' for i in range(5)]
        
        try:
            # Test complete workflow
            with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.close'), patch('builtins.print'):
                # Grad-CAM
                gradcam_result = grad_cam(model, input_tensor, temp_file.name, 1)
                assert isinstance(gradcam_result, Image.Image)
                
                # SHAP
                shap_result = shap(model, input_tensor, temp_file.name, 0)
                assert isinstance(shap_result, Image.Image)
                
                # Influence Functions
                influence_result = calculate_influence(
                    model, input_tensor, 1, training_dataset, filenames
                )
                assert isinstance(influence_result, pd.DataFrame)
                
        finally:
            os.unlink(temp_file.name)  # Clean up
            
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        model = MockSkinCancerModel()
        
        # Test with invalid image path
        with pytest.raises(FileNotFoundError):
            grad_cam(model, torch.randn(1, 3, 224, 224), "nonexistent.jpg", 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
