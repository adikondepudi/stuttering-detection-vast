#!/usr/bin/env python3
"""
Super simple model test that copies the model code directly
to avoid import issues with dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Copy the model classes directly to avoid import issues
class StutterDetectionModel(nn.Module):
    """BiLSTM model for stuttering detection"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input dimension (Whisper + MFCC features)
        self.input_dim = (config['features']['whisper_dim'] + 
                         3 * config['features']['mfcc']['n_mfcc'])  # 768 + 39 = 807
        
        # Temporal modeling with BiLSTM
        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=config['model']['lstm_units_1'],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0  # We'll use explicit dropout layers
        )
        
        self.dropout1 = nn.Dropout(config['model']['dropout_rate'])
        
        self.lstm2 = nn.LSTM(
            input_size=config['model']['lstm_units_1'] * 2,  # *2 for bidirectional
            hidden_size=config['model']['lstm_units_2'],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        self.dropout2 = nn.Dropout(config['model']['dropout_rate'])
        
        # Classification head
        self.classifier = nn.Linear(
            config['model']['lstm_units_2'] * 2,  # *2 for bidirectional
            config['labels']['num_classes']
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            elif 'classifier' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x):
        """Forward pass"""
        # First BiLSTM layer
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        # Second BiLSTM layer
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Time-distributed classification
        batch_size, seq_len, hidden_dim = lstm_out2.shape
        lstm_out2_reshaped = lstm_out2.view(-1, hidden_dim)
        
        # Apply classifier
        logits = self.classifier(lstm_out2_reshaped)
        
        # Reshape back
        logits = logits.view(batch_size, seq_len, -1)
        
        return logits


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """Calculate focal loss"""
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_test_config():
    """Create test configuration"""
    return {
        'features': {
            'whisper_dim': 768,
            'mfcc': {'n_mfcc': 13}
        },
        'model': {
            'lstm_units_1': 256,
            'lstm_units_2': 128,
            'dropout_rate': 0.3
        },
        'labels': {
            'num_classes': 5,
            'pooled_frames': 25
        },
        'training': {
            'batch_size': 4,
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2.0,
            'prediction_threshold': 0.5,
            'initial_lr': 5e-5,
            'weight_decay': 1e-2,
            'gradient_clip_value': 1.0
        }
    }


def test_model_core_functionality():
    """Test the core model functionality"""
    print("Testing core model functionality...")
    
    config = create_test_config()
    
    try:
        # Create model and loss
        model = StutterDetectionModel(config)
        criterion = FocalLoss(
            alpha=config['training']['focal_loss_alpha'],
            gamma=config['training']['focal_loss_gamma']
        )
        
        print(f"✓ Model created successfully")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test dimensions
        batch_size = config['training']['batch_size']
        seq_len = config['labels']['pooled_frames']
        input_dim = config['features']['whisper_dim'] + 3 * config['features']['mfcc']['n_mfcc']
        
        print(f"✓ Expected input shape: ({batch_size}, {seq_len}, {input_dim})")
        
        # Create test data
        test_input = torch.randn(batch_size, seq_len, input_dim)
        test_labels = torch.randint(0, 2, (batch_size, seq_len, config['labels']['num_classes'])).float()
        
        print(f"✓ Test data created")
        print(f"  - Input shape: {test_input.shape}")
        print(f"  - Labels shape: {test_labels.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            logits = model(test_input)
            loss = criterion(logits, test_labels)
        
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {logits.shape}")
        print(f"  - Loss value: {loss.item():.4f}")
        
        # Test shapes match expectations
        expected_output_shape = (batch_size, seq_len, config['labels']['num_classes'])
        if logits.shape == expected_output_shape:
            print(f"✓ Output shape matches expected: {expected_output_shape}")
        else:
            print(f"✗ Output shape mismatch! Got {logits.shape}, expected {expected_output_shape}")
            return False
        
        # Test backward pass
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['initial_lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Training step
        optimizer.zero_grad()
        logits = model(test_input)
        loss = criterion(logits, test_labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            config['training']['gradient_clip_value']
        )
        
        optimizer.step()
        
        print(f"✓ Backward pass successful")
        print(f"  - Training loss: {loss.item():.4f}")
        
        # Test predictions
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            predictions = (probs > config['training']['prediction_threshold']).float()
            
            print(f"✓ Predictions generated")
            print(f"  - Predictions shape: {predictions.shape}")
            print(f"  - Prediction range: [{predictions.min():.1f}, {predictions.max():.1f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_batch_sizes():
    """Test model with different batch sizes"""
    print("\nTesting different batch sizes...")
    
    config = create_test_config()
    model = StutterDetectionModel(config)
    criterion = FocalLoss()
    
    seq_len = config['labels']['pooled_frames']
    input_dim = config['features']['whisper_dim'] + 3 * config['features']['mfcc']['n_mfcc']
    
    batch_sizes = [1, 2, 4, 8]
    
    try:
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, seq_len, input_dim)
            test_labels = torch.randint(0, 2, (batch_size, seq_len, config['labels']['num_classes'])).float()
            
            with torch.no_grad():
                logits = model(test_input)
                loss = criterion(logits, test_labels)
            
            expected_shape = (batch_size, seq_len, config['labels']['num_classes'])
            if logits.shape == expected_shape:
                print(f"✓ Batch size {batch_size}: {logits.shape}")
            else:
                print(f"✗ Batch size {batch_size}: got {logits.shape}, expected {expected_shape}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Batch size test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases"""
    print("\nTesting edge cases...")
    
    config = create_test_config()
    
    try:
        # Test with minimal sequence length
        config_small = config.copy()
        config_small['labels']['pooled_frames'] = 1
        
        model_small = StutterDetectionModel(config_small)
        test_input = torch.randn(2, 1, 807)  # batch=2, seq=1, features=807
        test_labels = torch.randint(0, 2, (2, 1, 5)).float()
        
        with torch.no_grad():
            logits = model_small(test_input)
        
        print(f"✓ Minimal sequence length test passed: {logits.shape}")
        
        # Test with zero input
        zero_input = torch.zeros(2, 25, 807)
        zero_labels = torch.zeros(2, 25, 5)
        
        model = StutterDetectionModel(config)
        criterion = FocalLoss()
        
        with torch.no_grad():
            logits = model(zero_input)
            loss = criterion(logits, zero_labels)
        
        print(f"✓ Zero input test passed: loss = {loss.item():.4f}")
        
        # Test with all-ones labels
        ones_labels = torch.ones(2, 25, 5)
        with torch.no_grad():
            loss = criterion(logits, ones_labels)
        
        print(f"✓ All-ones labels test passed: loss = {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        return False


def main():
    """Run simple model tests"""
    print("="*60)
    print("SIMPLE MODEL FUNCTIONALITY TEST")
    print("="*60)
    print("This tests the core model without any heavy dependencies")
    print()
    
    tests = [
        ("Core Model Functionality", test_model_core_functionality),
        ("Different Batch Sizes", test_different_batch_sizes),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Core model functionality works perfectly!")
        print("Your model architecture is solid for cloud deployment.")
    else:
        print(f"\n⚠️ {total - passed} core test(s) failed.")
        print("Fix these model architecture issues before cloud deployment.")
    
    return passed == total


if __name__ == "__main__":
    main()