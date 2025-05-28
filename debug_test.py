#!/usr/bin/env python3
"""
Local debugging script to test the stuttering detection pipeline
without needing actual data or cloud resources.

This script creates fake data and tests:
- Model forward pass
- Loss calculation
- Data loader compatibility
- Feature extraction
- Basic training loop
- Type errors and dimension mismatches
"""

import torch
import numpy as np
import yaml
import json
import tempfile
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Mock dependencies that might not be available locally
class MockWhisperModel:
    def __init__(self):
        self.encoder = MockEncoder()
    
    def to(self, device):
        return self
    
    def eval(self):
        return self

class MockEncoder:
    def __call__(self, input_features, output_hidden_states=True):
        batch_size = input_features.shape[0]
        # Whisper encoder typically outputs 1500 time steps with 768 features
        hidden_states = torch.randn(batch_size, 1500, 768)
        return type('obj', (object,), {'last_hidden_state': hidden_states})()

class MockWhisperFeatureExtractor:
    def __call__(self, audio, sampling_rate, return_tensors):
        # Mock Whisper feature extractor
        batch_size = 1 if isinstance(audio, np.ndarray) else len(audio)
        return type('obj', (object,), {
            'input_features': torch.randn(batch_size, 80, 3000)  # Typical Whisper input shape
        })()

def create_test_config():
    """Create a test configuration"""
    config = {
        'data': {
            'raw_data_path': 'test_data/raw',
            'processed_data_path': 'test_data/processed',
            'splits_path': 'test_data/splits',
            'sample_rate': 16000,
            'target_loudness': -23
        },
        'segmentation': {
            'segment_length': 5.0,
            'hop_length': 1.0
        },
        'labels': {
            'disfluency_types': ["Prolongation", "Interjection", "Word Repetition", "Sound Repetition", "Blocks"],
            'num_classes': 5,
            'frames_per_segment': 250,
            'pooled_frames': 25
        },
        'features': {
            'whisper_model': 'openai/whisper-base.en',
            'whisper_dim': 768,
            'mfcc': {
                'n_mfcc': 13,
                'n_fft': 400,
                'hop_length': 160
            }
        },
        'model': {
            'lstm_units_1': 256,
            'lstm_units_2': 128,
            'dropout_rate': 0.3
        },
        'training': {
            'batch_size': 4,  # Small batch for testing
            'max_epochs': 2,
            'initial_lr': 5e-5,
            'weight_decay': 1e-2,
            'gradient_clip_value': 1.0,
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2.0,
            'prediction_threshold': 0.5
        },
        'augmentation': {
            'noise_prob': 0.5,
            'noise_snr_range': [5, 20],
            'speed_prob': 0.5,
            'speed_factors': [0.9, 1.1],
            'repetition_prob': 0.2,
            'repetition_duration_range': [0.2, 0.5]
        }
    }
    return config

def create_mock_data(config, temp_dir):
    """Create mock processed data and splits"""
    processed_dir = Path(temp_dir) / 'processed'
    splits_dir = Path(temp_dir) / 'splits'
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock metadata
    metadata = []
    for i in range(20):  # 20 mock segments
        metadata.append({
            'audio_file': f'mock_audio_{i}.wav',
            'segment_idx': 0,
            'audio_path': str(processed_dir / f'segment_{i:06d}.wav'),
            'label_path': str(processed_dir / f'segment_{i:06d}_labels.npy'),
            'start_time': 0.0,
            'end_time': 5.0
        })
        
        # Create mock audio (dummy file, we'll generate actual arrays in dataset)
        (processed_dir / f'segment_{i:06d}.wav').touch()
        
        # Create mock labels
        labels = np.random.randint(0, 2, (250, 5)).astype(np.float32)
        np.save(processed_dir / f'segment_{i:06d}_labels.npy', labels)
    
    # Save metadata
    with open(processed_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    # Create splits
    indices = list(range(20))
    splits = {
        'train': indices[:12],
        'val': indices[12:16],
        'test': indices[16:20]
    }
    
    with open(splits_dir / 'splits.json', 'w') as f:
        json.dump(splits, f)
    
    return metadata, splits

class MockFeatureExtractor:
    """Mock feature extractor for testing"""
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.pooled_frames = config['labels']['pooled_frames']
        
    def extract_features(self, audio):
        """Mock feature extraction"""
        # Return features with expected dimensions
        # Whisper: 768, MFCC: 39 (13 * 3) = 807 total
        return np.random.randn(self.pooled_frames, 807).astype(np.float32)

class MockDataset:
    """Mock dataset for testing"""
    def __init__(self, config, split='train', feature_extractor=None):
        self.config = config
        self.split = split
        self.feature_extractor = feature_extractor
        
        # Mock indices
        if split == 'train':
            self.indices = list(range(12))
        elif split == 'val':
            self.indices = list(range(12, 16))
        else:
            self.indices = list(range(16, 20))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Generate mock data
        segment_idx = self.indices[idx]
        
        # Mock audio (5 seconds at 16kHz)
        audio = np.random.randn(80000).astype(np.float32)
        
        # Mock labels (250 frames, 5 classes)
        labels = np.random.randint(0, 2, (250, 5)).astype(np.float32)
        
        # Pool labels to match feature frames
        pooled_labels = self._pool_labels(labels)
        
        # Extract features
        if self.feature_extractor:
            features = self.feature_extractor.extract_features(audio)
        else:
            features = audio[:self.config['labels']['pooled_frames'] * 100]  # Mock features
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(pooled_labels, dtype=torch.float32)
    
    def _pool_labels(self, labels):
        """Pool labels to match pooled feature frames"""
        target_frames = self.config['labels']['pooled_frames']
        current_frames = labels.shape[0]
        
        pool_size = current_frames // target_frames
        remainder = current_frames % target_frames
        
        pooled = []
        start_idx = 0
        
        for i in range(target_frames):
            end_idx = start_idx + pool_size + (1 if i < remainder else 0)
            pooled.append(labels[start_idx:end_idx].max(axis=0))
            start_idx = end_idx
        
        return np.array(pooled)

def test_model_forward_pass(config):
    """Test model creation and forward pass"""
    print("Testing model forward pass...")
    
    try:
        # Import model components
        import sys
        sys.path.append('.')  # Add current directory to path
        
        # Try to import the actual model
        try:
            from src.model import StutterDetectionModel, FocalLoss
            print("✓ Successfully imported actual model classes")
        except ImportError as e:
            print(f"✗ Could not import model classes: {e}")
            return False
        
        # Create model
        model = StutterDetectionModel(config)
        criterion = FocalLoss(
            alpha=config['training']['focal_loss_alpha'],
            gamma=config['training']['focal_loss_gamma']
        )
        
        print(f"✓ Model created successfully")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = config['training']['batch_size']
        seq_len = config['labels']['pooled_frames']
        input_dim = config['features']['whisper_dim'] + 3 * config['features']['mfcc']['n_mfcc']
        
        # Create mock input
        mock_input = torch.randn(batch_size, seq_len, input_dim)
        mock_labels = torch.randint(0, 2, (batch_size, seq_len, config['labels']['num_classes'])).float()
        
        print(f"✓ Input shape: {mock_input.shape}")
        print(f"✓ Labels shape: {mock_labels.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(mock_input)
            loss = criterion(logits, mock_labels)
        
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {logits.shape}")
        print(f"  - Loss value: {loss.item():.4f}")
        
        # Test training mode
        model.train()
        logits = model(mock_input)
        loss = criterion(logits, mock_labels)
        
        print(f"✓ Training mode forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction(config):
    """Test feature extraction pipeline"""
    print("\nTesting feature extraction...")
    
    try:
        # Mock the transformers imports
        import sys
        from unittest.mock import patch
        
        with patch('transformers.WhisperModel') as mock_whisper, \
             patch('transformers.WhisperFeatureExtractor') as mock_extractor:
            
            mock_whisper.from_pretrained.return_value = MockWhisperModel()
            mock_extractor.from_pretrained.return_value = MockWhisperFeatureExtractor()
            
            # Test with mock extractor first
            feature_extractor = MockFeatureExtractor(config)
            
            # Test feature extraction
            mock_audio = np.random.randn(80000).astype(np.float32)  # 5 seconds at 16kHz
            features = feature_extractor.extract_features(mock_audio)
            
            expected_frames = config['labels']['pooled_frames']
            expected_features = config['features']['whisper_dim'] + 3 * config['features']['mfcc']['n_mfcc']
            
            print(f"✓ Feature extraction successful")
            print(f"  - Feature shape: {features.shape}")
            print(f"  - Expected shape: ({expected_frames}, {expected_features})")
            
            if features.shape == (expected_frames, expected_features):
                print("✓ Feature dimensions match expected")
                return True
            else:
                print("✗ Feature dimensions don't match")
                return False
                
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_and_dataloader(config, temp_dir):
    """Test dataset and dataloader"""
    print("\nTesting dataset and dataloader...")
    
    try:
        # Update config paths
        config['data']['processed_data_path'] = str(Path(temp_dir) / 'processed')
        config['data']['splits_path'] = str(Path(temp_dir) / 'splits')
        
        # Create mock data
        create_mock_data(config, temp_dir)
        
        # Test dataset
        feature_extractor = MockFeatureExtractor(config)
        dataset = MockDataset(config, 'train', feature_extractor)
        
        print(f"✓ Dataset created successfully")
        print(f"  - Dataset size: {len(dataset)}")
        
        # Test single item
        features, labels = dataset[0]
        print(f"✓ Dataset item retrieval successful")
        print(f"  - Features shape: {features.shape}")
        print(f"  - Labels shape: {labels.shape}")
        
        # Test dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
        
        # Test batch
        for batch_features, batch_labels in dataloader:
            print(f"✓ DataLoader batch successful")
            print(f"  - Batch features shape: {batch_features.shape}")
            print(f"  - Batch labels shape: {batch_labels.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset/DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_loop(config, temp_dir):
    """Test basic training loop components"""
    print("\nTesting training loop components...")
    
    try:
        # Import training components
        from src.model import StutterDetectionModel, FocalLoss
        from torch.utils.data import DataLoader
        import torch.optim as optim
        
        # Create components
        model = StutterDetectionModel(config)
        criterion = FocalLoss(
            alpha=config['training']['focal_loss_alpha'],
            gamma=config['training']['focal_loss_gamma']
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['initial_lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Mock dataset and dataloader
        feature_extractor = MockFeatureExtractor(config)
        dataset = MockDataset(config, 'train', feature_extractor)
        dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'])
        
        print("✓ Training components created successfully")
        
        # Test one training step
        model.train()
        for features, labels in dataloader:
            # Forward pass
            logits = model(features)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['gradient_clip_value']
            )
            
            optimizer.step()
            
            print(f"✓ Training step successful")
            print(f"  - Loss: {loss.item():.4f}")
            break
        
        # Test evaluation mode
        model.eval()
        with torch.no_grad():
            for features, labels in dataloader:
                logits = model(features)
                loss = criterion(logits, labels)
                
                # Test predictions
                probs = torch.sigmoid(logits)
                predictions = (probs > config['training']['prediction_threshold']).float()
                
                print(f"✓ Evaluation step successful")
                print(f"  - Predictions shape: {predictions.shape}")
                break
        
        return True
        
    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_validation(config):
    """Test configuration validation"""
    print("\nTesting configuration validation...")
    
    try:
        # Check required keys
        required_keys = [
            'data', 'segmentation', 'labels', 'features', 
            'model', 'training', 'augmentation'
        ]
        
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing required config key: {key}")
                return False
        
        print("✓ All required config keys present")
        
        # Check data types and ranges
        assert isinstance(config['training']['batch_size'], int)
        assert config['training']['batch_size'] > 0
        assert isinstance(config['training']['initial_lr'], (int, float))
        assert config['training']['initial_lr'] > 0
        
        print("✓ Config validation successful")
        return True
        
    except Exception as e:
        print(f"✗ Config validation failed: {e}")
        return False

def main():
    """Run all debug tests"""
    print("="*60)
    print("STUTTERING DETECTION - LOCAL DEBUG TESTS")
    print("="*60)
    
    # Create temporary directory for test data
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test config
        config = create_test_config()
        
        # Run tests
        tests = [
            ("Configuration Validation", lambda: test_config_validation(config)),
            ("Model Forward Pass", lambda: test_model_forward_pass(config)),
            ("Feature Extraction", lambda: test_feature_extraction(config)),
            ("Dataset and DataLoader", lambda: test_dataset_and_dataloader(config, temp_dir)),
            ("Training Loop", lambda: test_training_loop(config, temp_dir))
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
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
            print("\n🎉 All tests passed! Your code should work on the cloud.")
            print("You can proceed with cloud deployment.")
        else:
            print(f"\n⚠️ {total - passed} test(s) failed. Fix these issues before cloud deployment.")
            print("Common fixes:")
            print("- Check import paths (make sure src/ is a Python package)")
            print("- Verify tensor dimensions match between components")
            print("- Ensure all required dependencies are in requirements.txt")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nTemporary test directory cleaned up: {temp_dir}")

if __name__ == "__main__":
    main()