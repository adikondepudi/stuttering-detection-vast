#!/usr/bin/env python3
"""
Feature pre-extraction system to speed up training
Extracts and caches all features before training starts
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
import os
from typing import Dict, List, Optional
import gc
import librosa

class FeaturePreprocessor:
    """Pre-extract and cache features for all training data"""
    
    def __init__(self, config, feature_extractor, device='cuda'):
        self.config = config
        self.feature_extractor = feature_extractor
        self.device = device
        self.sample_rate = config['data']['sample_rate']
        
        # Paths
        self.processed_path = Path(config['data']['processed_data_path'])
        self.features_path = self.processed_path / 'features'
        self.features_path.mkdir(exist_ok=True)
        
        # Feature info
        self.expected_feature_dim = feature_extractor.get_feature_dim()
        self.pooled_frames = config['labels']['pooled_frames']
        
        print(f"FeaturePreprocessor initialized:")
        print(f"  Device: {self.device}")
        print(f"  Features path: {self.features_path}")
        print(f"  Feature dimension: {self.expected_feature_dim}")
        print(f"  Pooled frames: {self.pooled_frames}")
        
    def check_if_features_exist(self) -> bool:
        """Check if features are already extracted"""
        feature_info_path = self.features_path / 'feature_info.json'
        
        if not feature_info_path.exists():
            return False
            
        try:
            with open(feature_info_path, 'r') as f:
                info = json.load(f)
            
            # Check if configuration matches
            if (info.get('feature_dim') == self.expected_feature_dim and
                info.get('pooled_frames') == self.pooled_frames and
                info.get('whisper_model') == self.config['features']['whisper_model']):
                
                # Verify some features actually exist
                feature_files = list(self.features_path.glob('*.npy'))
                if len(feature_files) > 0:
                    print(f"Found {len(feature_files)} pre-extracted features")
                    return True
                    
        except Exception as e:
            print(f"Error checking features: {e}")
            
        return False
    
    def extract_all_features(self, force_reextract=False):
        """Extract features for all segments"""
        if not force_reextract and self.check_if_features_exist():
            print("Features already extracted, skipping...")
            return
            
        print("\n" + "="*60)
        print("EXTRACTING FEATURES")
        print("="*60)
        
        # Ensure Whisper model is on correct device
        print(f"Moving Whisper model to {self.device}...")
        self.feature_extractor.whisper_model = self.feature_extractor.whisper_model.to(self.device)
        self.feature_extractor.device = self.device
        
        # Verify device placement
        for name, param in self.feature_extractor.whisper_model.named_parameters():
            print(f"Whisper parameter '{name}' on device: {param.device}")
            break  # Just check first parameter
        
        # Load metadata
        metadata_path = self.processed_path / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Processing {len(metadata)} segments...")
        
        # Process in batches for efficiency
        batch_size = 8 if self.device.type == 'cuda' else 4
        start_time = time.time()
        
        # Track progress
        processed = 0
        failed = 0
        
        # Process segments
        for i in tqdm(range(0, len(metadata), batch_size), desc="Extracting features"):
            batch_metadata = metadata[i:i+batch_size]
            
            try:
                # Load audio batch
                audio_batch = []
                for segment_data in batch_metadata:
                    audio = np.load(segment_data['audio_path'].replace('.wav', '.npy'))
                    audio_batch.append(audio)
                
                # Extract features for batch
                with torch.no_grad():
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Process each audio in batch (can't batch Whisper easily)
                    for j, (audio, segment_data) in enumerate(zip(audio_batch, batch_metadata)):
                        segment_idx = i + j
                        feature_path = self.features_path / f"features_{segment_idx:06d}.npy"
                        
                        try:
                            # Extract features
                            features = self.feature_extractor.extract_features(audio)
                            
                            # Validate shape
                            expected_shape = (self.pooled_frames, self.expected_feature_dim)
                            if features.shape != expected_shape:
                                print(f"Warning: Feature shape mismatch for segment {segment_idx}")
                                print(f"  Expected: {expected_shape}, Got: {features.shape}")
                            
                            # Save features
                            np.save(feature_path, features.astype(np.float32))
                            processed += 1
                            
                        except Exception as e:
                            print(f"Error processing segment {segment_idx}: {e}")
                            # Save zero features as fallback
                            zero_features = np.zeros((self.pooled_frames, self.expected_feature_dim), 
                                                   dtype=np.float32)
                            np.save(feature_path, zero_features)
                            failed += 1
                
                # Clear GPU cache periodically
                if self.device.type == 'cuda' and i % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing batch starting at {i}: {e}")
                failed += batch_size
        
        # Save feature info
        elapsed_time = time.time() - start_time
        feature_info = {
            'feature_dim': self.expected_feature_dim,
            'pooled_frames': self.pooled_frames,
            'whisper_model': self.config['features']['whisper_model'],
            'num_segments': len(metadata),
            'processed': processed,
            'failed': failed,
            'extraction_time': elapsed_time,
            'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device)
        }
        
        with open(self.features_path / 'feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"\nFeature extraction completed!")
        print(f"  Processed: {processed}/{len(metadata)} segments")
        print(f"  Failed: {failed}")
        print(f"  Time: {elapsed_time:.1f} seconds")
        print(f"  Average: {elapsed_time/len(metadata):.2f} seconds per segment")
        
        # Move model back to CPU to free GPU memory
        if self.device.type == 'cuda':
            self.feature_extractor.whisper_model = self.feature_extractor.whisper_model.cpu()
            torch.cuda.empty_cache()
            print("Moved Whisper model back to CPU to free GPU memory")


class PreExtractedDataset(torch.utils.data.Dataset):
    """Dataset that loads pre-extracted features instead of computing on-the-fly"""
    
    def __init__(self, config, split='train', augment=True, verbose=False):
        self.config = config
        self.split = split
        self.augment = augment and (split == 'train')
        self.verbose = verbose
        
        # Paths
        self.processed_path = Path(config['data']['processed_data_path'])
        self.features_path = self.processed_path / 'features'
        self.splits_path = Path(config['data']['splits_path'])
        
        # Expected dimensions
        self.expected_feature_dim = config['features'].get('whisper_dim', 768) + config['features']['mfcc']['n_mfcc'] * 3
        self.expected_seq_len = config['labels']['pooled_frames']
        self.num_classes = config['labels']['num_classes']
        
        # Verify features exist
        if not self.features_path.exists():
            raise RuntimeError("Pre-extracted features not found! Run feature extraction first.")
        
        # Load metadata and splits
        with open(self.processed_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
            
        with open(self.splits_path / 'splits.json', 'r') as f:
            splits = json.load(f)
            
        self.indices = splits[split]
        
        # Verify feature files exist
        missing_features = []
        for idx in self.indices[:10]:  # Check first 10
            feature_path = self.features_path / f"features_{idx:06d}.npy"
            if not feature_path.exists():
                missing_features.append(idx)
        
        if missing_features:
            raise RuntimeError(f"Missing feature files for indices: {missing_features}")
        
        print(f"PreExtractedDataset initialized for {split}:")
        print(f"  Samples: {len(self.indices)}")
        print(f"  Feature dimension: {self.expected_feature_dim}")
        print(f"  Sequence length: {self.expected_seq_len}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Load pre-extracted features and labels"""
        segment_idx = self.indices[idx]
        
        # Load pre-extracted features
        feature_path = self.features_path / f"features_{segment_idx:06d}.npy"
        try:
            features = np.load(feature_path)
            
            # Validate shape
            if features.shape != (self.expected_seq_len, self.expected_feature_dim):
                if self.verbose:
                    print(f"Feature shape mismatch for {segment_idx}: {features.shape}")
                # Fix shape if needed
                fixed_features = np.zeros((self.expected_seq_len, self.expected_feature_dim), dtype=np.float32)
                min_seq = min(features.shape[0], self.expected_seq_len)
                min_feat = min(features.shape[1], self.expected_feature_dim)
                fixed_features[:min_seq, :min_feat] = features[:min_seq, :min_feat]
                features = fixed_features
                
        except Exception as e:
            print(f"Error loading features for segment {segment_idx}: {e}")
            features = np.zeros((self.expected_seq_len, self.expected_feature_dim), dtype=np.float32)
        
        # Load labels
        segment_data = self.metadata[segment_idx]
        try:
            labels = np.load(segment_data['label_path'])
            # Pool labels to match feature frames
            labels = self._pool_labels(labels)
        except Exception as e:
            print(f"Error loading labels for segment {segment_idx}: {e}")
            labels = np.zeros((self.expected_seq_len, self.num_classes), dtype=np.float32)
        
        # Apply augmentations to labels if needed
        if self.augment:
            # Simple label augmentation (e.g., label smoothing)
            smoothing = 0.1
            labels = labels * (1 - smoothing) + smoothing / self.num_classes
        
        return (torch.tensor(features, dtype=torch.float32), 
                torch.tensor(labels, dtype=torch.float32))
    
    def _pool_labels(self, labels: np.ndarray) -> np.ndarray:
        """Pool labels to match feature frames"""
        current_frames = labels.shape[0]
        target_frames = self.expected_seq_len
        
        if current_frames == target_frames:
            return labels
            
        # Simple pooling
        if current_frames > target_frames:
            # Downsample
            indices = np.linspace(0, current_frames - 1, target_frames).astype(int)
            return labels[indices]
        else:
            # Upsample
            indices = np.linspace(0, current_frames - 1, target_frames).astype(int)
            return labels[indices]


def create_fast_dataloaders(config, num_workers=4):
    """Create data loaders using pre-extracted features"""
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = PreExtractedDataset(
                config, 
                split=split,
                augment=(split == 'train')
            )
            
            loaders[split] = torch.utils.data.DataLoader(
                dataset,
                batch_size=config['training']['batch_size'],
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(num_workers > 0) and (split == 'train'),
                prefetch_factor=2 if num_workers > 0 else None,
                drop_last=(split == 'train')
            )
            
            print(f"Created {split} dataloader with {len(dataset)} samples")
            
        except Exception as e:
            print(f"Error creating {split} dataloader: {e}")
            raise
    
    return loaders['train'], loaders['val'], loaders['test']


# Integration functions for main.py and train.py

def run_feature_extraction(config, device=None):
    """Run feature extraction before training"""
    from src.feature_extraction import FeatureExtractor
    
    print("\n" + "="*60)
    print("FEATURE PRE-EXTRACTION")
    print("="*60)
    
    # Get device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check GPU memory before starting
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {allocated:.2f}/{total:.2f} GB allocated")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(config, device=device, verbose=True)
    
    # Initialize preprocessor
    preprocessor = FeaturePreprocessor(config, feature_extractor, device=device)
    
    # Extract features
    preprocessor.extract_all_features(force_reextract=False)
    
    # Clean up
    del feature_extractor
    del preprocessor
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    print("\nFeature extraction completed!")
    return True


def verify_gpu_usage():
    """Verify that GPU is being used properly"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available!")
        return False
        
    # Create a test tensor and operation
    test_tensor = torch.randn(1000, 1000).cuda()
    result = torch.matmul(test_tensor, test_tensor)
    
    # Check memory usage
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"GPU test successful. Memory allocated: {allocated:.3f} GB")
    
    # Clean up
    del test_tensor, result
    torch.cuda.empty_cache()
    
    return True