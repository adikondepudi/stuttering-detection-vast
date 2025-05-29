import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import librosa
import random
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')


class StutterDataset(Dataset):
    def __init__(self, config, split='train', feature_extractor=None, augment=True, verbose=False):
        self.config = config
        self.split = split
        self.augment = augment and (split == 'train')
        self.feature_extractor = feature_extractor
        self.verbose = verbose
        
        # Expected dimensions
        self.expected_samples = int(config['segmentation']['segment_length'] * config['data']['sample_rate'])
        self.expected_label_frames = config['labels']['frames_per_segment']
        self.pooled_frames = config['labels']['pooled_frames']
        self.num_classes = config['labels']['num_classes']
        
        # Get expected feature dimension from feature extractor if available
        if self.feature_extractor:
            self.expected_feature_dim = self.feature_extractor.get_feature_dim()
        else:
            # Fallback to calculation from config (but this might be wrong if whisper_dim is incorrect)
            whisper_dim = config['features'].get('whisper_dim', 512)
            n_mfcc = config['features']['mfcc']['n_mfcc']
            mfcc_total_dim = n_mfcc * 3  # base + delta + delta2
            self.expected_feature_dim = whisper_dim + mfcc_total_dim
        
        if self.verbose:
            print(f"Dataset initialized for {split} split:")
            print(f"  Expected audio samples: {self.expected_samples}")
            print(f"  Expected label frames: {self.expected_label_frames}")
            print(f"  Target pooled frames: {self.pooled_frames}")
            print(f"  Expected feature dimension: {self.expected_feature_dim}")
            if self.feature_extractor:
                print(f"  (Feature dimension from extractor)")
        
        # Load metadata and splits
        processed_path = Path(config['data']['processed_data_path'])
        splits_path = Path(config['data']['splits_path'])
        
        # Validate paths exist
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data path not found: {processed_path}")
        if not splits_path.exists():
            raise FileNotFoundError(f"Splits path not found: {splits_path}")
        
        # Load metadata
        metadata_file = processed_path / 'metadata.json'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Load splits
        splits_file = splits_path / 'splits.json'
        if not splits_file.exists():
            raise FileNotFoundError(f"Splits file not found: {splits_file}")
        
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        # Validate split exists
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in splits file. Available: {list(splits.keys())}")
        
        self.indices = splits[split]
        self.sample_rate = config['data']['sample_rate']
        
        # Validate indices
        max_idx = len(self.metadata) - 1
        invalid_indices = [idx for idx in self.indices if idx > max_idx]
        if invalid_indices:
            print(f"Warning: Found {len(invalid_indices)} invalid indices. Removing them.")
            self.indices = [idx for idx in self.indices if idx <= max_idx]
        
        if len(self.indices) == 0:
            raise ValueError(f"No valid samples found for {split} split")
        
        print(f"Loaded {len(self.indices)} samples for {split} split")
        
        # Augmentation parameters
        if self.augment:
            self.noise_prob = config['augmentation']['noise_prob']
            self.noise_snr_range = config['augmentation']['noise_snr_range']
            self.speed_prob = config['augmentation']['speed_prob']
            self.speed_factors = config['augmentation']['speed_factors']
            self.repetition_prob = config['augmentation']['repetition_prob']
            self.repetition_duration_range = config['augmentation']['repetition_duration_range']
            
            # Load noise samples if available
            self.noise_samples = self._load_noise_samples()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item with robust error handling and validation"""
        try:
            segment_idx = self.indices[idx]
            segment_data = self.metadata[segment_idx]
            
            # Load and validate audio
            audio = self._load_and_validate_audio(segment_data['audio_path'])
            
            # Load and validate labels
            labels = self._load_and_validate_labels(segment_data['label_path'])
            
            # Apply augmentations
            if self.augment:
                audio, labels = self._apply_augmentations(audio, labels)
            
            # Extract features if extractor provided
            if self.feature_extractor:
                features = self.feature_extractor.extract_features(audio)
                # Validate feature shape
                features = self._validate_features(features)
                features = torch.tensor(features, dtype=torch.float32)
            else:
                features = torch.tensor(audio, dtype=torch.float32)
            
            # Pool labels to match feature frames
            pooled_labels = self._pool_labels(labels)
            # Validate pooled labels
            pooled_labels = self._validate_pooled_labels(pooled_labels)
            labels_tensor = torch.tensor(pooled_labels, dtype=torch.float32)
            
            return features, labels_tensor
            
        except Exception as e:
            print(f"Error loading sample {idx} (segment {segment_idx}): {e}")
            # Return valid dummy data to prevent training crash
            if self.feature_extractor:
                dummy_features = torch.zeros((self.pooled_frames, self.expected_feature_dim), dtype=torch.float32)
            else:
                dummy_features = torch.zeros(self.expected_samples, dtype=torch.float32)
            dummy_labels = torch.zeros((self.pooled_frames, self.num_classes), dtype=torch.float32)
            return dummy_features, dummy_labels
    
    def _load_and_validate_audio(self, audio_path: str) -> np.ndarray:
        """Load audio with validation and automatic fixing"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading audio from {audio_path}: {e}")
            # Return silence
            return np.zeros(self.expected_samples, dtype=np.float32)
        
        # Validate and fix length
        if len(audio) != self.expected_samples:
            if self.verbose:
                print(f"Audio length mismatch in {audio_path}: {len(audio)} vs {self.expected_samples}")
            
            if len(audio) < self.expected_samples:
                # Pad with zeros
                pad_length = self.expected_samples - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant')
            else:
                # Truncate
                audio = audio[:self.expected_samples]
        
        # Check for NaN or Inf
        if not np.isfinite(audio).all():
            print(f"Warning: Non-finite values in audio from {audio_path}")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        
        return audio.astype(np.float32)
    
    def _load_and_validate_labels(self, label_path: str) -> np.ndarray:
        """Load labels with validation and automatic fixing"""
        try:
            labels = np.load(label_path)
        except Exception as e:
            print(f"Error loading labels from {label_path}: {e}")
            # Return zero labels
            return np.zeros((self.expected_label_frames, self.num_classes), dtype=np.float32)
        
        # Validate shape
        expected_shape = (self.expected_label_frames, self.num_classes)
        if labels.shape != expected_shape:
            if self.verbose:
                print(f"Label shape mismatch in {label_path}: {labels.shape} vs {expected_shape}")
            
            # Fix temporal dimension
            if labels.shape[0] != expected_shape[0]:
                if labels.shape[0] < expected_shape[0]:
                    # Pad
                    pad_frames = expected_shape[0] - labels.shape[0]
                    labels = np.pad(labels, ((0, pad_frames), (0, 0)), mode='constant')
                else:
                    # Truncate
                    labels = labels[:expected_shape[0]]
            
            # Fix class dimension
            if labels.shape[1] != expected_shape[1]:
                fixed_labels = np.zeros(expected_shape, dtype=labels.dtype)
                min_classes = min(labels.shape[1], expected_shape[1])
                fixed_labels[:labels.shape[0], :min_classes] = labels[:, :min_classes]
                labels = fixed_labels
        
        # Ensure binary labels
        labels = (labels > 0.5).astype(np.float32)
        
        return labels
    
    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        """Validate and fix feature dimensions"""
        expected_shape = (self.pooled_frames, self.expected_feature_dim)
        
        if features.shape == expected_shape:
            return features
        
        if self.verbose:
            print(f"Feature shape mismatch: {features.shape} vs {expected_shape}")
        
        # Create properly shaped array
        fixed_features = np.zeros(expected_shape, dtype=features.dtype)
        
        # Copy what we can
        min_frames = min(features.shape[0], expected_shape[0])
        min_features = min(features.shape[1], expected_shape[1]) if len(features.shape) > 1 else 0
        
        if min_frames > 0 and min_features > 0:
            fixed_features[:min_frames, :min_features] = features[:min_frames, :min_features]
        
        return fixed_features
    
    def _validate_pooled_labels(self, labels: np.ndarray) -> np.ndarray:
        """Validate and fix pooled label dimensions"""
        expected_shape = (self.pooled_frames, self.num_classes)
        
        if labels.shape == expected_shape:
            return labels
        
        if self.verbose:
            print(f"Pooled label shape mismatch: {labels.shape} vs {expected_shape}")
        
        # Create properly shaped array
        fixed_labels = np.zeros(expected_shape, dtype=labels.dtype)
        
        # Copy what we can
        min_frames = min(labels.shape[0], expected_shape[0])
        min_classes = min(labels.shape[1], expected_shape[1]) if len(labels.shape) > 1 else 0
        
        if min_frames > 0 and min_classes > 0:
            fixed_labels[:min_frames, :min_classes] = labels[:min_frames, :min_classes]
        
        return fixed_labels
    
    def _pool_labels(self, labels: np.ndarray) -> np.ndarray:
        """Pool labels to match pooled feature frames with robust handling"""
        # Validate input
        if len(labels.shape) != 2:
            print(f"Warning: Unexpected labels shape: {labels.shape}")
            return np.zeros((self.pooled_frames, self.num_classes), dtype=np.float32)
        
        target_frames = self.pooled_frames
        current_frames = labels.shape[0]
        
        # Handle edge cases
        if current_frames == 0 or target_frames <= 0:
            return np.zeros((max(1, target_frames), self.num_classes), dtype=np.float32)
        
        if current_frames == target_frames:
            return labels
        
        if current_frames < target_frames:
            # Interpolate or repeat
            if current_frames == 1:
                # Repeat single frame
                return np.repeat(labels, target_frames, axis=0)
            else:
                # Use nearest neighbor interpolation for labels
                indices = np.round(np.linspace(0, current_frames - 1, target_frames)).astype(int)
                return labels[indices]
        
        # Standard pooling case (more frames than target)
        # Calculate adaptive pool sizes
        pool_sizes = self._calculate_adaptive_pool_sizes(current_frames, target_frames)
        
        pooled = []
        start_idx = 0
        
        for pool_size in pool_sizes:
            end_idx = start_idx + pool_size
            if end_idx > current_frames:
                end_idx = current_frames
            
            if start_idx < end_idx:
                # Max pooling for labels (if any frame has the label, pooled frame has it)
                pooled_frame = labels[start_idx:end_idx].max(axis=0)
            else:
                # Fallback: use last valid frame
                pooled_frame = labels[min(start_idx, current_frames - 1)]
            
            pooled.append(pooled_frame)
            start_idx = end_idx
        
        pooled_array = np.array(pooled)
        
        # Ensure we have exactly target_frames
        if len(pooled_array) != target_frames:
            if self.verbose:
                print(f"Warning: Label pooling produced {len(pooled_array)} frames, expected {target_frames}")
            # Fix by repeating or truncating
            if len(pooled_array) < target_frames:
                repeats = target_frames - len(pooled_array)
                pooled_array = np.vstack([pooled_array] + [pooled_array[-1:]] * repeats)
            else:
                pooled_array = pooled_array[:target_frames]
        
        return pooled_array.astype(np.float32)
    
    def _calculate_adaptive_pool_sizes(self, current_frames: int, target_frames: int) -> List[int]:
        """Calculate adaptive pool sizes for even distribution"""
        base_pool_size = current_frames // target_frames
        remainder = current_frames % target_frames
        
        pool_sizes = []
        for i in range(target_frames):
            if i < remainder:
                pool_sizes.append(base_pool_size + 1)
            else:
                pool_sizes.append(base_pool_size)
        
        return pool_sizes
    
    def _apply_augmentations(self, audio: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentations with error handling"""
        try:
            # Additive noise
            if random.random() < self.noise_prob and self.noise_samples:
                audio = self._add_noise(audio)
            
            # Speed perturbation
            if random.random() < self.speed_prob:
                audio = self._apply_speed_perturbation(audio)
            
            # Repetition insertion (only if no existing repetitions)
            if random.random() < self.repetition_prob:
                has_repetition = labels[:, 2].any() or labels[:, 3].any()  # Word/Sound repetition
                if not has_repetition:
                    audio, labels = self._insert_repetition(audio, labels)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Augmentation failed: {e}")
            # Return original data if augmentation fails
        
        return audio, labels
    
    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add background noise at random SNR"""
        if not self.noise_samples:
            return audio
        
        try:
            # Select random noise
            noise = random.choice(self.noise_samples)
            
            # Match lengths
            if len(noise) < len(audio):
                repeats = (len(audio) // len(noise)) + 1
                noise = np.tile(noise, repeats)[:len(audio)]
            else:
                start = random.randint(0, max(0, len(noise) - len(audio)))
                noise = noise[start:start + len(audio)]
            
            # Calculate SNR
            snr = random.uniform(*self.noise_snr_range)
            
            # Add noise
            signal_power = np.mean(audio ** 2) + 1e-10  # Avoid division by zero
            noise_power = np.mean(noise ** 2) + 1e-10
            noise_scaling = np.sqrt(signal_power / (noise_power * (10 ** (snr / 10))))
            
            noisy_audio = audio + noise_scaling * noise
            
            # Normalize if needed
            max_val = np.abs(noisy_audio).max()
            if max_val > 1.0:
                noisy_audio = noisy_audio / max_val
            
            return noisy_audio
            
        except Exception as e:
            if self.verbose:
                print(f"Error adding noise: {e}")
            return audio
    
    def _apply_speed_perturbation(self, audio: np.ndarray) -> np.ndarray:
        """Apply speed perturbation"""
        try:
            speed_factor = random.choice(self.speed_factors)
            
            # Resample to change speed
            resampled = librosa.effects.time_stretch(audio, rate=speed_factor)
            
            # Ensure we maintain the original length
            target_length = len(audio)
            if len(resampled) > target_length:
                resampled = resampled[:target_length]
            else:
                pad_length = target_length - len(resampled)
                resampled = np.pad(resampled, (0, pad_length), mode='edge')
            
            return resampled.astype(np.float32)
            
        except Exception as e:
            if self.verbose:
                print(f"Error applying speed perturbation: {e}")
            return audio
    
    def _insert_repetition(self, audio: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Insert artificial repetition"""
        try:
            # Find fluent speech regions (no labels)
            fluent_mask = labels.sum(axis=1) == 0
            
            if not fluent_mask.any():
                return audio, labels
            
            # Select random fluent segment
            fluent_indices = np.where(fluent_mask)[0]
            
            # Convert frame indices to sample indices
            frames_to_samples = len(audio) / len(labels)
            
            # Select duration
            duration_frames = int(random.uniform(*self.repetition_duration_range) * 
                                len(labels) / (len(audio) / self.sample_rate))
            duration_frames = max(1, min(duration_frames, len(labels) // 4))  # Limit duration
            
            # Find suitable start position
            valid_starts = []
            for i in fluent_indices:
                if (i + duration_frames < len(labels) and 
                    i + duration_frames >= 0 and
                    fluent_mask[i:i+duration_frames].all()):
                    valid_starts.append(i)
            
            if not valid_starts:
                return audio, labels
            
            start_frame = random.choice(valid_starts)
            start_sample = int(start_frame * frames_to_samples)
            end_sample = int((start_frame + duration_frames) * frames_to_samples)
            
            # Ensure valid sample indices
            start_sample = max(0, min(start_sample, len(audio) - 1))
            end_sample = max(start_sample + 1, min(end_sample, len(audio)))
            
            # Extract segment to repeat
            segment = audio[start_sample:end_sample]
            
            if len(segment) == 0:
                return audio, labels
            
            # Insert repetition (don't exceed original length)
            if end_sample + len(segment) <= len(audio):
                # Simple insertion within bounds
                audio_with_rep = audio.copy()
                audio_with_rep[end_sample:end_sample+len(segment)] = segment
                audio = audio_with_rep
            
            # Update labels (mark as repetition)
            repetition_type = 2 if duration_frames > 10 else 3  # Word vs Sound repetition
            labels[start_frame:start_frame+duration_frames, repetition_type] = 1
            
            return audio, labels
            
        except Exception as e:
            if self.verbose:
                print(f"Error inserting repetition: {e}")
            return audio, labels
    
    def _load_noise_samples(self) -> List[np.ndarray]:
        """Load noise samples for augmentation"""
        noise_samples = []
        noise_dir = Path('data/noise')  # Assume noise samples are here
        
        if noise_dir.exists():
            noise_files = list(noise_dir.glob('*.wav')) + list(noise_dir.glob('*.mp3'))
            
            for noise_file in noise_files[:10]:  # Limit number of noise files
                try:
                    noise, _ = librosa.load(noise_file, sr=self.sample_rate, mono=True)
                    if len(noise) > 0:
                        noise_samples.append(noise.astype(np.float32))
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading noise file {noise_file}: {e}")
        
        if self.verbose:
            print(f"Loaded {len(noise_samples)} noise samples")
        
        return noise_samples


def create_dataloaders(config, feature_extractor=None, num_workers=4, verbose=False):
    """Create data loaders for train, val, and test sets with error handling"""
    # Adjust num_workers based on system
    try:
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
        num_workers = min(num_workers, max_workers)
    except:
        num_workers = 0  # Fallback to main thread
    
    # Create datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        try:
            datasets[split] = StutterDataset(
                config, 
                split, 
                feature_extractor, 
                augment=(split == 'train'),
                verbose=verbose
            )
            print(f"Created {split} dataset with {len(datasets[split])} samples")
        except Exception as e:
            print(f"Error creating {split} dataset: {e}")
            # Create dummy dataset
            datasets[split] = None
    
    # Create dataloaders
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        if datasets[split] is not None:
            try:
                loaders[split] = DataLoader(
                    datasets[split],
                    batch_size=config['training']['batch_size'],
                    shuffle=(split == 'train'),
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=(num_workers > 0),
                    prefetch_factor=2 if num_workers > 0 else None,
                    drop_last=(split == 'train')  # Drop incomplete batches in training
                )
            except Exception as e:
                print(f"Error creating {split} dataloader: {e}")
                # Fallback to simpler dataloader
                loaders[split] = DataLoader(
                    datasets[split],
                    batch_size=config['training']['batch_size'],
                    shuffle=(split == 'train'),
                    num_workers=0  # Use main thread
                )
        else:
            # Create dummy dataloader
            print(f"Warning: Using dummy dataloader for {split}")
            loaders[split] = None
    
    # Validate dataloaders
    for split, loader in loaders.items():
        if loader is not None:
            try:
                # Test loading one batch
                batch = next(iter(loader))
                features, labels = batch
                print(f"{split} dataloader test - Features: {features.shape}, Labels: {labels.shape}")
            except Exception as e:
                print(f"Error testing {split} dataloader: {e}")
    
    return loaders['train'], loaders['val'], loaders['test']


# Utility function for testing
def test_dataset(config_path='config/config.yaml'):
    """Test dataset loading and dimensions"""
    import yaml
    
    print("Testing dataset...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create feature extractor
    from .feature_extraction import FeatureExtractor
    feature_extractor = FeatureExtractor(config, device='cpu', verbose=True)
    
    # Create dataset
    dataset = StutterDataset(config, 'train', feature_extractor, augment=False, verbose=True)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading samples
    print("\nTesting sample loading...")
    for i in range(min(5, len(dataset))):
        try:
            features, labels = dataset[i]
            print(f"Sample {i}: Features {features.shape}, Labels {labels.shape}")
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
    
    # Test dataloader
    print("\nTesting dataloader...")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    for i, (features, labels) in enumerate(dataloader):
        print(f"Batch {i}: Features {features.shape}, Labels {labels.shape}")
        if i >= 2:  # Test first 3 batches
            break
    
    print("\nDataset test complete!")


if __name__ == "__main__":
    test_dataset()