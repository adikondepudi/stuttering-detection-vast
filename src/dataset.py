import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import librosa
import random
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import soundfile as sf


class StutterDataset(Dataset):
    def __init__(self, config, split='train', feature_extractor=None, augment=True):
        self.config = config
        self.split = split
        self.augment = augment and (split == 'train')
        self.feature_extractor = feature_extractor
        
        # Load metadata and splits
        processed_path = Path(config['data']['processed_data_path'])
        splits_path = Path(config['data']['splits_path'])
        
        with open(processed_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        with open(splits_path / 'splits.json', 'r') as f:
            splits = json.load(f)
        
        self.indices = splits[split]
        self.sample_rate = config['data']['sample_rate']
        
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
        segment_idx = self.indices[idx]
        segment_data = self.metadata[segment_idx]
        
        # Load audio
        audio, _ = librosa.load(segment_data['audio_path'], sr=self.sample_rate)
        
        # Load labels
        labels = np.load(segment_data['label_path'])
        
        # Apply augmentations
        if self.augment:
            audio, labels = self._apply_augmentations(audio, labels)
        
        # Extract features if extractor provided
        if self.feature_extractor:
            features = self.feature_extractor.extract_features(audio)
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = torch.tensor(audio, dtype=torch.float32)
        
        # Pool labels to match feature frames
        pooled_labels = self._pool_labels(labels)
        labels_tensor = torch.tensor(pooled_labels, dtype=torch.float32)
        
        return features, labels_tensor
    
    def _pool_labels(self, labels: np.ndarray) -> np.ndarray:
        """Pool labels to match pooled feature frames"""
        target_frames = self.config['labels']['pooled_frames']
        current_frames = labels.shape[0]
        # Guard against division by zero
        if target_frames <= 0 or current_frames <= 0:
            return np.zeros((max(1, target_frames), labels.shape[1] if len(labels.shape) > 1 else 1), dtype=labels.dtype)
        # Same pooling logic as features
        pool_size = current_frames // target_frames
        remainder = current_frames % target_frames
        
        pooled = []
        start_idx = 0
        
        for i in range(target_frames):
            end_idx = start_idx + pool_size + (1 if i < remainder else 0)
            # Use max pooling for labels (if any frame has the label, pooled frame has it)
            pooled.append(labels[start_idx:end_idx].max(axis=0))
            start_idx = end_idx
        
        return np.array(pooled)
    
    def _apply_augmentations(self, audio: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentations"""
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
        
        return audio, labels
    
    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add background noise at random SNR"""
        if not self.noise_samples:
            return audio
        
        # Select random noise
        noise = random.choice(self.noise_samples)
        
        # Match lengths
        if len(noise) < len(audio):
            noise = np.tile(noise, (len(audio) // len(noise) + 1))[:len(audio)]
        else:
            start = random.randint(0, len(noise) - len(audio))
            noise = noise[start:start + len(audio)]
        
        # Calculate SNR
        snr = random.uniform(*self.noise_snr_range)
        
        # Add noise
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        noise_scaling = np.sqrt(signal_power / (noise_power * (10 ** (snr / 10))))
        
        return audio + noise_scaling * noise
    
    def _apply_speed_perturbation(self, audio: np.ndarray) -> np.ndarray:
        """Apply speed perturbation"""
        speed_factor = random.choice(self.speed_factors)
        
        # Resample to change speed
        resampled = librosa.effects.time_stretch(audio, rate=speed_factor)
        
        # Ensure we maintain the original length
        target_length = len(audio)
        if len(resampled) > target_length:
            resampled = resampled[:target_length]
        else:
            resampled = np.pad(resampled, (0, target_length - len(resampled)), mode='constant')
        
        return resampled
    
    def _insert_repetition(self, audio: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Insert artificial repetition"""
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
        
        # Find suitable start position
        valid_starts = [i for i in fluent_indices 
                       if i + duration_frames < len(labels) and i + duration_frames >= 0
                       and fluent_mask[i:i+duration_frames].all()]
        
        if not valid_starts:
            return audio, labels
        
        start_frame = random.choice(valid_starts)
        start_sample = int(start_frame * frames_to_samples)
        end_sample = int((start_frame + duration_frames) * frames_to_samples)
        
        # Extract segment to repeat
        segment = audio[start_sample:end_sample]
        
        # Insert repetition
        audio = np.concatenate([
            audio[:end_sample],
            segment,
            audio[end_sample:]
        ])
        
        # Trim to original length
        audio = audio[:len(audio)]
        
        # Update labels (mark as repetition)
        repetition_type = 2 if duration_frames > 10 else 3  # Word vs Sound repetition
        labels[start_frame:start_frame+duration_frames, repetition_type] = 1
        
        return audio, labels
    
    def _load_noise_samples(self) -> List[np.ndarray]:
        """Load noise samples for augmentation"""
        noise_samples = []
        noise_dir = Path('data/noise')  # Assume noise samples are here
        
        if noise_dir.exists():
            for noise_file in noise_dir.glob('*.wav'):
                try:
                    noise, _ = librosa.load(noise_file, sr=self.sample_rate)
                    noise_samples.append(noise)
                except Exception as e:
                    print(f"Error loading noise file {noise_file}: {e}")
        
        return noise_samples


def create_dataloaders(config, feature_extractor=None, num_workers=4):
    """Create data loaders for train, val, and test sets"""
    train_dataset = StutterDataset(config, 'train', feature_extractor, augment=True)
    val_dataset = StutterDataset(config, 'val', feature_extractor, augment=False)
    test_dataset = StutterDataset(config, 'test', feature_extractor, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader