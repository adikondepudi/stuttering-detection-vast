import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import WhisperModel, WhisperFeatureExtractor
from typing import Tuple, Optional, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    def __init__(self, config, device='cuda', verbose=False):
        self.config = config
        self.verbose = verbose
        
        # Fix device handling
        if isinstance(device, str):
            if device == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Initialize Whisper
        self.whisper_model = WhisperModel.from_pretrained(
            config['features']['whisper_model']
        ).to(self.device)
        self.whisper_model.eval()
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
            config['features']['whisper_model']
        )
        
        # MFCC parameters
        self.n_mfcc = config['features']['mfcc']['n_mfcc']
        self.n_fft = config['features']['mfcc']['n_fft']
        self.hop_length = config['features']['mfcc']['hop_length']
        self.sample_rate = config['data']['sample_rate']
        
        # Target dimensions
        self.pooled_frames = config['labels']['pooled_frames']
        self.whisper_dim = config['features'].get('whisper_dim', 768)
        self.mfcc_total_dim = self.n_mfcc * 3  # base + delta + delta2
        self.expected_feature_dim = self.whisper_dim + self.mfcc_total_dim
        
        # Expected audio length
        self.segment_length = config['segmentation']['segment_length']
        self.expected_samples = int(self.segment_length * self.sample_rate)
        
        if self.verbose:
            print(f"FeatureExtractor initialized:")
            print(f"  Expected audio samples: {self.expected_samples}")
            print(f"  Target pooled frames: {self.pooled_frames}")
            print(f"  Expected feature dimension: {self.expected_feature_dim}")
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract fused features from audio segment with robust error handling"""
        # Validate and fix audio input
        audio = self._validate_and_fix_audio(audio)
        
        # Extract features with error handling
        try:
            whisper_features = self._extract_whisper_features(audio)
        except Exception as e:
            print(f"Warning: Whisper extraction failed: {e}")
            print("Using zero features for Whisper")
            whisper_features = np.zeros((self.pooled_frames, self.whisper_dim))
        
        try:
            mfcc_features = self._extract_mfcc_features(audio)
        except Exception as e:
            print(f"Warning: MFCC extraction failed: {e}")
            print("Using zero features for MFCC")
            mfcc_features = np.zeros((self.pooled_frames, self.mfcc_total_dim))
        
        # Validate shapes before fusion
        whisper_features = self._validate_feature_shape(
            whisper_features, 
            expected_shape=(self.pooled_frames, self.whisper_dim),
            feature_name="Whisper"
        )
        
        mfcc_features = self._validate_feature_shape(
            mfcc_features,
            expected_shape=(self.pooled_frames, self.mfcc_total_dim),
            feature_name="MFCC"
        )
        
        # Fuse features
        fused_features = np.concatenate([whisper_features, mfcc_features], axis=1)
        
        # Final validation
        if fused_features.shape != (self.pooled_frames, self.expected_feature_dim):
            print(f"Warning: Unexpected fused feature shape: {fused_features.shape}")
            print(f"Expected: ({self.pooled_frames}, {self.expected_feature_dim})")
            # Attempt to fix
            fused_features = self._fix_feature_shape(
                fused_features,
                target_shape=(self.pooled_frames, self.expected_feature_dim)
            )
        
        return fused_features
    
    def _validate_and_fix_audio(self, audio: np.ndarray) -> np.ndarray:
        """Validate and fix audio input to expected length"""
        if len(audio) == self.expected_samples:
            return audio
        
        if self.verbose:
            print(f"Audio length mismatch: got {len(audio)}, expected {self.expected_samples}")
        
        if len(audio) < self.expected_samples:
            # Pad with zeros if too short
            pad_length = self.expected_samples - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
            if self.verbose:
                print(f"Padded audio with {pad_length} zeros")
        else:
            # Truncate if too long
            audio = audio[:self.expected_samples]
            if self.verbose:
                print(f"Truncated audio to {self.expected_samples} samples")
        
        return audio
    
    def _extract_whisper_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features using Whisper encoder with robust handling"""
        with torch.no_grad():
            # Prepare input
            inputs = self.whisper_feature_extractor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)
            
            # Get encoder outputs
            encoder_outputs = self.whisper_model.encoder(
                input_features=input_features,
                output_hidden_states=True
            )
            
            # Get last hidden state
            hidden_states = encoder_outputs.last_hidden_state[0].cpu().numpy()
            
            if self.verbose:
                print(f"Whisper raw output shape: {hidden_states.shape}")
            
            # Apply temporal pooling with validation
            pooled_features = self._apply_temporal_pooling(
                hidden_states, 
                target_frames=self.pooled_frames,
                feature_name="Whisper"
            )
            
            if self.verbose:
                print(f"Whisper pooled shape: {pooled_features.shape}")
        
        return pooled_features
    
    def _extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features with deltas and robust handling"""
        # Extract base MFCCs
        try:
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window='hann'
            )
            
            # Calculate deltas
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Stack features
            mfcc_features = np.vstack([mfccs, mfcc_delta, mfcc_delta2]).T
            
            if self.verbose:
                print(f"MFCC raw shape: {mfcc_features.shape}")
            
        except Exception as e:
            print(f"Error in MFCC extraction: {e}")
            # Create dummy features with expected shape
            expected_frames = self._calculate_mfcc_frames(len(audio))
            mfcc_features = np.zeros((expected_frames, self.mfcc_total_dim))
        
        # Apply pooling to match target resolution
        pooled_mfcc = self._apply_temporal_pooling(
            mfcc_features,
            target_frames=self.pooled_frames,
            feature_name="MFCC"
        )
        
        if self.verbose:
            print(f"MFCC pooled shape: {pooled_mfcc.shape}")
        
        return pooled_mfcc
    
    def _calculate_mfcc_frames(self, audio_length: int) -> int:
        """Calculate expected number of MFCC frames"""
        return max(1, (audio_length - self.n_fft) // self.hop_length + 1)
    
    def _apply_temporal_pooling(self, features: np.ndarray, target_frames: int, 
                               feature_name: str = "") -> np.ndarray:
        """Apply mean pooling to match target frame count with robust handling"""
        # Validate inputs
        if target_frames <= 0:
            raise ValueError(f"Invalid target_frames: {target_frames}")
        
        if features.size == 0:
            feature_dim = features.shape[1] if len(features.shape) > 1 else 1
            return np.zeros((target_frames, feature_dim), dtype=features.dtype)
        
        current_frames = features.shape[0]
        
        if self.verbose:
            print(f"Pooling {feature_name} from {current_frames} to {target_frames} frames")
        
        # Handle edge cases
        if current_frames == 0:
            return np.zeros((target_frames, features.shape[1]), dtype=features.dtype)
        
        if current_frames == target_frames:
            return features
        
        if current_frames < target_frames:
            # Pad or interpolate if too few frames
            if current_frames == 1:
                # Repeat single frame
                return np.repeat(features, target_frames, axis=0)
            else:
                # Linear interpolation
                indices = np.linspace(0, current_frames - 1, target_frames)
                pooled = []
                for idx in indices:
                    if idx == int(idx):
                        pooled.append(features[int(idx)])
                    else:
                        # Interpolate between frames
                        lower = int(np.floor(idx))
                        upper = int(np.ceil(idx))
                        weight = idx - lower
                        interpolated = (1 - weight) * features[lower] + weight * features[upper]
                        pooled.append(interpolated)
                return np.array(pooled)
        
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
                # Mean pooling
                pooled_frame = features[start_idx:end_idx].mean(axis=0)
            else:
                # Fallback: use last valid frame
                pooled_frame = features[min(start_idx, current_frames - 1)]
            
            pooled.append(pooled_frame)
            start_idx = end_idx
        
        pooled_array = np.array(pooled)
        
        # Ensure we have exactly target_frames
        if len(pooled_array) != target_frames:
            if self.verbose:
                print(f"Warning: Pooling produced {len(pooled_array)} frames, expected {target_frames}")
            # Adjust by repeating or removing frames
            if len(pooled_array) < target_frames:
                # Repeat last frame
                repeats = target_frames - len(pooled_array)
                pooled_array = np.vstack([pooled_array] + [pooled_array[-1:]] * repeats)
            else:
                # Truncate
                pooled_array = pooled_array[:target_frames]
        
        return pooled_array
    
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
    
    def _validate_feature_shape(self, features: np.ndarray, expected_shape: Tuple[int, int], 
                               feature_name: str = "") -> np.ndarray:
        """Validate and fix feature shape if necessary"""
        if features.shape == expected_shape:
            return features
        
        print(f"Warning: {feature_name} shape mismatch. Got {features.shape}, expected {expected_shape}")
        
        # Fix temporal dimension
        if features.shape[0] != expected_shape[0]:
            features = self._apply_temporal_pooling(
                features, 
                target_frames=expected_shape[0],
                feature_name=feature_name
            )
        
        # Fix feature dimension
        if features.shape[1] != expected_shape[1]:
            if features.shape[1] < expected_shape[1]:
                # Pad with zeros
                pad_width = expected_shape[1] - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Truncate
                features = features[:, :expected_shape[1]]
        
        return features
    
    def _fix_feature_shape(self, features: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Force features to target shape as last resort"""
        if features.shape == target_shape:
            return features
        
        # Create output array
        fixed_features = np.zeros(target_shape, dtype=features.dtype)
        
        # Copy what we can
        min_frames = min(features.shape[0], target_shape[0])
        min_features = min(features.shape[1], target_shape[1])
        
        fixed_features[:min_frames, :min_features] = features[:min_frames, :min_features]
        
        return fixed_features
    
    def extract_batch_features(self, audio_batch: List[np.ndarray]) -> torch.Tensor:
        """Extract features for a batch of audio segments"""
        features_list = []
        
        for i, audio in enumerate(audio_batch):
            try:
                features = self.extract_features(audio)
                features_list.append(features)
            except Exception as e:
                print(f"Error extracting features for sample {i}: {e}")
                # Use zero features as fallback
                fallback_features = np.zeros((self.pooled_frames, self.expected_feature_dim))
                features_list.append(fallback_features)
        
        # Convert to tensor with validation
        features_array = np.array(features_list)
        
        # Ensure consistent shape
        expected_batch_shape = (len(audio_batch), self.pooled_frames, self.expected_feature_dim)
        if features_array.shape != expected_batch_shape:
            print(f"Warning: Batch shape mismatch. Got {features_array.shape}, expected {expected_batch_shape}")
            # Fix shape
            fixed_array = np.zeros(expected_batch_shape, dtype=features_array.dtype)
            min_batch = min(features_array.shape[0], expected_batch_shape[0])
            min_frames = min(features_array.shape[1], expected_batch_shape[1])
            min_features = min(features_array.shape[2], expected_batch_shape[2])
            fixed_array[:min_batch, :min_frames, :min_features] = \
                features_array[:min_batch, :min_frames, :min_features]
            features_array = fixed_array
        
        features_tensor = torch.tensor(features_array, dtype=torch.float32)
        
        return features_tensor


class CachedFeatureExtractor(FeatureExtractor):
    """Feature extractor with caching capability"""
    
    def __init__(self, config, device='cuda', cache_dir=None, verbose=False):
        super().__init__(config, device, verbose)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_features_cached(self, audio_path: str, segment_idx: int) -> np.ndarray:
        """Extract features with caching"""
        if self.cache_dir:
            cache_path = self.cache_dir / f"{Path(audio_path).stem}_seg{segment_idx}.npy"
            
            if cache_path.exists():
                try:
                    cached_features = np.load(cache_path)
                    # Validate cached features
                    if cached_features.shape == (self.pooled_frames, self.expected_feature_dim):
                        return cached_features
                    else:
                        print(f"Invalid cached features shape: {cached_features.shape}")
                except Exception as e:
                    print(f"Error loading cached features: {e}")
        
        # Load audio and extract features
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading audio from {audio_path}: {e}")
            # Return zero features
            return np.zeros((self.pooled_frames, self.expected_feature_dim))
        
        features = self.extract_features(audio)
        
        # Save to cache
        if self.cache_dir:
            try:
                np.save(cache_path, features)
            except Exception as e:
                print(f"Warning: Could not save features to cache: {e}")
        
        return features