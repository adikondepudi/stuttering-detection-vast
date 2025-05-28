import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import WhisperModel, WhisperFeatureExtractor
from typing import Tuple, Optional, List
from pathlib import Path


class FeatureExtractor:
    def __init__(self, config, device='cuda'):
        self.config = config
        # Fix device handling
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
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
        
        # Pooling parameters
        self.pooled_frames = config['labels']['pooled_frames']
        self.whisper_pool_size = config['labels']['frames_per_segment'] // self.pooled_frames
        
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract fused features from audio segment"""
        # Extract Whisper features
        whisper_features = self._extract_whisper_features(audio)
        
        # Extract MFCC features
        mfcc_features = self._extract_mfcc_features(audio)
        
        # Fuse features
        fused_features = np.concatenate([whisper_features, mfcc_features], axis=1)
        
        return fused_features
    
    def _extract_whisper_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features using Whisper encoder"""
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
            
            # Apply mean pooling
            pooled_features = self._apply_temporal_pooling(
                hidden_states, 
                target_frames=self.pooled_frames
            )
            
        return pooled_features
    
    def _extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features with deltas"""
        # Extract base MFCCs
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
        
        # Apply pooling to match Whisper temporal resolution
        pooled_mfcc = self._apply_temporal_pooling(
            mfcc_features,
            target_frames=self.pooled_frames
        )
        
        return pooled_mfcc
    
    def _apply_temporal_pooling(self, features: np.ndarray, target_frames: int) -> np.ndarray:
        """Apply mean pooling to match target frame count"""
        current_frames = features.shape[0]
        # Guard against invalid inputs
        if target_frames <= 0 or current_frames <= 0:
            return np.zeros((max(1, target_frames), features.shape[1]), dtype=features.dtype)
        if current_frames <= target_frames:
            # Pad if necessary
            pad_frames = target_frames - current_frames
            if pad_frames > 0:
                features = np.pad(features, ((0, pad_frames), (0, 0)), mode='edge')
            return features[:target_frames]
        # Calculate pool size
        pool_size = current_frames // target_frames
        remainder = current_frames % target_frames
        
        pooled = []
        start_idx = 0
        
        for i in range(target_frames):
            # Distribute remainder frames
            end_idx = start_idx + pool_size + (1 if i < remainder else 0)
            pooled.append(features[start_idx:end_idx].mean(axis=0))
            start_idx = end_idx
        
        return np.array(pooled)
    
    def extract_batch_features(self, audio_batch: List[np.ndarray]) -> torch.Tensor:
        """Extract features for a batch of audio segments"""
        features_list = []
        
        for audio in audio_batch:
            features = self.extract_features(audio)
            features_list.append(features)
        
        # Convert to tensor
        features_tensor = torch.tensor(
            np.array(features_list), 
            dtype=torch.float32
        )
        
        return features_tensor


class CachedFeatureExtractor(FeatureExtractor):
    """Feature extractor with caching capability"""
    
    def __init__(self, config, device='cuda', cache_dir=None):
        super().__init__(config, device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_features_cached(self, audio_path: str, segment_idx: int) -> np.ndarray:
        """Extract features with caching"""
        if self.cache_dir:
            cache_path = self.cache_dir / f"{Path(audio_path).stem}_seg{segment_idx}.npy"
            
            if cache_path.exists():
                return np.load(cache_path)
        
        # Load audio and extract features
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        features = self.extract_features(audio)
        
        # Save to cache
        if self.cache_dir:
            np.save(cache_path, features)
        
        return features