import os
import json
import csv
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import pyloudnorm as pyln
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
import zipfile
import gdown
import tempfile
import shutil
warnings.filterwarnings('ignore')

# Cloud storage imports
try:
    import boto3
    from botocore.exceptions import NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from google.cloud import storage as gcs
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.raw_path = Path(config['data']['raw_data_path'])
        self.processed_path = Path(config['data']['processed_data_path'])
        self.splits_path = Path(config['data']['splits_path'])
        
        # Create directories
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.splits_path.mkdir(parents=True, exist_ok=True)
        
        self.sample_rate = config['data']['sample_rate']
        self.segment_length = config['segmentation']['segment_length']
        self.hop_length = config['segmentation']['hop_length']
        self.disfluency_types = config['labels']['disfluency_types']
        self.frames_per_segment = config['labels']['frames_per_segment']
        
        # Loudness meter
        self.meter = pyln.Meter(self.sample_rate)
        
        # Cloud storage configuration
        self.cloud_config = config.get('cloud', {})
        self.use_cloud_storage = self.cloud_config.get('enabled', False)
        
    def download_data_from_cloud(self):
        """Download data from cloud storage"""
        if not self.use_cloud_storage:
            print("Cloud storage not enabled, skipping download...")
            return
            
        storage_type = self.cloud_config.get('type', 'gdrive')
        print(f"Downloading data from {storage_type}...")
        
        try:
            if storage_type == 'gdrive':
                self._download_from_gdrive()
            elif storage_type == 's3':
                self._download_from_s3()
            elif storage_type == 'gcs':
                self._download_from_gcs()
            else:
                print(f"Unknown storage type: {storage_type}")
                return False
                
            print("Data download completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False
    
    def _download_from_gdrive(self):
        """Download from Google Drive"""
        file_id = self.cloud_config.get('gdrive_file_id')
        if not file_id or file_id == 'your_google_drive_file_id_here':
            raise ValueError("Please set your Google Drive file ID in config.yaml")
        
        # Create raw data directory
        self.raw_path.mkdir(parents=True, exist_ok=True)
        
        # Download file
        output_path = self.raw_path / 'dataset.zip'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        print(f"Downloading from Google Drive: {file_id}")
        gdown.download(url, str(output_path), quiet=False)
        
        # Extract if it's a zip file
        if output_path.suffix == '.zip':
            print("Extracting downloaded archive...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_path)
            output_path.unlink()  # Remove zip file
            print("Archive extracted successfully!")
    
    def _download_from_s3(self):
        """Download from AWS S3"""
        if not HAS_BOTO3:
            raise ImportError("boto3 not installed. Install with: pip install boto3")
        
        bucket = self.cloud_config.get('s3_bucket')
        key = self.cloud_config.get('s3_key')
        
        if not bucket or not key:
            raise ValueError("Please set s3_bucket and s3_key in config.yaml")
        
        # Create raw data directory
        self.raw_path.mkdir(parents=True, exist_ok=True)
        
        try:
            s3 = boto3.client('s3')
            output_path = self.raw_path / 'dataset.zip'
            
            print(f"Downloading from S3: s3://{bucket}/{key}")
            s3.download_file(bucket, key, str(output_path))
            
            # Extract if it's a zip file
            if output_path.suffix == '.zip':
                print("Extracting downloaded archive...")
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_path)
                output_path.unlink()
                print("Archive extracted successfully!")
                
        except NoCredentialsError:
            raise ValueError("AWS credentials not found. Configure AWS CLI or set environment variables.")
    
    def _download_from_gcs(self):
        """Download from Google Cloud Storage"""
        if not HAS_GCS:
            raise ImportError("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
        
        bucket_name = self.cloud_config.get('gcs_bucket')
        blob_name = self.cloud_config.get('gcs_blob')
        
        if not bucket_name or not blob_name:
            raise ValueError("Please set gcs_bucket and gcs_blob in config.yaml")
        
        # Create raw data directory
        self.raw_path.mkdir(parents=True, exist_ok=True)
        
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        output_path = self.raw_path / 'dataset.zip'
        
        print(f"Downloading from GCS: gs://{bucket_name}/{blob_name}")
        blob.download_to_filename(str(output_path))
        
        # Extract if it's a zip file
        if output_path.suffix == '.zip':
            print("Extracting downloaded archive...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_path)
            output_path.unlink()
            print("Archive extracted successfully!")
        
    def process_dataset(self):
        """Main processing pipeline with cloud support"""
        print("Starting data preprocessing...")
        
        # Download data from cloud if configured
        if self.use_cloud_storage:
            success = self.download_data_from_cloud()
            if not success:
                print("Failed to download data from cloud. Exiting...")
                return
        
        # Check if preprocessing already done (for resuming)
        if self._is_preprocessing_complete():
            print("Preprocessing already completed, skipping...")
            return
        
        # Check if raw data exists
        if not self._check_raw_data_exists():
            print("No raw data found. Please check your data configuration.")
            return
        
        # 1. Parse annotations
        print("Parsing annotations...")
        annotations = self._parse_annotations()
        if not annotations:
            print("No annotations found! Please check your annotation files.")
            return
        
        # 2. Process audio files and create segments
        print("Creating segments...")
        segments_data = self._create_segments(annotations)
        if not segments_data:
            print("No segments created! Please check your audio files.")
            return
        
        # 3. Create splits
        print("Creating train/val/test splits...")
        splits = self._create_splits(segments_data)
        
        # 4. Save processed data
        print("Saving processed data...")
        self._save_processed_data(segments_data, splits)
        
        print("Data preprocessing completed successfully!")
        
    def _is_preprocessing_complete(self):
        """Check if preprocessing is already done"""
        metadata_file = self.processed_path / "metadata.json"
        splits_file = self.splits_path / "splits.json"
        
        if metadata_file.exists() and splits_file.exists():
            # Also check if we have actual processed files
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if len(metadata) > 0:
                    # Check if first segment files exist
                    first_segment = metadata[0]
                    audio_path = Path(first_segment['audio_path'])
                    label_path = Path(first_segment['label_path'])
                    
                    return audio_path.exists() and label_path.exists()
            except:
                pass
        
        return False
    
    def _check_raw_data_exists(self):
        """Check if raw data directory has files"""
        if not self.raw_path.exists():
            return False
        
        # Look for audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(self.raw_path.glob(f"*{ext}")))
            audio_files.extend(list(self.raw_path.glob(f"**/*{ext}")))
        
        return len(audio_files) > 0
        
    def _parse_annotations(self) -> Dict:
        """Parse annotation files into unified format"""
        annotations = defaultdict(list)
        
        # Look for CSV or TextGrid files
        annotation_files = (list(self.raw_path.glob("*.csv")) + 
                          list(self.raw_path.glob("**/*.csv")) +
                          list(self.raw_path.glob("*.TextGrid")) +
                          list(self.raw_path.glob("**/*.TextGrid")))
        
        print(f"Found {len(annotation_files)} annotation files")
        
        for ann_file in annotation_files:
            print(f"Processing annotation file: {ann_file}")
            try:
                if ann_file.suffix == '.csv':
                    df = pd.read_csv(ann_file)
                    # Expected columns: audio_file, start_time, end_time, disfluency_type, annotator_id
                    for _, row in df.iterrows():
                        audio_file = row['audio_file']
                        annotations[audio_file].append({
                            'start': float(row['start_time']),
                            'end': float(row['end_time']),
                            'type': row['disfluency_type'],
                            'annotator': row.get('annotator_id', 'default')
                        })
            except Exception as e:
                print(f"Error processing {ann_file}: {e}")
        
        # Apply inter-annotator agreement
        filtered_annotations = self._apply_annotator_agreement(annotations)
        
        print(f"Loaded annotations for {len(filtered_annotations)} audio files")
        return filtered_annotations
    
    def _apply_annotator_agreement(self, annotations: Dict) -> Dict:
        """Filter annotations based on inter-annotator agreement"""
        filtered = defaultdict(list)
        
        for audio_file, anns in annotations.items():
            # Group by time overlap and type
            grouped = defaultdict(list)
            for ann in anns:
                key = (round(ann['start'], 1), round(ann['end'], 1), ann['type'])
                grouped[key].append(ann['annotator'])
            
            # Keep annotations with at least 2 annotators agreeing (or just 1 if that's all we have)
            for (start, end, dtype), annotators in grouped.items():
                unique_annotators = set(annotators)
                if len(unique_annotators) >= min(2, len(annotators)):
                    filtered[audio_file].append({
                        'start': start,
                        'end': end,
                        'type': dtype
                    })
        
        return filtered
    
    def _load_and_normalize_audio(self, audio_path: Path) -> Optional[np.ndarray]:
        """Load audio file and normalize to target loudness"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            if len(audio) == 0:
                print(f"Empty audio file: {audio_path}")
                return None
            
            # Normalize loudness
            loudness = self.meter.integrated_loudness(audio)
            if np.isfinite(loudness):
                audio = pyln.normalize.loudness(audio, loudness, self.config['data']['target_loudness'])
            
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def _create_segments(self, annotations: Dict) -> List[Dict]:
        """Create overlapping segments from audio files"""
        segments_data = []
        
        for audio_file, anns in annotations.items():
            # Try to find the audio file in various locations
            audio_path = self._find_audio_file(audio_file)
            if not audio_path:
                print(f"Audio file not found: {audio_file}")
                continue
                
            audio = self._load_and_normalize_audio(audio_path)
            if audio is None:
                continue
            
            # Calculate segment parameters
            segment_samples = int(self.segment_length * self.sample_rate)
            hop_samples = int(self.hop_length * self.sample_rate)
            
            # Create segments
            for i, start_sample in enumerate(range(0, len(audio) - segment_samples + 1, hop_samples)):
                end_sample = start_sample + segment_samples
                segment = audio[start_sample:end_sample]
                
                # Calculate time boundaries
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                
                # Generate frame-level labels
                labels = self._generate_frame_labels(anns, start_time, end_time)
                
                # Skip segments with no speech or marked as difficult
                if self._should_skip_segment(labels):
                    continue
                
                segments_data.append({
                    'audio_file': audio_file,
                    'segment_idx': i,
                    'audio': segment,
                    'labels': labels,
                    'start_time': start_time,
                    'end_time': end_time
                })
        
        print(f"Created {len(segments_data)} segments")
        return segments_data
    
    def _find_audio_file(self, audio_file: str) -> Optional[Path]:
        """Find audio file in raw data directory"""
        # Try exact path
        exact_path = self.raw_path / audio_file
        if exact_path.exists():
            return exact_path
        
        # Try searching recursively
        audio_files = list(self.raw_path.rglob(audio_file))
        if audio_files:
            return audio_files[0]
        
        # Try different extensions
        base_name = Path(audio_file).stem
        extensions = ['.wav', '.mp3', '.flac', '.m4a']
        for ext in extensions:
            potential_files = list(self.raw_path.rglob(f"{base_name}{ext}"))
            if potential_files:
                return potential_files[0]
        
        return None
    
    def _generate_frame_labels(self, annotations: List[Dict], start_time: float, end_time: float) -> np.ndarray:
        """Generate frame-level multi-label matrix for a segment"""
        labels = np.zeros((self.frames_per_segment, len(self.disfluency_types)), dtype=np.float32)
        
        for ann in annotations:
            # Check if annotation overlaps with segment
            if ann['end'] <= start_time or ann['start'] >= end_time:
                continue
            
            # Map to frame indices
            rel_start = max(0, ann['start'] - start_time)
            rel_end = min(self.segment_length, ann['end'] - start_time)
            
            frame_start = int(rel_start * self.frames_per_segment / self.segment_length)
            frame_end = int(rel_end * self.frames_per_segment / self.segment_length)
            
            # Set labels
            if ann['type'] in self.disfluency_types:
                class_idx = self.disfluency_types.index(ann['type'])
                labels[frame_start:frame_end, class_idx] = 1
        
        return labels
    
    def _should_skip_segment(self, labels: np.ndarray) -> bool:
        """Check if segment should be skipped (e.g., no speech, music)"""
        # For now, don't skip any segments - let the model learn from all data
        return False
    
    def _create_splits(self, segments_data: List[Dict]) -> Dict[str, List[int]]:
        """Create train/val/test splits ensuring speaker independence"""
        # Group segments by audio file (proxy for speaker)
        file_groups = defaultdict(list)
        for i, seg in enumerate(segments_data):
            file_groups[seg['audio_file']].append(i)
        
        # Split at file level
        files = list(file_groups.keys())
        np.random.shuffle(files)
        
        n_files = len(files)
        train_end = int(n_files * self.config['splits']['train_ratio'])
        val_end = train_end + int(n_files * self.config['splits']['val_ratio'])
        
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        
        # Get segment indices
        splits = {
            'train': [idx for f in train_files for idx in file_groups[f]],
            'val': [idx for f in val_files for idx in file_groups[f]],
            'test': [idx for f in test_files for idx in file_groups[f]]
        }
        
        return splits
    
    def _save_processed_data(self, segments_data: List[Dict], splits: Dict[str, List[int]]):
        """Save processed segments and splits"""
        # Save segments
        for i, segment in enumerate(segments_data):
            # Save audio
            audio_path = self.processed_path / f"segment_{i:06d}.wav"
            sf.write(audio_path, segment['audio'], self.sample_rate)
            
            # Save labels
            label_path = self.processed_path / f"segment_{i:06d}_labels.npy"
            np.save(label_path, segment['labels'])
            
            # Update segment data
            segment['audio_path'] = str(audio_path)
            segment['label_path'] = str(label_path)
            del segment['audio']  # Remove audio array to save memory
        
        # Save metadata
        metadata_path = self.processed_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(segments_data, f, indent=2)
        
        # Save splits
        splits_path = self.splits_path / "splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"Saved {len(segments_data)} segments")
        print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


if __name__ == "__main__":
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    preprocessor = DataPreprocessor(config)
    preprocessor.process_dataset()