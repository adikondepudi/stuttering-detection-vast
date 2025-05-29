import os
import json
import csv
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import pyloudnorm as pyln
from pathlib import Path, PurePath
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
import zipfile
import gdown
import tempfile
import shutil
import multiprocessing as mp
from functools import partial
import time
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


def process_audio_file_chunk(args):
    """
    Worker function to process a chunk of audio files in parallel.
    This function will be called by each worker process.
    """
    audio_files_chunk, config, raw_path = args
    
    # Initialize loudness meter for this worker
    sample_rate = config['data']['sample_rate']
    segment_length = config['segmentation']['segment_length']
    hop_length = config['segmentation']['hop_length']
    disfluency_types = config['labels']['disfluency_types']
    frames_per_segment = config['labels']['frames_per_segment']
    
    meter = pyln.Meter(sample_rate)
    
    # CSV to disfluency mapping
    csv_to_disfluency_map = {
        'Prolongation': 'Prolongation',
        'Interjection': 'Interjection',
        'WordRep': 'Word Repetition',
        'SoundRep': 'Sound Repetition',
        'Block': 'Blocks'
    }
    
    def find_audio_file(audio_file: str) -> Optional[Path]:
        """Find audio file in raw data directory"""
        if PurePath(audio_file).name != audio_file:
            print(f"Invalid audio file name (possible path traversal): {audio_file}")
            return None
        
        exact_path = raw_path / audio_file
        if exact_path.exists():
            return exact_path
        
        # Try searching recursively
        audio_files = list(raw_path.rglob(audio_file))
        if audio_files:
            return audio_files[0]
        
        # Try different extensions
        base_name = Path(audio_file).stem
        extensions = ['.wav', '.mp3', '.flac', '.m4a']
        for ext in extensions:
            potential_files = list(raw_path.rglob(f"{base_name}{ext}"))
            if potential_files:
                return potential_files[0]
        
        return None
    
    def load_and_normalize_audio(audio_path: Path) -> Optional[np.ndarray]:
        """Load audio file and normalize to target loudness"""
        try:
            if not str(audio_path).lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                print(f"Unsupported audio file format: {audio_path}")
                return None
            
            audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            if len(audio) == 0 or not np.isfinite(audio).all():
                print(f"Empty or corrupted audio file: {audio_path}")
                return None
            
            loudness = meter.integrated_loudness(audio)
            if np.isfinite(loudness):
                audio = pyln.normalize.loudness(audio, loudness, config['data']['target_loudness'])
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def generate_clip_labels(annotations: List[Dict], clip_duration: float) -> np.ndarray:
        """Generate frame-level labels for an entire clip"""
        labels = np.zeros((frames_per_segment, len(disfluency_types)), dtype=np.float32)
        
        for ann in annotations:
            rel_start = ann['start']
            rel_end = min(ann['end'], clip_duration)
            
            frame_start = int(rel_start * frames_per_segment / segment_length)
            frame_end = int(rel_end * frames_per_segment / segment_length)
            
            frame_start = max(0, frame_start)
            frame_end = min(frames_per_segment, frame_end)
            
            if ann['type'] in disfluency_types:
                class_idx = disfluency_types.index(ann['type'])
                labels[frame_start:frame_end, class_idx] = 1
        
        return labels
    
    def generate_segment_labels(annotations: List[Dict], start_time: float, 
                               end_time: float, clip_duration: float) -> np.ndarray:
        """Generate frame-level labels for a segment within a clip"""
        labels = np.zeros((frames_per_segment, len(disfluency_types)), dtype=np.float32)
        
        for ann in annotations:
            ann_start = ann['start']
            ann_end = min(ann['end'], clip_duration)
            
            if ann_end <= start_time or ann_start >= end_time:
                continue
            
            overlap_start = max(ann_start, start_time)
            overlap_end = min(ann_end, end_time)
            
            rel_start = overlap_start - start_time
            rel_end = overlap_end - start_time
            
            frame_start = int(rel_start * frames_per_segment / segment_length)
            frame_end = int(rel_end * frames_per_segment / segment_length)
            
            frame_start = max(0, frame_start)
            frame_end = min(frames_per_segment, frame_end)
            
            if ann['type'] in disfluency_types:
                class_idx = disfluency_types.index(ann['type'])
                labels[frame_start:frame_end, class_idx] = 1
        
        return labels
    
    # Process this chunk of audio files
    chunk_segments_data = []
    worker_id = mp.current_process().pid
    
    print(f"Worker {worker_id}: Processing {len(audio_files_chunk)} audio files...")
    
    for audio_file, anns in audio_files_chunk.items():
        # Find the audio file
        audio_path = find_audio_file(audio_file)
        if not audio_path:
            print(f"Worker {worker_id}: Audio file not found: {audio_file}")
            continue
            
        audio = load_and_normalize_audio(audio_path)
        if audio is None:
            continue
        
        audio_duration = len(audio) / sample_rate
        
        if audio_duration <= segment_length:
            # Use entire clip as one segment
            segment_samples = int(segment_length * sample_rate)
            if len(audio) < segment_samples:
                audio = np.pad(audio, (0, segment_samples - len(audio)), mode='constant')
            else:
                audio = audio[:segment_samples]
            
            labels = generate_clip_labels(anns, audio_duration)
            
            chunk_segments_data.append({
                'audio_file': audio_file,
                'segment_idx': 0,
                'audio': audio,
                'labels': labels,
                'start_time': 0.0,
                'end_time': segment_length
            })
            
        else:
            # Split longer clips into overlapping segments
            segment_samples = int(segment_length * sample_rate)
            hop_samples = int(hop_length * sample_rate)
            
            for i, start_sample in enumerate(range(0, len(audio) - segment_samples + 1, hop_samples)):
                end_sample = start_sample + segment_samples
                segment = audio[start_sample:end_sample]
                
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                
                labels = generate_segment_labels(anns, start_time, end_time, audio_duration)
                
                chunk_segments_data.append({
                    'audio_file': audio_file,
                    'segment_idx': i,
                    'audio': segment,
                    'labels': labels,
                    'start_time': start_time,
                    'end_time': end_time
                })
    
    print(f"Worker {worker_id}: Created {len(chunk_segments_data)} segments")
    return chunk_segments_data


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
        
        # CSV column mapping
        self.csv_to_disfluency_map = {
            'Prolongation': 'Prolongation',
            'Interjection': 'Interjection',
            'WordRep': 'Word Repetition',
            'SoundRep': 'Sound Repetition',
            'Block': 'Blocks'
        }
        
        # Multiprocessing configuration
        self.num_workers = min(mp.cpu_count(), config.get('processing', {}).get('max_workers', mp.cpu_count()))
        print(f"Using {self.num_workers} worker processes for parallel processing")
        
        # Ensure we don't use more workers than we have files
        if hasattr(self, 'annotations') and self.annotations:
            self.num_workers = min(self.num_workers, len(self.annotations))
        
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
            print("Warning: Google Drive file ID not set. Skipping cloud download. If you have local data, this is fine.")
            return
        
        self.raw_path.mkdir(parents=True, exist_ok=True)
        output_path = self.raw_path / 'dataset.zip'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        print(f"Downloading from Google Drive: {file_id}")
        gdown.download(url, str(output_path), quiet=False)
        
        if output_path.suffix == '.zip':
            print("Extracting downloaded archive...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_path)
            output_path.unlink()
            print("Archive extracted successfully!")
    
    def _download_from_s3(self):
        """Download from AWS S3"""
        if not HAS_BOTO3:
            raise ImportError("boto3 not installed. Install with: pip install boto3")
        
        bucket = self.cloud_config.get('s3_bucket')
        key = self.cloud_config.get('s3_key')
        
        if not bucket or not key:
            raise ValueError("Please set s3_bucket and s3_key in config.yaml")
        
        self.raw_path.mkdir(parents=True, exist_ok=True)
        
        try:
            s3 = boto3.client('s3')
            output_path = self.raw_path / 'dataset.zip'
            
            print(f"Downloading from S3: s3://{bucket}/{key}")
            s3.download_file(bucket, key, str(output_path))
            
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
        
        self.raw_path.mkdir(parents=True, exist_ok=True)
        
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        output_path = self.raw_path / 'dataset.zip'
        
        print(f"Downloading from GCS: gs://{bucket_name}/{blob_name}")
        blob.download_to_filename(str(output_path))
        
        if output_path.suffix == '.zip':
            print("Extracting downloaded archive...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_path)
            output_path.unlink()
            print("Archive extracted successfully!")
        
    def process_dataset(self):
        """Main processing pipeline with cloud support and parallel processing"""
        print("Starting data preprocessing...")
        
        # Download data from cloud if configured
        if self.use_cloud_storage:
            success = self.download_data_from_cloud()
            if not success:
                print("Failed to download data from cloud. Exiting...")
                return
        
        # Check if preprocessing already done
        if self._is_preprocessing_complete():
            print("Preprocessing already completed, skipping...")
            return
        
        # Check if raw data exists
        if not self._check_raw_data_exists():
            print("No raw data found. Please check your data configuration.")
            return
        
        # Parse clip labels CSV
        print("Parsing clip labels CSV...")
        annotations = self._parse_clip_labels_csv()
        if not annotations:
            print("No annotations found! Please check your clip_labels.csv file.")
            return
        
        # Create segments using parallel processing
        print("Creating segments with parallel processing...")
        start_time = time.time()
        segments_data = self._create_segments_from_clips_parallel(annotations)
        processing_time = time.time() - start_time
        
        if not segments_data:
            print("No segments created! Please check your audio files.")
            return
        
        print(f"Parallel processing completed in {processing_time:.2f} seconds")
        print(f"Created {len(segments_data)} segments total")
        
        # Create splits
        print("Creating train/val/test splits...")
        splits = self._create_splits(segments_data)
        
        # Save processed data
        print("Saving processed data...")
        self._save_processed_data(segments_data, splits)
        
        print("Data preprocessing completed successfully!")
        
    def _is_preprocessing_complete(self):
        """Check if preprocessing is already done"""
        metadata_file = self.processed_path / "metadata.json"
        splits_file = self.splits_path / "splits.json"
        
        if metadata_file.exists() and splits_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if len(metadata) > 0:
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
        
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(self.raw_path.glob(f"*{ext}")))
            audio_files.extend(list(self.raw_path.glob(f"**/*{ext}")))
        
        csv_files = list(self.raw_path.glob("clip_labels.csv")) + list(self.raw_path.glob("**/clip_labels.csv"))
        
        return len(audio_files) > 0 and len(csv_files) > 0
        
    def _parse_clip_labels_csv(self) -> Dict:
        """Parse the clip_labels.csv file"""
        csv_files = (list(self.raw_path.glob("clip_labels.csv")) + 
                    list(self.raw_path.glob("**/clip_labels.csv")))
        
        if not csv_files:
            print("clip_labels.csv not found!")
            return {}
        
        csv_file = csv_files[0]
        print(f"Processing clip labels: {csv_file}")
        
        annotations = defaultdict(list)
        
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} rows from clip_labels.csv")
            print(f"CSV columns: {list(df.columns)}")
            
            required_cols = ['Show', 'EpId', 'ClipId', 'Start', 'Stop']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return {}
            
            for row in df.itertuples(index=False):
                show = getattr(row, 'Show', None)
                ep_id_val = getattr(row, 'EpId', None)
                ep_id_numeric = pd.to_numeric(ep_id_val, errors='coerce')
                if pd.isna(ep_id_numeric):
                    print(f"Invalid EpId: {ep_id_val}, skipping row")
                    continue
                ep_id = f"{int(ep_id_numeric):03d}"
                clip_id_val = getattr(row, 'ClipId', 0)
                clip_id_numeric = pd.to_numeric(clip_id_val, errors='coerce')
                if pd.isna(clip_id_numeric):
                    print(f"Invalid ClipId: {clip_id_val}, skipping row")
                    continue
                clip_id = int(clip_id_numeric)
                audio_filename = f"{show}_{ep_id}_{clip_id}.wav"
                
                start_time = float(getattr(row, 'Start', 0)) / 1000.0
                stop_time = float(getattr(row, 'Stop', 0)) / 1000.0
                clip_duration = stop_time - start_time
                
                if clip_duration <= 0.5:
                    continue
                
                clip_annotations = []
                
                for csv_col, disfluency_type in self.csv_to_disfluency_map.items():
                    val = getattr(row, csv_col, None)
                    if pd.notna(val):
                        numeric_val = pd.to_numeric(val, errors='coerce')
                        if pd.notna(numeric_val) and numeric_val > 0:
                            clip_annotations.append({
                                'start': 0.0,
                                'end': clip_duration,
                                'type': disfluency_type
                            })
                
                no_stutter_val = pd.to_numeric(getattr(row, 'NoStutteredWords', 0), errors='coerce')
                if clip_annotations or (pd.notna(no_stutter_val) and no_stutter_val > 0):
                    annotations[audio_filename] = clip_annotations
                    
                    if len(annotations) <= 5:
                        print(f"  {audio_filename}: {len(clip_annotations)} disfluencies")
            
            print(f"Processed annotations for {len(annotations)} audio files")
            
        except Exception as e:
            print(f"Error processing clip_labels.csv: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
        return annotations
    
    def _create_segments_from_clips_parallel(self, annotations: Dict) -> List[Dict]:
        """Create segments from clip-based annotations using parallel processing"""
        if not annotations:
            return []
        
        # Split annotations into chunks for parallel processing
        items = list(annotations.items())
        chunk_size = max(1, len(items) // self.num_workers)
        chunks = []
        
        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i + chunk_size])
            chunks.append((chunk, self.config, self.raw_path))
        
        print(f"Split {len(items)} audio files into {len(chunks)} chunks for parallel processing")
        
        # Process chunks in parallel
        all_segments_data = []
        
        if self.num_workers == 1:
            # Single-threaded processing for debugging
            for chunk_args in chunks:
                chunk_results = process_audio_file_chunk(chunk_args)
                all_segments_data.extend(chunk_results)
        else:
            # Multi-threaded processing
            try:
                with mp.Pool(processes=self.num_workers) as pool:
                    chunk_results = pool.map(process_audio_file_chunk, chunks)
                    
                    # Flatten results
                    for chunk_result in chunk_results:
                        all_segments_data.extend(chunk_result)
                        
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                print("Falling back to single-threaded processing...")
                
                # Fallback to single-threaded
                for chunk_args in chunks:
                    chunk_results = process_audio_file_chunk(chunk_args)
                    all_segments_data.extend(chunk_results)
        
        print(f"Created {len(all_segments_data)} segments from {len(annotations)} clips")
        return all_segments_data
    
    def _create_splits(self, segments_data: List[Dict]) -> Dict[str, List[int]]:
        """Create train/val/test splits ensuring speaker independence"""
        episode_groups = defaultdict(list)
        for i, seg in enumerate(segments_data):
            audio_file = seg['audio_file']
            parts = audio_file.split('_')
            if len(parts) >= 2:
                episode_key = f"{parts[0]}_{parts[1]}"
            else:
                episode_key = audio_file
            episode_groups[episode_key].append(i)
        
        episodes = list(episode_groups.keys())
        np.random.shuffle(episodes)
        
        n_episodes = len(episodes)
        train_end = int(n_episodes * self.config['splits']['train_ratio'])
        val_end = train_end + int(n_episodes * self.config['splits']['val_ratio'])
        
        train_episodes = episodes[:train_end]
        val_episodes = episodes[train_end:val_end]
        test_episodes = episodes[val_end:]
        
        splits = {
            'train': [idx for ep in train_episodes for idx in episode_groups[ep]],
            'val': [idx for ep in val_episodes for idx in episode_groups[ep]],
            'test': [idx for ep in test_episodes for idx in episode_groups[ep]]
        }
        
        print(f"Split into {len(train_episodes)} train episodes, {len(val_episodes)} val episodes, {len(test_episodes)} test episodes")
        
        return splits
    
    def _save_processed_data(self, segments_data: List[Dict], splits: Dict[str, List[int]]):
        """Save processed segments and splits"""
        print("Saving segments to disk...")
        
        # Create metadata without numpy arrays
        metadata = []
        
        for i, segment in enumerate(segments_data):
            # Save audio
            audio_path = self.processed_path / f"segment_{i:06d}.wav"
            sf.write(audio_path, segment['audio'], self.sample_rate)
            
            # Save labels
            label_path = self.processed_path / f"segment_{i:06d}_labels.npy"
            np.save(label_path, segment['labels'])
            
            # Create metadata entry (without numpy arrays)
            metadata_entry = {
                'audio_file': segment['audio_file'],
                'segment_idx': segment['segment_idx'],
                'audio_path': str(audio_path),
                'label_path': str(label_path),
                'start_time': float(segment['start_time']),  # Ensure it's a Python float
                'end_time': float(segment['end_time'])       # Ensure it's a Python float
            }
            metadata.append(metadata_entry)
        
        # Save metadata (now serializable)
        metadata_path = self.processed_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
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