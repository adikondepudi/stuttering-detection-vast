#!/usr/bin/env python3
"""
Main entry point for stuttering detection training pipeline (Optimized for GPU)
"""

import argparse
import yaml
import sys
import time
from pathlib import Path
import torch
import gc

# Add the feature preprocessing module
sys.path.append(str(Path(__file__).parent))
try:
    from feature_preprocessing import run_feature_extraction, verify_gpu_usage, FeaturePreprocessor, PreExtractedDataset, create_fast_dataloaders
    from fast_trainer import FastTrainer
    FAST_MODE_AVAILABLE = True
except ImportError:
    print("Warning: Fast mode modules not found. Using standard training.")
    FAST_MODE_AVAILABLE = False
    from src import Trainer as FastTrainer  # Fallback to regular trainer

from src import DataPreprocessor, set_seed, get_device, count_parameters, print_training_header


def validate_config(config):
    required_keys = [
        ('data.raw_data_path', str),
        ('data.processed_data_path', str),
        ('labels.disfluency_types', list),
        ('training.batch_size', int),
        ('training.max_epochs', int)
    ]
    for key_path, expected_type in required_keys:
        keys = key_path.split('.')
        current = config
        try:
            for k in keys:
                current = current[k]
            if not isinstance(current, expected_type):
                raise ValueError(f"Config key {key_path} should be {expected_type}, got {type(current)}")
        except KeyError:
            raise ValueError(f"Missing required config key: {key_path}")


def check_gpu_status():
    """Check and report GPU status"""
    print("\n" + "="*50)
    print("GPU STATUS CHECK")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available!")
        print("Training will run on CPU (very slow)")
        return False
    
    print(f"CUDA available: Yes")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch built with CUDA: {torch.version.cuda}")
    
    # GPU info
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Multi-processor count: {props.multi_processor_count}")
    
    # Test GPU
    print("\nTesting GPU...")
    if verify_gpu_usage():
        print("GPU test passed!")
        return True
    else:
        print("GPU test failed!")
        return False


def check_preprocessed_data_exists(config):
    """Check if preprocessed data exists and is complete"""
    processed_path = Path(config['data']['processed_data_path'])
    splits_path = Path(config['data']['splits_path'])
    
    metadata_file = processed_path / "metadata.json"
    splits_file = splits_path / "splits.json"
    
    return metadata_file.exists() and splits_file.exists()


def check_features_exist(config):
    """Check if pre-extracted features exist"""
    features_path = Path(config['data']['processed_data_path']) / 'features'
    feature_info_file = features_path / 'feature_info.json'
    
    if not feature_info_file.exists():
        return False
    
    # Check if we have actual feature files
    feature_files = list(features_path.glob('*.npy'))
    return len(feature_files) > 0


def main(args):
    # Print header with system information
    device = print_training_header()
    
    # GPU status check
    gpu_available = check_gpu_status()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"\nConfiguration loaded from: {args.config}")
        validate_config(config)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        sys.exit(1)
    
    # Set random seed for reproducibility
    set_seed(42)
    print("Random seed set to 42 for reproducibility")
    
    # Override settings for fast mode
    if args.fast:
        print("\nFAST MODE ENABLED - Using pre-extracted features")
        config['features']['use_cached'] = True
    
    # Override WandB setting if specified
    if args.use_wandb:
        if 'monitoring' not in config:
            config['monitoring'] = {}
        config['monitoring']['use_wandb'] = True
        print("WandB logging enabled via command line argument")
    
    start_time = time.time()
    
    try:
        if args.mode == 'preprocess':
            # Run data preprocessing only
            print("\n" + "="*50)
            print("STARTING DATA PREPROCESSING")
            print("="*50)
            
            preprocessor = DataPreprocessor(config)
            preprocessor.process_dataset()
            print("Preprocessing completed successfully!")
            
        elif args.mode == 'extract-features':
            # Run feature extraction only
            if not check_preprocessed_data_exists(config):
                print("No preprocessed data found. Please run preprocessing first.")
                sys.exit(1)
            
            print("\n" + "="*50)
            print("STARTING FEATURE EXTRACTION")
            print("="*50)
            
            # Force GPU if available
            device = torch.device('cuda' if gpu_available else 'cpu')
            run_feature_extraction(config, device=device)
            
        elif args.mode == 'train':
            # Check if preprocessed data exists
            if not check_preprocessed_data_exists(config):
                print("No preprocessed data found. Please run preprocessing first.")
                print("Run: python main.py --mode preprocess")
                sys.exit(1)
            
            # Run training with fast mode
            print("\n" + "="*50)
            print("STARTING MODEL TRAINING (FAST MODE)")
            print("="*50)
            
            # Use GPU if available
            device = 'cuda' if gpu_available else 'cpu'
            
            # Initialize fast trainer
            trainer = FastTrainer(config, device=device, use_wandb=args.use_wandb)
            
            # Print estimated training time
            samples_per_epoch = len(trainer.train_loader.dataset)
            batch_size = config['training']['batch_size']
            max_epochs = config['training']['max_epochs']
            
            # With pre-extracted features, expect ~0.1s per batch on GPU
            time_per_batch = 0.1 if device == 'cuda' else 1.0
            batches_per_epoch = len(trainer.train_loader)
            estimated_time = max_epochs * batches_per_epoch * time_per_batch
            
            print(f"\nTraining estimates:")
            print(f"  Batches per epoch: {batches_per_epoch}")
            print(f"  Expected time per batch: {time_per_batch:.2f}s")
            print(f"  Estimated total time: {estimated_time/3600:.1f} hours")
            
            # Train model
            trainer.train()
            print("Training completed successfully!")
            
        elif args.mode == 'all':
            # Run full pipeline with optimizations
            print("\n" + "="*50)
            print("RUNNING FULL PIPELINE (OPTIMIZED)")
            print("="*50)
            
            # Step 1: Preprocessing
            if check_preprocessed_data_exists(config):
                print("\nPreprocessed data already exists. Skipping preprocessing...")
            else:
                print("\n--- DATA PREPROCESSING ---")
                preprocessor = DataPreprocessor(config)
                preprocessor.process_dataset()
            
            # Step 2: Feature extraction (critical for speed)
            if check_features_exist(config):
                print("\nPre-extracted features already exist. Skipping feature extraction...")
            else:
                print("\n--- FEATURE EXTRACTION ---")
                device = torch.device('cuda' if gpu_available else 'cpu')
                run_feature_extraction(config, device=device)
                
                # Clear memory after feature extraction
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Step 3: Training
            print("\n--- MODEL TRAINING ---")
            device = 'cuda' if gpu_available else 'cpu'
            trainer = FastTrainer(config, device=device, use_wandb=args.use_wandb)
            trainer.train()
            
            print("\nFull pipeline completed successfully!")
        
        else:
            print(f"Unknown mode: {args.mode}")
            print("Available modes: preprocess, extract-features, train, all")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Checkpoints have been saved and can be resumed later.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Print total execution time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"
        
        print(f"\nTotal execution time: {time_str}")
        print("="*60)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Stuttering Detection Training Pipeline (GPU Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode all --fast                     # Run full pipeline with optimizations
  python main.py --mode extract-features               # Extract features only (after preprocessing)
  python main.py --mode train --fast --use-wandb      # Fast training with WandB
  python main.py --config config/custom.yaml --mode all --fast  # Custom config
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['preprocess', 'extract-features', 'train', 'all'],
        default='all',
        help='Mode to run: preprocess only, extract features, train only, or all (default: all)'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        default=True,
        help='Use fast training with pre-extracted features (default: True)'
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases logging for experiment tracking'
    )
    
    parser.add_argument(
        '--gpu-check',
        action='store_true',
        help='Run GPU check only and exit'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Quick GPU check mode
    if args.gpu_check:
        check_gpu_status()
        sys.exit(0)
    
    main(args)