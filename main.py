#!/usr/bin/env python3
"""
Main entry point for stuttering detection training pipeline (Vast.ai optimized)
"""

import argparse
import yaml
import sys
import time
from pathlib import Path
import atexit

from src import DataPreprocessor, Trainer, set_seed, get_device, count_parameters, print_training_header


def validate_config(config):
    required_keys = [
        'data.raw_data_path',
        'data.processed_data_path',
        'labels.disfluency_types',
        'training.batch_size'
    ]
    for key in required_keys:
        keys = key.split('.')
        current = config
        try:
            for k in keys:
                current = current[k]
        except KeyError:
            raise ValueError(f"Missing required config key: {key}")


def main(args):
    # Print header with system information
    device = print_training_header()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {args.config}")
        validate_config(config)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        sys.exit(1)
    
    # Set random seed for reproducibility
    set_seed(42)
    print("Random seed set to 42 for reproducibility")
    
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
            
        elif args.mode == 'train':
            # Check if preprocessed data exists
            processed_path = Path(config['data']['processed_data_path'])
            if not processed_path.exists() or not any(processed_path.iterdir()):
                print("No preprocessed data found. Running preprocessing first...")
                
                print("\n" + "="*50)
                print("PREPROCESSING DATA FIRST")
                print("="*50)
                
                preprocessor = DataPreprocessor(config)
                preprocessor.process_dataset()
            
            # Run training
            print("\n" + "="*50)
            print("STARTING MODEL TRAINING")
            print("="*50)
            
            trainer = Trainer(config, device, use_wandb=args.use_wandb)
            
            # Print model info
            model_params = count_parameters(trainer.model)
            print(f"Model parameters: {model_params:,}")
            print(f"Training device: {device}")
            
            # Estimate training time if possible
            try:
                from src.utils import estimate_training_time
                if hasattr(trainer, 'train_loader'):
                    samples_per_epoch = len(trainer.train_loader.dataset)
                    batch_size = config['training']['batch_size']
                    max_epochs = config['training']['max_epochs']
                    
                    # Rough estimate: 0.1 seconds per batch (will vary by GPU)
                    estimated_time_per_batch = 0.1
                    estimate_training_time(max_epochs, samples_per_epoch, batch_size, estimated_time_per_batch)
            except Exception as e:
                print(f"Could not estimate training time: {e}")
            
            # Train model
            trainer.train()
            print("Training completed successfully!")
            
        elif args.mode == 'all':
            # Run both preprocessing and training
            print("\n" + "="*50)
            print("RUNNING FULL PIPELINE")
            print("="*50)
            
            # Preprocessing
            print("\n--- DATA PREPROCESSING ---")
            preprocessor = DataPreprocessor(config)
            preprocessor.process_dataset()
            
            # Training
            print("\n--- MODEL TRAINING ---")
            trainer = Trainer(config, device, use_wandb=args.use_wandb)
            
            model_params = count_parameters(trainer.model)
            print(f"Model parameters: {model_params:,}")
            print(f"Training device: {device}")
            
            trainer.train()
            
            print("\nFull pipeline completed successfully!")
        
        else:
            print(f"Unknown mode: {args.mode}")
            print("Available modes: preprocess, train, all")
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
        description="Stuttering Detection Training Pipeline (Vast.ai Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode all                                    # Run full pipeline
  python main.py --mode preprocess                            # Preprocessing only
  python main.py --mode train --use-wandb                     # Training with WandB
  python main.py --config config/custom.yaml --mode all       # Custom config
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
        choices=['preprocess', 'train', 'all'],
        default='all',
        help='Mode to run: preprocess only, train only, or all (default: all)'
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases logging for experiment tracking'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint file to resume training from'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    temp_config_path = None
    args = parse_arguments()
    
    # Handle resume argument
    if args.resume:
        # Load config and set resume checkpoint
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            if 'training' not in config:
                config['training'] = {}
            config['training']['resume_checkpoint'] = args.resume
            # Save modified config temporarily
            temp_config_path = Path(args.config).parent / f"temp_{Path(args.config).name}"
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f)
            args.config = str(temp_config_path)
            print(f"Resume checkpoint set to: {args.resume}")
        except Exception as e:
            print(f"Error setting resume checkpoint: {e}")
            sys.exit(1)
        
        def cleanup_temp_config():
            try:
                if temp_config_path and Path(temp_config_path).exists():
                    Path(temp_config_path).unlink()
                    print(f"Temporary config file {temp_config_path} deleted.")
            except Exception as e:
                print(f"Warning: Could not delete temporary config file: {e}")
        atexit.register(cleanup_temp_config)
    
    main(args)