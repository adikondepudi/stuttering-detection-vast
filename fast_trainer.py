# Updated train.py with pre-extracted features
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, Tuple, List, Optional
import warnings
import gc
warnings.filterwarnings('ignore')

# Import the new components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from feature_preprocessing import (
    run_feature_extraction, 
    create_fast_dataloaders,
    verify_gpu_usage
)

# WandB import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from src.model import build_model
from src.utils import (
    EarlyStopping, CostOptimizer, save_checkpoint, load_checkpoint, 
    monitor_resources, MetricTracker,
    cleanup_checkpoints, format_time
)


class FastTrainer:
    """Optimized trainer using pre-extracted features"""
    
    def __init__(self, config, device='cuda', use_wandb=False, verbose=False):
        self.config = config
        self.verbose = verbose
        
        # Set device with validation
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            # Verify GPU is actually working
            if not verify_gpu_usage():
                print("WARNING: GPU verification failed, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
            if device == 'cuda':
                print("WARNING: CUDA requested but not available, using CPU")
        
        print(f"Training device: {self.device}")
        
        # Monitor initial GPU state
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Initial GPU memory: {allocated:.2f}/{total:.2f} GB")
        
        self.use_wandb = use_wandb and HAS_WANDB
        
        # Initialize Weights & Biases
        if self.use_wandb:
            try:
                wandb.init(
                    project=config.get('monitoring', {}).get('wandb_project', 'stuttering-detection'),
                    config=config,
                    name=f"fast_run_{time.strftime('%Y%m%d_%H%M%S')}",
                    save_code=True
                )
                print("WandB initialized for experiment tracking")
            except Exception as e:
                print(f"WandB initialization failed: {e}")
                self.use_wandb = False
        
        # Run feature extraction if needed
        print("\nChecking for pre-extracted features...")
        features_path = Path(config['data']['processed_data_path']) / 'features'
        if not features_path.exists() or len(list(features_path.glob('*.npy'))) == 0:
            print("Pre-extracted features not found. Running feature extraction...")
            run_feature_extraction(config, device=self.device)
            # Clear GPU memory after feature extraction
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        else:
            print("Pre-extracted features found!")
        
        # Set expected dimensions
        self.num_classes = config['labels']['num_classes']
        self.expected_seq_len = config['labels']['pooled_frames']
        whisper_dim = config['features'].get('whisper_dim', 768)
        n_mfcc = config['features']['mfcc']['n_mfcc']
        self.expected_feature_dim = whisper_dim + n_mfcc * 3
        
        # Create data loaders using pre-extracted features
        print("\nCreating fast data loaders...")
        try:
            self.train_loader, self.val_loader, self.test_loader = create_fast_dataloaders(
                config, 
                num_workers=min(4, torch.get_num_threads())
            )
            print(f"Data loaders created successfully!")
            print(f"  Train batches: {len(self.train_loader)}")
            print(f"  Val batches: {len(self.val_loader)}")
            print(f"  Test batches: {len(self.test_loader)}")
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            raise
        
        # Validate data loaders
        self._validate_data_loaders()
        
        # Build model
        print("\nBuilding model...")
        try:
            self.model, self.criterion = build_model(config, verbose=self.verbose)
            self.model.to(self.device)
            
            # Verify model is on GPU
            if self.device.type == 'cuda':
                # Check a parameter to verify device
                first_param = next(self.model.parameters())
                print(f"Model device check: {first_param.device}")
                
                # Check memory after model loading
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"GPU memory after model load: {allocated:.2f} GB")
                
        except Exception as e:
            print(f"Error building model: {e}")
            raise
        
        # Print model parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['initial_lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config['training']['lr_reduction_factor'],
            patience=config['training']['lr_scheduler_patience'],
            min_lr=config['training']['min_lr'],
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            mode='max'
        )
        
        # Cost optimization
        cost_config = config.get('cost_optimization', {})
        self.cost_optimizer = CostOptimizer(
            max_hours=cost_config.get('max_training_hours', 8.0),
            target_performance=cost_config.get('target_performance', 0.80)
        ) if cost_config.get('enable_early_cost_stop', True) else None
        
        # Setup directories
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter('runs/fast_stutter_detection')
        
        # Class names and metrics
        self.class_names = config['labels']['disfluency_types']
        self.threshold = config['training']['prediction_threshold']
        
        # Metric tracking
        metric_names = ['train_loss', 'val_loss', 'val_macro_f1', 'val_weighted_f1', 'val_uar']
        metric_names.extend([f'val_{name}_f1' for name in self.class_names])
        self.metric_tracker = MetricTracker(metric_names)
        
        # Training state
        self.best_val_f1 = 0
        self.start_epoch = 0
        self.training_start_time = None
        
        # Print final GPU state
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"Final initialization GPU memory: {allocated:.2f} GB")
    
    def _validate_data_loaders(self):
        """Validate that data loaders are properly initialized"""
        print("\nValidating data loaders...")
        
        for name, loader in [('train', self.train_loader), ('val', self.val_loader), ('test', self.test_loader)]:
            if loader is None:
                raise ValueError(f"{name} loader is None")
            
            if len(loader) == 0:
                raise ValueError(f"{name} loader has no batches")
            
            # Test loading one batch
            try:
                features, labels = next(iter(loader))
                print(f"{name} loader - Features: {features.shape}, Labels: {labels.shape}")
                
                # Move to device to test
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Check GPU memory usage
                if self.device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated() / 1e9
                    print(f"  GPU memory after loading batch: {allocated:.2f} GB")
                
                # Clean up
                del features, labels
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                raise ValueError(f"Error loading batch from {name} loader: {e}")
    
    def train(self):
        """Main training loop optimized for speed"""
        print("\n" + "="*60)
        print("STARTING FAST TRAINING")
        print("="*60)
        
        self.training_start_time = time.time()
        
        # Initial resource monitoring
        if self.config.get('monitoring', {}).get('print_resource_usage', True):
            monitor_resources()
        
        # Training loop
        try:
            for epoch in range(self.start_epoch, self.config['training']['max_epochs']):
                epoch_start_time = time.time()
                
                # Clear GPU cache at start of epoch
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                print(f"\nEpoch {epoch+1}/{self.config['training']['max_epochs']}")
                print("-" * 50)
                
                # Training phase
                train_loss = self._train_epoch(epoch)
                
                # Validation phase
                val_metrics = self._validate(epoch)
                val_f1 = val_metrics['macro_f1']
                
                # Timing
                epoch_time = time.time() - epoch_start_time
                total_time = time.time() - self.training_start_time
                
                print(f"Epoch time: {format_time(epoch_time)} ({epoch_time/len(self.train_loader):.2f}s/batch)")
                print(f"Total time: {format_time(total_time)}")
                
                # GPU memory monitoring
                if self.device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated() / 1e9
                    max_allocated = torch.cuda.max_memory_allocated() / 1e9
                    print(f"GPU memory: {allocated:.2f} GB (peak: {max_allocated:.2f} GB)")
                
                # Update metrics and checkpoints
                self._update_metrics_and_save(epoch, train_loss, val_metrics, val_f1, epoch_time)
                
                # Check stopping criteria
                if self._check_stopping_criteria(epoch, val_f1):
                    break
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nUnexpected error during training: {e}")
            import traceback
            traceback.print_exc()
        
        # Final evaluation
        self._finalize_training()
    
    def _train_epoch(self, epoch: int) -> float:
        """Optimized training epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Use autocast for mixed precision training
        use_amp = self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (features, labels) in enumerate(pbar):
            # Move to device
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Mixed precision training
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(features)
                    loss = self.criterion(logits, labels)
                
                # Backward pass with scaler
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip_value']
                )
                
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # Standard training
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip_value']
                )
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * features.size(0)
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'gpu_mb': f"{torch.cuda.memory_allocated()/1e6:.0f}" if self.device.type == 'cuda' else "N/A"
            })
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss
    
    def _validate(self, epoch: int) -> Dict:
        """Fast validation"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_val_loss = 0
        
        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validating"):
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Use autocast for validation too
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits = self.model(features)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(features)
                    loss = self.criterion(logits, labels)
                
                total_val_loss += loss.item() * features.size(0)
                
                # Get predictions
                probs = torch.sigmoid(logits)
                predictions = (probs > self.threshold).float()
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        avg_val_loss = total_val_loss / len(self.val_loader.dataset)
        metrics = self._calculate_metrics(all_predictions, all_labels)
        metrics['val_loss'] = avg_val_loss
        
        return metrics
    
    def _calculate_metrics(self, predictions: List, labels: List) -> Dict:
        """Calculate evaluation metrics"""
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        # Flatten for frame-level evaluation
        predictions_flat = predictions.reshape(-1, self.num_classes)
        labels_flat = labels.reshape(-1, self.num_classes)
        
        metrics = {}
        f1_scores = []
        recalls = []
        
        for i, class_name in enumerate(self.class_names):
            f1 = f1_score(labels_flat[:, i], predictions_flat[:, i], zero_division=0)
            recall = recall_score(labels_flat[:, i], predictions_flat[:, i], zero_division=0)
            
            safe_class_name = class_name.replace(' ', '_')
            metrics[f'val_{safe_class_name}_f1'] = f1
            
            f1_scores.append(f1)
            recalls.append(recall)
        
        metrics['macro_f1'] = np.mean(f1_scores)
        metrics['weighted_f1'] = np.average(f1_scores, weights=[labels_flat[:, i].sum() for i in range(self.num_classes)])
        metrics['uar'] = np.mean(recalls)
        
        return metrics
    
    def _update_metrics_and_save(self, epoch, train_loss, val_metrics, val_f1, epoch_time):
        """Update metrics and save checkpoints"""
        # Update metric tracker
        metrics_update = {
            'train_loss': train_loss,
            'val_loss': val_metrics.get('val_loss', 0),
            'val_macro_f1': val_metrics['macro_f1'],
            'val_weighted_f1': val_metrics['weighted_f1'],
            'val_uar': val_metrics['uar']
        }
        
        for class_name in self.class_names:
            safe_class_name = class_name.replace(' ', '_')
            key = f'val_{safe_class_name}_f1'
            if key in val_metrics:
                metrics_update[f'val_{class_name}_f1'] = val_metrics[key]
        
        self.metric_tracker.update(metrics_update)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics.get('val_loss', 0):.4f}")
        print(f"  Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Val UAR: {val_metrics['uar']:.4f}")
        
        # Learning rate scheduling
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(val_f1)
        new_lr = self.optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  Learning rate: {old_lr:.2e} -> {new_lr:.2e}")
        
        # Save best model
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            save_checkpoint(
                self.model, self.optimizer, epoch, self.best_val_f1,
                self.checkpoint_dir / 'best_model.pth',
                {'config': self.config, 'feature_dim': self.expected_feature_dim}
            )
            print(f"  New best model! F1: {self.best_val_f1:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                self.model, self.optimizer, epoch, val_f1,
                self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            )
        
        # Tensorboard logging
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_metrics.get('val_loss', 0), epoch)
        self.writer.add_scalar('F1/val_macro', val_metrics['macro_f1'], epoch)
        self.writer.add_scalar('Time/epoch', epoch_time, epoch)
        
        # WandB logging
        if self.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_macro_f1': val_metrics['macro_f1'],
                'epoch_time': epoch_time,
                'learning_rate': new_lr
            })
    
    def _check_stopping_criteria(self, epoch: int, val_f1: float) -> bool:
        """Check if training should stop"""
        # Cost optimization
        if self.cost_optimizer and self.cost_optimizer.should_stop_training(val_f1):
            print("Cost optimizer triggered early stopping")
            return True
        
        # Early stopping
        if self.early_stopping(val_f1):
            print("Early stopping triggered")
            return True
        
        return False
    
    def _finalize_training(self):
        """Final evaluation and cleanup"""
        total_time = time.time() - self.training_start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        
        # Load best model
        best_model_path = self.checkpoint_dir / 'best_model.pth'
        if best_model_path.exists():
            checkpoint_data = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            print(f"Loaded best model (F1: {checkpoint_data.get('score', 0):.4f})")
        
        # Test evaluation
        print("\nEvaluating on test set...")
        test_metrics = self._test()
        
        # Save results
        results = {
            'test_metrics': test_metrics,
            'training_time': total_time,
            'best_val_f1': float(self.best_val_f1),
            'device': str(self.device),
            'feature_dim': self.expected_feature_dim
        }
        
        with open(self.checkpoint_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nFinal Test Results:")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  UAR: {test_metrics['uar']:.4f}")
        
        # Cleanup
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
    
    def _test(self) -> Dict:
        """Test evaluation"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(self.test_loader, desc="Testing"):
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                logits = self.model(features)
                probs = torch.sigmoid(logits)
                predictions = (probs > self.threshold).float()
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        return self._calculate_metrics(all_predictions, all_labels)