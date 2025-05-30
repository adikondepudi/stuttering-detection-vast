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
warnings.filterwarnings('ignore')

# WandB import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from .model import build_model
from .dataset import create_dataloaders
from .feature_extraction import FeatureExtractor
from .utils import (EarlyStopping, CostOptimizer, save_checkpoint, load_checkpoint, 
                   monitor_resources, MetricTracker,
                   cleanup_checkpoints, format_time)


class Trainer:
    def __init__(self, config, device='cuda', use_wandb=False, verbose=False):
        self.config = config
        self.verbose = verbose
        
        # Set device with validation
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            if device == 'cuda':
                print("WARNING: CUDA requested but not available, using CPU")
        
        self.use_wandb = use_wandb and HAS_WANDB
        
        # Initialize Weights & Biases
        if self.use_wandb:
            try:
                wandb.init(
                    project=config.get('monitoring', {}).get('wandb_project', 'stuttering-detection'),
                    config=config,
                    name=f"run_{time.strftime('%Y%m%d_%H%M%S')}",
                    save_code=True
                )
                print("WandB initialized for experiment tracking")
            except Exception as e:
                print(f"WandB initialization failed: {e}")
                self.use_wandb = False
        
        # Initialize feature extractor
        print("Initializing feature extractor...")
        try:
            self.feature_extractor = FeatureExtractor(config, self.device, verbose=self.verbose)
            # Get actual dimensions from feature extractor
            self.expected_feature_dim = self.feature_extractor.get_feature_dim()
            self.expected_seq_len = config['labels']['pooled_frames']
        except Exception as e:
            print(f"Error initializing feature extractor: {e}")
            raise
        
        # Set expected dimensions BEFORE creating data loaders
        self.num_classes = config['labels']['num_classes']
        
        # Create data loaders with error handling
        print("Creating data loaders...")
        try:
            self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
                config, self.feature_extractor, 
                num_workers=min(4, torch.get_num_threads()),
                verbose=self.verbose
            )
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            raise
        
        # Validate data loaders
        self._validate_data_loaders()
        
        # Build model
        print("Building model...")
        try:
            # Pass actual feature dimension to model builder
            self.model, self.criterion = build_model(
                config, 
                verbose=self.verbose,
            )
            self.model.to(self.device)
        except Exception as e:
            print(f"Error building model: {e}")
            raise
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['initial_lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize F1 score
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
        self.writer = SummaryWriter('runs/stutter_detection')
        
        # Class names and metrics
        self.class_names = config['labels']['disfluency_types']
        self.threshold = config['training']['prediction_threshold']
        
        # Metric tracking
        metric_names = ['train_loss', 'val_loss', 'val_macro_f1', 'val_weighted_f1', 'val_uar']
        metric_names.extend([f'val_{name}_f1' for name in self.class_names])
        self.metric_tracker = MetricTracker(metric_names)
        
        # Checkpoint resuming
        self.resume_from_checkpoint = config.get('training', {}).get('resume_checkpoint')
        self.start_epoch = 0
        
        if self.resume_from_checkpoint is not None and Path(self.resume_from_checkpoint).exists():
            self._load_checkpoint_for_resume()
        
        # Monitoring settings
        self.print_resource_usage = config.get('monitoring', {}).get('print_resource_usage', True)
        self.save_checkpoint_every = config['training'].get('save_checkpoint_every', 5)
        
        # Training state
        self.best_val_f1 = 0
        self.training_start_time = None
    
    def _validate_data_loaders(self):
        """Validate that data loaders are properly initialized"""
        loaders = {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader
        }
        
        for name, loader in loaders.items():
            if loader is None:
                raise ValueError(f"{name} loader is None")
            
            if len(loader) == 0:
                raise ValueError(f"{name} loader has no batches")
            
            # Test loading one batch
            try:
                features, labels = next(iter(loader))
                expected_shape = (loader.batch_size, self.expected_seq_len, self.expected_feature_dim)
                actual_shape = features.shape
                
                if self.verbose:
                    print(f"{name} loader - Features: {actual_shape}, Labels: {labels.shape}")
                
                # Check dimensions (allow for last batch to be smaller)
                if actual_shape[1] != expected_shape[1] or actual_shape[2] != expected_shape[2]:
                    print(f"Warning: {name} loader dimension mismatch")
                    print(f"  Expected: (batch, {expected_shape[1]}, {expected_shape[2]})")
                    print(f"  Actual: {actual_shape}")
                    
            except Exception as e:
                raise ValueError(f"Error loading batch from {name} loader: {e}")
        
        print(f"Data loaders validated - Train: {len(self.train_loader.dataset)}, "
              f"Val: {len(self.val_loader.dataset)}, Test: {len(self.test_loader.dataset)}")
    
    def _load_checkpoint_for_resume(self):
        """Load checkpoint for resuming training"""
        print(f"Resuming from checkpoint: {self.resume_from_checkpoint}")
        try:
            checkpoint_data = torch.load(self.resume_from_checkpoint, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint_data:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Load training state
            self.start_epoch = checkpoint_data.get('epoch', 0) + 1
            self.best_val_f1 = checkpoint_data.get('score', 0)
            
            # Load metrics history if available
            if 'metrics_history' in checkpoint_data:
                self.metric_tracker.history = checkpoint_data['metrics_history']
            
            print(f"Successfully resumed from epoch {self.start_epoch}")
            print(f"Best validation F1 so far: {self.best_val_f1:.4f}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
            self.start_epoch = 0
    
    def train(self):
        """Main training loop with comprehensive error handling"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        self.training_start_time = time.time()
        
        # Print initial resource usage
        if self.print_resource_usage:
            monitor_resources()
        
        try:
            for epoch in range(self.start_epoch, self.config['training']['max_epochs']):
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                epoch_start_time = time.time()
                print(f"\nEpoch {epoch+1}/{self.config['training']['max_epochs']}")
                print("-" * 50)
                
                # Training phase
                try:
                    train_loss = self._train_epoch(epoch)
                except Exception as e:
                    print(f"Error during training epoch: {e}")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()
                    continue
                
                # Validation phase
                try:
                    val_metrics = self._validate(epoch)
                    val_f1 = val_metrics['macro_f1']
                except Exception as e:
                    print(f"Error during validation: {e}")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()
                    val_metrics = {'macro_f1': 0}
                    val_f1 = 0
                
                # Calculate timing
                epoch_time = time.time() - epoch_start_time
                total_elapsed = time.time() - self.training_start_time
                
                print(f"Epoch time: {format_time(epoch_time)}, "
                      f"Total time: {format_time(total_elapsed)}")
                
                # Update metrics
                self._update_and_log_metrics(epoch, train_loss, val_metrics, epoch_time)
                
                # Learning rate scheduling
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_f1)
                new_lr = self.optimizer.param_groups[0]['lr']
                
                if new_lr != old_lr:
                    print(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
                
                # Save checkpoints
                self._save_checkpoints(epoch, val_f1)
                
                # Check stopping criteria
                if self._check_stopping_criteria(epoch, val_f1):
                    break
                
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        except Exception as e:
            print(f"\nUnexpected error during training: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        
        # Final evaluation and cleanup
        self._finalize_training()
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch with error handling"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        num_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch_data in enumerate(pbar):
            try:
                # Validate batch data
                if batch_data is None or len(batch_data) != 2:
                    print(f"Warning: Invalid batch data at index {batch_idx}")
                    continue
                
                features, labels = batch_data
                
                # Move to device
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Validate shapes
                if features.shape[1:] != (self.expected_seq_len, self.expected_feature_dim):
                    if self.verbose:
                        print(f"Warning: Feature shape mismatch in batch {batch_idx}: {features.shape}")
                
                # Forward pass
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss in batch {batch_idx}")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip_value']
                )
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item() * features.size(0)
                num_batches += 1
                num_samples += features.size(0)
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}",
                    'batch': f"{num_batches}/{len(self.train_loader)}"
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Calculate average loss
        avg_loss = total_loss / max(num_samples, 1)
        print(f"Training Loss: {avg_loss:.4f} ({num_batches} batches processed)")
        
        return avg_loss
    
    def _validate(self, epoch: int) -> Dict:
        """Validate model with error handling"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_val_loss = 0
        num_batches = 0
        num_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            
            for batch_idx, batch_data in enumerate(pbar):
                try:
                    if batch_data is None or len(batch_data) != 2:
                        continue
                    
                    features, labels = batch_data
                    features = features.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    logits = self.model(features)
                    loss = self.criterion(logits, labels)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_val_loss += loss.item() * features.size(0)
                        num_samples += features.size(0)
                    
                    # Get predictions
                    probs = torch.sigmoid(logits)
                    predictions = (probs > self.threshold).float()
                    
                    # Collect predictions
                    all_predictions.append(predictions.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    num_batches += 1
                    
                    pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Calculate metrics
        avg_val_loss = total_val_loss / max(num_samples, 1)
        
        if len(all_predictions) > 0:
            metrics = self._calculate_metrics(all_predictions, all_labels)
            metrics['val_loss'] = avg_val_loss
        else:
            print("Warning: No valid validation batches")
            metrics = {
                'val_loss': avg_val_loss,
                'macro_f1': 0,
                'weighted_f1': 0,
                'uar': 0
            }
        
        return metrics
    
    def _calculate_metrics(self, predictions: List, labels: List, detailed: bool = False) -> Dict:
        """Calculate evaluation metrics with error handling"""
        try:
            # Concatenate all predictions and labels
            if not predictions or not labels:
                return {'macro_f1': 0.0, 'weighted_f1': 0.0, 'uar': 0.0}
            
            predictions = np.concatenate(predictions, axis=0)
            labels = np.concatenate(labels, axis=0)
            
            # Flatten for frame-level evaluation
            predictions_flat = predictions.reshape(-1, self.num_classes)
            labels_flat = labels.reshape(-1, self.num_classes)
            
            metrics = {}
            
            # Per-class metrics
            f1_scores = []
            precisions = []
            recalls = []
            
            for i, class_name in enumerate(self.class_names):
                pred_class = predictions_flat[:, i]
                label_class = labels_flat[:, i]
                
                # Calculate metrics with zero_division handling
                f1 = f1_score(label_class, pred_class, zero_division=0)
                precision = precision_score(label_class, pred_class, zero_division=0)
                recall = recall_score(label_class, pred_class, zero_division=0)
                
                # Store per-class metrics
                safe_class_name = class_name.replace(' ', '_')
                metrics[f'val_{safe_class_name}_f1'] = f1
                metrics[f'val_{safe_class_name}_precision'] = precision
                metrics[f'val_{safe_class_name}_recall'] = recall
                
                f1_scores.append(f1)
                precisions.append(precision)
                recalls.append(recall)
            
            # Macro averages
            metrics['macro_f1'] = np.mean(f1_scores)
            metrics['macro_precision'] = np.mean(precisions)
            metrics['macro_recall'] = np.mean(recalls)
            
            # Weighted average F1
            supports = [labels_flat[:, i].sum() for i in range(self.num_classes)]
            total_support = sum(supports)
            
            if total_support > 0:
                weighted_f1 = sum(f1 * sup / total_support for f1, sup in zip(f1_scores, supports))
                metrics['weighted_f1'] = weighted_f1
            else:
                metrics['weighted_f1'] = 0.0
            
            # UAR (Unweighted Average Recall)
            metrics['uar'] = np.mean(recalls)
            
            # Confusion matrices if detailed
            if detailed:
                metrics['confusion_matrices'] = {}
                for i, class_name in enumerate(self.class_names):
                    cm = confusion_matrix(labels_flat[:, i], predictions_flat[:, i])
                    metrics['confusion_matrices'][class_name] = cm.tolist()
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'uar': 0.0,
                'macro_precision': 0.0,
                'macro_recall': 0.0
            }
    
    def _update_and_log_metrics(self, epoch: int, train_loss: float, 
                               val_metrics: Dict, epoch_time: float):
        """Update and log metrics"""
        # Update metric tracker
        metrics_update = {
            'train_loss': train_loss,
            'val_loss': val_metrics.get('val_loss', 0),
            'val_macro_f1': val_metrics['macro_f1'],
            'val_weighted_f1': val_metrics['weighted_f1'],
            'val_uar': val_metrics['uar']
        }
        
        # Add per-class metrics
        for class_name in self.class_names:
            safe_class_name = class_name.replace(' ', '_')
            key = f'val_{safe_class_name}_f1'
            if key in val_metrics:
                metrics_update[f'val_{class_name}_f1'] = val_metrics[key]
        
        self.metric_tracker.update(metrics_update)
        
        # Console logging
        self._print_epoch_summary(train_loss, val_metrics)
        
        # Tensorboard logging
        self._log_to_tensorboard(epoch, train_loss, val_metrics, epoch_time)
        
        # WandB logging
        if self.use_wandb:
            self._log_to_wandb(epoch, train_loss, val_metrics, epoch_time)
    
    def _print_epoch_summary(self, train_loss: float, val_metrics: Dict):
        """Print epoch summary to console"""
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics.get('val_loss', 0):.4f}")
        print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Val Weighted F1: {val_metrics['weighted_f1']:.4f}")
        print(f"Val UAR: {val_metrics['uar']:.4f}")
        
        print("Per-class F1 scores:")
        for class_name in self.class_names:
            safe_class_name = class_name.replace(' ', '_')
            f1_key = f'val_{safe_class_name}_f1'
            if f1_key in val_metrics:
                print(f"  {class_name}: {val_metrics[f1_key]:.4f}")
    
    def _log_to_tensorboard(self, epoch: int, train_loss: float, 
                           val_metrics: Dict, epoch_time: float):
        """Log metrics to tensorboard"""
        try:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics.get('val_loss', 0), epoch)
            self.writer.add_scalar('F1/val_macro', val_metrics['macro_f1'], epoch)
            self.writer.add_scalar('F1/val_weighted', val_metrics['weighted_f1'], epoch)
            self.writer.add_scalar('UAR/val', val_metrics['uar'], epoch)
            self.writer.add_scalar('Time/epoch_time', epoch_time, epoch)
            
            # Per-class metrics
            for class_name in self.class_names:
                safe_class_name = class_name.replace(' ', '_')
                f1_key = f'val_{safe_class_name}_f1'
                if f1_key in val_metrics:
                    self.writer.add_scalar(f'F1/{class_name}', val_metrics[f1_key], epoch)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
        except Exception as e:
            print(f"Error logging to tensorboard: {e}")
    
    def _log_to_wandb(self, epoch: int, train_loss: float, 
                     val_metrics: Dict, epoch_time: float):
        """Log metrics to WandB"""
        try:
            wandb_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics.get('val_loss', 0),
                'val_macro_f1': val_metrics['macro_f1'],
                'val_weighted_f1': val_metrics['weighted_f1'],
                'val_uar': val_metrics['uar'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            
            # Add per-class metrics
            for class_name in self.class_names:
                safe_class_name = class_name.replace(' ', '_')
                f1_key = f'val_{safe_class_name}_f1'
                if f1_key in val_metrics:
                    wandb_metrics[f'val_{class_name}_f1'] = val_metrics[f1_key]
            
            # Add cost optimization info
            if self.cost_optimizer:
                wandb_metrics['elapsed_hours'] = self.cost_optimizer.get_elapsed_time()
            
            wandb.log(wandb_metrics)
            
        except Exception as e:
            print(f"Error logging to WandB: {e}")
    
    def _save_checkpoints(self, epoch: int, val_f1: float):
        """Save checkpoints with error handling"""
        try:
            # Save periodic checkpoint
            if (epoch + 1) % self.save_checkpoint_every == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
                
                # Include metrics history in checkpoint
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'score': val_f1,
                    'metrics_history': self.metric_tracker.history,
                    'config': self.config,
                    'timestamp': time.time()
                }
                
                torch.save(checkpoint_data, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path.name}")
                
                # Cleanup old checkpoints
                cleanup_checkpoints(self.checkpoint_dir, keep_last_n=3)
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                best_model_path = self.checkpoint_dir / 'best_model.pth'
                
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'score': self.best_val_f1,
                    'config': self.config,
                    'timestamp': time.time()
                }
                
                torch.save(checkpoint_data, best_model_path)
                print(f"New best model saved! F1: {self.best_val_f1:.4f}")
                
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def _check_stopping_criteria(self, epoch: int, val_f1: float) -> bool:
        """Check if training should stop"""
        # Resource monitoring
        if self.print_resource_usage and (epoch + 1) % 5 == 0:
            monitor_resources()
        
        # Cost optimization check
        if self.cost_optimizer and self.cost_optimizer.should_stop_training(val_f1):
            print(f"Cost optimizer triggered early stopping at epoch {epoch+1}")
            return True
        
        # Early stopping check
        if self.early_stopping(val_f1):
            print(f"Early stopping triggered at epoch {epoch+1}")
            return True
        
        return False
    
    def _finalize_training(self):
        """Finalize training and run evaluation"""
        total_training_time = time.time() - self.training_start_time
        print(f"\nTraining completed in {format_time(total_training_time)}")
        
        # Load best model
        print("\nLoading best model for final evaluation...")
        best_model_path = self.checkpoint_dir / 'best_model.pth'
        
        if best_model_path.exists():
            try:
                checkpoint_data = torch.load(best_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
                print(f"Best model loaded (F1: {checkpoint_data.get('score', 0):.4f})")
            except Exception as e:
                print(f"Error loading best model: {e}")
                print("Using current model state for evaluation")
        else:
            print("No best model found, using current model state")
        
        # Final evaluation on test set
        try:
            test_metrics = self._test()
            self._save_results(test_metrics, total_training_time)
        except Exception as e:
            print(f"Error during final evaluation: {e}")
            test_metrics = {'error': str(e)}
            self._save_results(test_metrics, total_training_time)
        
        # Final resource monitoring
        if self.print_resource_usage:
            monitor_resources()
        
        # Print final summary
        self.metric_tracker.print_summary()
        
        # Close resources
        try:
            self.writer.close()
            if self.use_wandb:
                wandb.finish()
        except:
            pass
    
    def _test(self) -> Dict:
        """Test model on test set with comprehensive evaluation"""
        print("\nEvaluating on test set...")
        self.model.eval()
        all_predictions = []
        all_labels = []
        test_loss = 0
        num_batches = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.test_loader, desc="Testing")):
                try:
                    if batch_data is None or len(batch_data) != 2:
                        continue
                    
                    features, labels = batch_data
                    features = features.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    logits = self.model(features)
                    loss = self.criterion(logits, labels)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        test_loss += loss.item() * features.size(0)
                        num_samples += features.size(0)
                    
                    # Get predictions
                    probs = torch.sigmoid(logits)
                    predictions = (probs > self.threshold).float()
                    
                    # Collect predictions
                    all_predictions.append(predictions.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error in test batch {batch_idx}: {e}")
                    continue
        
        # Calculate metrics
        avg_test_loss = test_loss / max(num_samples, 1)
        
        if len(all_predictions) > 0:
            metrics = self._calculate_metrics(all_predictions, all_labels, detailed=True)
            metrics['test_loss'] = avg_test_loss
            metrics['num_test_samples'] = num_samples
            metrics['num_test_batches'] = num_batches
        else:
            print("Warning: No valid test batches")
            metrics = {
                'test_loss': avg_test_loss,
                'macro_f1': 0,
                'weighted_f1': 0,
                'uar': 0,
                'error': 'No valid test batches'
            }
        
        return metrics
    
    def _save_results(self, test_metrics: Dict, total_training_time: float):
        """Save comprehensive results with error handling"""
        results_path = self.checkpoint_dir / 'test_results.json'
        
        # Prepare results
        results = {
            'config': self.config,
            'test_metrics': self._serialize_metrics(test_metrics),
            'class_names': self.class_names,
            'training_info': {
                'total_training_time_seconds': total_training_time,
                'total_training_time_formatted': format_time(total_training_time),
                'device': str(self.device),
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'final_learning_rate': self.optimizer.param_groups[0]['lr'],
                'best_val_f1': float(self.best_val_f1),
                'total_epochs_trained': len(self.metric_tracker.history.get('train_loss', [])),
                'actual_feature_dim': self.expected_feature_dim
            },
            'dataset_info': {
                'train_size': len(self.train_loader.dataset) if self.train_loader else 0,
                'val_size': len(self.val_loader.dataset) if self.val_loader else 0,
                'test_size': len(self.test_loader.dataset) if self.test_loader else 0
            }
        }
        
        # Add metric history
        results['metric_history'] = self._serialize_metrics(self.metric_tracker.history)
        
        # Save results
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {results_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
            # Try saving minimal results
            try:
                minimal_results = {
                    'test_f1': test_metrics.get('macro_f1', 0),
                    'error': str(e)
                }
                with open(results_path.with_suffix('.minimal.json'), 'w') as f:
                    json.dump(minimal_results, f)
            except:
                pass
        
        # Save metrics history separately
        metrics_path = self.checkpoint_dir / 'training_metrics.json'
        try:
            self.metric_tracker.save(metrics_path)
        except Exception as e:
            print(f"Error saving metrics history: {e}")
        
        # Print final summary
        self._print_final_summary(test_metrics, total_training_time)
        
        # Log to WandB
        if self.use_wandb:
            try:
                wandb.log({
                    'test_macro_f1': test_metrics.get('macro_f1', 0),
                    'test_weighted_f1': test_metrics.get('weighted_f1', 0),
                    'test_uar': test_metrics.get('uar', 0),
                    'total_training_time': total_training_time
                })
                
                # Upload result files as artifacts
                artifact = wandb.Artifact('training_results', type='results')
                artifact.add_file(str(results_path))
                if metrics_path.exists():
                    artifact.add_file(str(metrics_path))
                if (self.checkpoint_dir / 'best_model.pth').exists():
                    artifact.add_file(str(self.checkpoint_dir / 'best_model.pth'))
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Error logging to WandB: {e}")
    
    def _serialize_metrics(self, metrics: Dict) -> Dict:
        """Convert numpy types to Python native types for JSON serialization"""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        return convert_numpy(metrics)
    
    def _print_final_summary(self, test_metrics: Dict, training_time: float):
        """Print comprehensive final summary"""
        print("\n" + "="*70)
        print("FINAL TEST RESULTS")
        print("="*70)
        
        if 'error' in test_metrics:
            print(f"Warning: Test evaluation had errors: {test_metrics['error']}")
        
        # Overall metrics
        print(f"Test Loss: {test_metrics.get('test_loss', 'N/A'):.4f}")
        print(f"Macro F1 Score: {test_metrics.get('macro_f1', 0):.4f}")
        print(f"Weighted F1 Score: {test_metrics.get('weighted_f1', 0):.4f}")
        print(f"Unweighted Average Recall (UAR): {test_metrics.get('uar', 0):.4f}")
        print(f"Macro Precision: {test_metrics.get('macro_precision', 0):.4f}")
        print(f"Macro Recall: {test_metrics.get('macro_recall', 0):.4f}")
        
        # Per-class results
        print(f"\nPer-class Results:")
        print("-" * 70)
        print(f"{'Class':<20} {'F1':<8} {'Precision':<12} {'Recall':<8}")
        print("-" * 70)
        
        for class_name in self.class_names:
            safe_class_name = class_name.replace(' ', '_')
            f1 = test_metrics.get(f'val_{safe_class_name}_f1', 0)
            precision = test_metrics.get(f'val_{safe_class_name}_precision', 0)
            recall = test_metrics.get(f'val_{safe_class_name}_recall', 0)
            print(f"{class_name:<20} {f1:<8.4f} {precision:<12.4f} {recall:<8.4f}")
        
        print("-" * 70)
        
        # Training info
        print(f"\nTraining Information:")
        print(f"Total Training Time: {format_time(training_time)}")
        print(f"Device Used: {self.device}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Best Validation F1: {self.best_val_f1:.4f}")
        print(f"Feature Dimension: {self.expected_feature_dim}")
        
        # Dataset info
        print(f"\nDataset Information:")
        print(f"Training Samples: {len(self.train_loader.dataset) if self.train_loader else 'N/A'}")
        print(f"Validation Samples: {len(self.val_loader.dataset) if self.val_loader else 'N/A'}")
        print(f"Test Samples: {test_metrics.get('num_test_samples', 'N/A')}")
        
        # Confusion matrices if available
        if 'confusion_matrices' in test_metrics:
            print(f"\nConfusion Matrices:")
            print("-" * 40)
            
            for class_name in self.class_names:
                if class_name in test_metrics['confusion_matrices']:
                    cm = np.array(test_metrics['confusion_matrices'][class_name])
                    print(f"\n{class_name}:")
                    if cm.shape == (2, 2):
                        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
                        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
                    else:
                        print(f"  Shape: {cm.shape}")
        
        print("\n" + "="*70)
        print("Training completed successfully!")
        print("="*70)


# Utility function for testing
def test_trainer(config_path='config/config.yaml'):
    """Test trainer initialization and basic functionality"""
    import yaml
    
    print("Testing trainer...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override some settings for testing
    config['training']['max_epochs'] = 2
    config['training']['batch_size'] = 4
    
    try:
        # Initialize trainer
        trainer = Trainer(config, device='cpu', use_wandb=False, verbose=True)
        print("Trainer initialized successfully!")
        
        # Test one training step
        print("\nTesting one training batch...")
        trainer.model.train()
        
        batch = next(iter(trainer.train_loader))
        features, labels = batch
        features = features.to(trainer.device)
        labels = labels.to(trainer.device)
        
        # Forward pass
        logits = trainer.model(features)
        loss = trainer.criterion(logits, labels)
        
        print(f"Sample batch shapes - Features: {features.shape}, Labels: {labels.shape}")
        print(f"Model output shape: {logits.shape}")
        print(f"Loss value: {loss.item():.4f}")
        
        # Test validation
        print("\nTesting validation...")
        val_metrics = trainer._validate(0)
        print(f"Validation metrics: {val_metrics}")
        
        print("\nTrainer test complete!")
        
    except Exception as e:
        print(f"Error testing trainer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_trainer()