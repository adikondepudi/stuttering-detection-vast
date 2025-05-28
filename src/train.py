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
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# WandB import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not available. Install with: pip install wandb")

from .model import build_model
from .dataset import create_dataloaders
from .feature_extraction import FeatureExtractor
from .utils import (EarlyStopping, CostOptimizer, save_checkpoint, load_checkpoint, 
                   monitor_resources, MetricTracker,
                   cleanup_checkpoints, format_time)

class Trainer:
    def __init__(self, config, device='cuda', use_wandb=False):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb and HAS_WANDB
        
        # Initialize Weights & Biases for experiment tracking
        if self.use_wandb:
            wandb.init(
                project=config.get('monitoring', {}).get('wandb_project', 'stuttering-detection'),
                config=config,
                name=f"run_{time.strftime('%Y%m%d_%H%M%S')}",
                save_code=True
            )
            print("WandB initialized for experiment tracking")
        
        # Initialize feature extractor
        print("Initializing feature extractor...")
        self.feature_extractor = FeatureExtractor(config, self.device)
        
        # Create data loaders
        print("Creating data loaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config, self.feature_extractor, num_workers=min(4, torch.get_num_threads())
        )
        
        print(f"Dataset sizes - Train: {len(self.train_loader.dataset)}, "
              f"Val: {len(self.val_loader.dataset)}, Test: {len(self.test_loader.dataset)}")
        
        # Build model
        print("Building model...")
        self.model, self.criterion = build_model(config)
        self.model.to(self.device)
        
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
        
        if self.resume_from_checkpoint and Path(self.resume_from_checkpoint).exists():
            self._load_checkpoint_for_resume()
        
        # Monitoring settings
        self.print_resource_usage = config.get('monitoring', {}).get('print_resource_usage', True)
        self.save_checkpoint_every = config['training'].get('save_checkpoint_every', 5)
        
        # Training timing
        self.epoch_start_time = None
        
    def _load_checkpoint_for_resume(self):
        """Load checkpoint for resuming training"""
        print(f"Resuming from checkpoint: {self.resume_from_checkpoint}")
        try:
            self.start_epoch, _ = load_checkpoint(
                self.resume_from_checkpoint, 
                self.model, 
                self.optimizer
            )
            print(f"Successfully resumed from epoch {self.start_epoch + 1}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
            self.start_epoch = 0
    
    def train(self):
        """Main training loop with cloud optimizations"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        best_val_f1 = 0
        training_start_time = time.time()
        
        # Print initial resource usage
        if self.print_resource_usage:
            monitor_resources()
        
        for epoch in range(self.start_epoch, self.config['training']['max_epochs']):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{self.config['training']['max_epochs']}")
            print("-" * 50)
            
            # Training phase
            train_loss = self._train_epoch(epoch)
            
            # Validation phase
            val_metrics = self._validate(epoch)
            val_f1 = val_metrics['macro_f1']
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            total_elapsed = time.time() - training_start_time
            
            print(f"Epoch time: {format_time(epoch_time)}, "
                  f"Total time: {format_time(total_elapsed)}")
            
            # Update metric tracker
            metrics_update = {
                'train_loss': train_loss,
                'val_macro_f1': val_f1,
                **val_metrics
            }
            self.metric_tracker.update(metrics_update)
            
            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_f1)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != old_lr:
                print(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
            
            # Log metrics
            self._log_metrics(epoch, train_loss, val_metrics, epoch_time)
            
            # Save checkpoint periodically
            if (epoch + 1) % self.save_checkpoint_every == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
                save_checkpoint(self.model, self.optimizer, epoch, val_f1, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path.name}")
                
                # Cleanup old checkpoints to save space
                cleanup_checkpoints(self.checkpoint_dir, keep_last_n=3)
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_path = self.checkpoint_dir / 'best_model.pth'
                save_checkpoint(self.model, self.optimizer, epoch, best_val_f1, best_model_path)
                print(f"New best model saved! F1: {best_val_f1:.4f}")
            
            # Resource monitoring
            if self.print_resource_usage and (epoch + 1) % 5 == 0:
                monitor_resources()
            
            # Cost optimization check
            if self.cost_optimizer and self.cost_optimizer.should_stop_training(val_f1):
                print(f"Cost optimizer triggered early stopping at epoch {epoch+1}")
                break
            
            # Early stopping check
            if self.early_stopping(val_f1):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_training_time = time.time() - training_start_time
        print(f"\nTraining completed in {format_time(total_training_time)}")
        
        # Load best model and evaluate on test set
        print("\nLoading best model for final evaluation...")
        try:
            best_model_path = self.checkpoint_dir / 'best_model.pth'
            if best_model_path.exists():
                load_checkpoint(best_model_path, self.model, self.optimizer)
                print("Best model loaded successfully")
            else:
                print("No best model found, using current model state")
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Using current model state for evaluation")
        
        # Final evaluation
        test_metrics = self._test()
        self._save_results(test_metrics, total_training_time)
        
        # Final resource monitoring
        if self.print_resource_usage:
            monitor_resources()
        
        # Print final summary
        self.metric_tracker.print_summary()
        
        # Close WandB
        if self.use_wandb:
            wandb.finish()
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            
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
            total_loss += loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}",
                'batch': f"{batch_idx+1}/{num_batches}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"Training Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _validate(self, epoch: int) -> Dict:
        """Validate model"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_val_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            
            for features, labels in pbar:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                total_val_loss += loss.item()
                
                # Get predictions
                probs = torch.sigmoid(logits)
                predictions = (probs > self.threshold).float()
                
                # Collect predictions
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        # Calculate metrics
        avg_val_loss = total_val_loss / len(self.val_loader)
        metrics = self._calculate_metrics(all_predictions, all_labels)
        metrics['val_loss'] = avg_val_loss
        
        return metrics
    
    def _test(self) -> Dict:
        """Test model on test set"""
        print("\nEvaluating on test set...")
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(self.test_loader, desc="Testing"):
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                logits = self.model(features)
                probs = torch.sigmoid(logits)
                
                # Threshold predictions
                predictions = (probs > self.threshold).float()
                
                # Collect predictions
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_labels, detailed=True)
        
        return metrics
    
    def _calculate_metrics(self, predictions: List, labels: List, detailed: bool = False) -> Dict:
        """Calculate evaluation metrics"""
        # Concatenate all predictions and labels
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        # Flatten for frame-level evaluation
        predictions_flat = predictions.reshape(-1, len(self.class_names))
        labels_flat = labels.reshape(-1, len(self.class_names))
        
        metrics = {}
        
        # Per-class metrics
        f1_scores = []
        precisions = []
        recalls = []
        
        for i, class_name in enumerate(self.class_names):
            pred_class = predictions_flat[:, i]
            label_class = labels_flat[:, i]
            
            # Handle cases where there are no positive samples
            if np.sum(label_class) == 0:
                if np.sum(pred_class) == 0:
                    f1, precision, recall = 1.0, 1.0, 1.0  # Perfect score for true negatives
                else:
                    f1, precision, recall = 0.0, 0.0, 1.0  # False positives only
            else:
                f1 = f1_score(label_class, pred_class, zero_division=0)
                precision = precision_score(label_class, pred_class, zero_division=0)
                recall = recall_score(label_class, pred_class, zero_division=0)
            
            metrics[f'{class_name}_f1'] = f1
            metrics[f'{class_name}_precision'] = precision
            metrics[f'{class_name}_recall'] = recall
            
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
        
        # Macro averages
        metrics['macro_f1'] = np.mean(f1_scores)
        metrics['macro_precision'] = np.mean(precisions)
        metrics['macro_recall'] = np.mean(recalls)
        
        # Weighted average F1
        supports = [labels_flat[:, i].sum() for i in range(len(self.class_names))]
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
    
    def _log_metrics(self, epoch: int, train_loss: float, val_metrics: Dict, epoch_time: float):
        """Log metrics to tensorboard, wandb, and console"""
        # Console logging
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics.get('val_loss', 0):.4f}")
        print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Val Weighted F1: {val_metrics['weighted_f1']:.4f}")
        print(f"Val UAR: {val_metrics['uar']:.4f}")
        
        # Print per-class F1 scores
        print("Per-class F1 scores:")
        for class_name in self.class_names:
            f1_score = val_metrics[f'{class_name}_f1']
            print(f"  {class_name}: {f1_score:.4f}")
        
        # Tensorboard logging
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_metrics.get('val_loss', 0), epoch)
        self.writer.add_scalar('F1/val_macro', val_metrics['macro_f1'], epoch)
        self.writer.add_scalar('F1/val_weighted', val_metrics['weighted_f1'], epoch)
        self.writer.add_scalar('UAR/val', val_metrics['uar'], epoch)
        self.writer.add_scalar('Time/epoch_time', epoch_time, epoch)
        
        # Per-class metrics
        for class_name in self.class_names:
            self.writer.add_scalar(f'F1/{class_name}', val_metrics[f'{class_name}_f1'], epoch)
            self.writer.add_scalar(f'Precision/{class_name}', val_metrics[f'{class_name}_precision'], epoch)
            self.writer.add_scalar(f'Recall/{class_name}', val_metrics[f'{class_name}_recall'], epoch)
        
        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # WandB logging
        if self.use_wandb:
            wandb_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics.get('val_loss', 0),
                'val_macro_f1': val_metrics['macro_f1'],
                'val_weighted_f1': val_metrics['weighted_f1'],
                'val_uar': val_metrics['uar'],
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }
            
            # Add per-class metrics
            for class_name in self.class_names:
                wandb_metrics[f'val_{class_name}_f1'] = val_metrics[f'{class_name}_f1']
                wandb_metrics[f'val_{class_name}_precision'] = val_metrics[f'{class_name}_precision']
                wandb_metrics[f'val_{class_name}_recall'] = val_metrics[f'{class_name}_recall']
            
            # Add cost optimization info
            if self.cost_optimizer:
                wandb_metrics['elapsed_hours'] = self.cost_optimizer.get_elapsed_time()
            
            wandb.log(wandb_metrics)
    
    def _save_results(self, test_metrics: Dict, total_training_time: float):
        """Save final test results"""
        results_path = self.checkpoint_dir / 'test_results.json'
        
        # Prepare results
        results = {
            'config': self.config,
            'test_metrics': test_metrics,
            'class_names': self.class_names,
            'training_info': {
                'total_training_time_seconds': total_training_time,
                'total_training_time_formatted': format_time(total_training_time),
                'device': str(self.device),
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'dataset_info': {
                'train_size': len(self.train_loader.dataset),
                'val_size': len(self.val_loader.dataset),
                'test_size': len(self.test_loader.dataset)
            }
        }
        
        # Add metric history
        results['metric_history'] = self.metric_tracker.history
        
        # Save
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics history separately
        metrics_path = self.checkpoint_dir / 'training_metrics.json'
        self.metric_tracker.save(metrics_path)
        
        # Print final summary
        self._print_final_summary(test_metrics, total_training_time)
        
        # Upload to WandB if enabled
        if self.use_wandb:
            wandb.log({
                'test_macro_f1': test_metrics['macro_f1'],
                'test_weighted_f1': test_metrics['weighted_f1'],
                'test_uar': test_metrics['uar'],
                'total_training_time': total_training_time
            })
            
            # Upload result files as artifacts
            artifact = wandb.Artifact('training_results', type='results')
            artifact.add_file(str(results_path))
            artifact.add_file(str(metrics_path))
            if (self.checkpoint_dir / 'best_model.pth').exists():
                artifact.add_file(str(self.checkpoint_dir / 'best_model.pth'))
            wandb.log_artifact(artifact)
    
    def _print_final_summary(self, test_metrics: Dict, training_time: float):
        """Print comprehensive final summary"""
        print("\n" + "="*70)
        print("FINAL TEST RESULTS")
        print("="*70)
        
        # Overall metrics
        print(f"Macro F1 Score: {test_metrics['macro_f1']:.4f}")
        print(f"Weighted F1 Score: {test_metrics['weighted_f1']:.4f}")
        print(f"Unweighted Average Recall (UAR): {test_metrics['uar']:.4f}")
        print(f"Macro Precision: {test_metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {test_metrics['macro_recall']:.4f}")
        
        print(f"\nPer-class Results:")
        print("-" * 70)
        print(f"{'Class':<20} {'F1':<8} {'Precision':<12} {'Recall':<8}")
        print("-" * 70)
        
        for class_name in self.class_names:
            f1 = test_metrics[f'{class_name}_f1']
            precision = test_metrics[f'{class_name}_precision']
            recall = test_metrics[f'{class_name}_recall']
            print(f"{class_name:<20} {f1:<8.4f} {precision:<12.4f} {recall:<8.4f}")
        
        print("-" * 70)
        
        # Training info
        print(f"\nTraining Information:")
        print(f"Total Training Time: {format_time(training_time)}")
        print(f"Device Used: {self.device}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Dataset info
        print(f"\nDataset Information:")
        print(f"Training Samples: {len(self.train_loader.dataset):,}")
        print(f"Validation Samples: {len(self.val_loader.dataset):,}")
        print(f"Test Samples: {len(self.test_loader.dataset):,}")
        
        # Best scores achieved during training
        best_val_f1 = self.metric_tracker.get_best('val_macro_f1', mode='max')
        if best_val_f1:
            print(f"\nBest Validation F1 (during training): {best_val_f1:.4f}")
        
        print("="*70)
        
        # Print confusion matrices if available
        if 'confusion_matrices' in test_metrics:
            print(f"\nConfusion Matrices:")
            print("-" * 40)
            
            for class_name in self.class_names:
                cm = np.array(test_metrics['confusion_matrices'][class_name])
                print(f"\n{class_name}:")
                print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
                print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        print("\nTraining completed successfully!")
        print("Results saved to:", self.checkpoint_dir / 'test_results.json')