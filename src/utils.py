import torch
import numpy as np
from pathlib import Path
import json
import random
import os
import time
import psutil
from typing import Dict, Optional, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# GPU monitoring imports
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


class EarlyStopping:
    """Early stopping to prevent overfitting with robust handling"""
    
    def __init__(self, patience=10, mode='max', delta=0.0001, verbose=False):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch=None):
        """Check if should stop training"""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch if epoch is not None else 0
            if self.verbose:
                print(f"EarlyStopping: Initial best score = {score:.4f}")
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            if self.verbose:
                print(f"EarlyStopping: Score improved from {self.best_score:.4f} to {score:.4f}")
            self.best_score = score
            self.best_epoch = epoch if epoch is not None else self.best_epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement. Counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping! Best score was {self.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class CostOptimizer:
    """Monitor training costs and performance for early stopping"""
    
    def __init__(self, max_hours=8.0, target_performance=0.85, check_interval=300):
        self.max_hours = max_hours
        self.target_performance = target_performance
        self.check_interval = check_interval  # Check every N seconds
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.performance_history = []
        
    def should_stop_training(self, current_performance):
        """Decide if training should stop based on cost and performance"""
        current_time = time.time()
        elapsed_hours = (current_time - self.start_time) / 3600
        
        # Record performance
        self.performance_history.append({
            'time_hours': elapsed_hours,
            'performance': current_performance
        })
        
        # Stop if we've reached target performance early
        if current_performance >= self.target_performance:
            print(f"Target performance ({self.target_performance:.3f}) reached! Current: {current_performance:.3f}")
            print("Stopping training early to save costs.")
            return True
        
        # Stop if we're approaching time limit
        if elapsed_hours >= self.max_hours * 0.95:  # 95% of max time
            print(f"Approaching time limit ({self.max_hours:.1f}h). Elapsed: {elapsed_hours:.1f}h")
            print("Stopping training to prevent unexpected costs.")
            return True
        
        # Check if performance is plateauing (optional)
        if len(self.performance_history) >= 5:
            recent_performances = [p['performance'] for p in self.performance_history[-5:]]
            performance_std = np.std(recent_performances)
            if performance_std < 0.001:  # Very little change
                print(f"Performance plateaued (std={performance_std:.4f}). Consider stopping.")
        
        # Periodic status update
        if current_time - self.last_check_time > self.check_interval:
            self.last_check_time = current_time
            print(f"Cost optimizer: {elapsed_hours:.1f}h elapsed, "
                  f"performance={current_performance:.3f}, "
                  f"target={self.target_performance:.3f}")
        
        return False
    
    def get_elapsed_time(self):
        """Get elapsed training time in hours"""
        return (time.time() - self.start_time) / 3600
    
    def get_status(self) -> Dict[str, Any]:
        """Get current cost optimizer status"""
        elapsed_hours = self.get_elapsed_time()
        return {
            'elapsed_hours': elapsed_hours,
            'remaining_hours': max(0, self.max_hours - elapsed_hours),
            'progress_percent': min(100, (elapsed_hours / self.max_hours) * 100),
            'performance_history': self.performance_history
        }


def save_checkpoint(model, optimizer, epoch, score, filepath, additional_info=None):
    """Save model checkpoint with comprehensive information"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'score': score,
        'timestamp': time.time(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    # Add additional info if provided
    if additional_info:
        checkpoint.update(additional_info)
    
    # Save checkpoint
    try:
        torch.save(checkpoint, filepath)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"Checkpoint saved: {filepath} ({file_size:.1f} MB)")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        # Try saving without optimizer state as fallback
        try:
            checkpoint_minimal = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': score
            }
            torch.save(checkpoint_minimal, filepath)
            print("Saved minimal checkpoint (without optimizer state)")
        except Exception as e2:
            print(f"Failed to save even minimal checkpoint: {e2}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """Load model checkpoint with error handling"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint)
            return 0, 0.0  # Default epoch and score
        
        # Load optimizer state if available and requested
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # Get metadata
        epoch = checkpoint.get('epoch', 0)
        score = checkpoint.get('score', 0.0)
        
        # Print info
        timestamp = checkpoint.get('timestamp', 0)
        if timestamp:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
            print(f"Loaded checkpoint from {time_str}")
        
        return epoch, score
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def count_parameters(model) -> int:
    """Count trainable parameters in model"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")
    return trainable


def get_device() -> torch.device:
    """Get available device with detailed information"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1e9
            print(f"  GPU {i}: {props.name}, {total_memory:.1f} GB")
        
        # CUDA version info
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
        # CPU info
        try:
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            print(f"CPU Cores: {cpu_count} physical, {cpu_count_logical} logical")
        except:
            pass
    
    return device


def monitor_resources(verbose=True):
    """Monitor system resources with enhanced error handling"""
    if not verbose:
        return
    
    print("\n" + "="*50)
    print("RESOURCE MONITORING")
    print("="*50)
    
    # GPU monitoring
    if torch.cuda.is_available():
        try:
            # PyTorch CUDA memory info
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                cached = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                
                print(f"GPU {i} Memory:")
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Cached: {cached:.2f} GB")
                print(f"  Total: {total:.2f} GB")
                print(f"  Free: {total - allocated:.2f} GB")
            
            # GPUtil for additional info
            if HAS_GPUTIL:
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        print(f"\nGPU {gpu.id} Status:")
                        print(f"  Utilization: {gpu.load*100:.1f}%")
                        print(f"  Temperature: {gpu.temperature}°C")
                except:
                    pass
                    
        except Exception as e:
            print(f"GPU monitoring error: {e}")
    
    # CPU and RAM monitoring
    try:
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        print(f"\nCPU:")
        print(f"  Usage: {cpu_percent:.1f}%")
        if cpu_freq:
            print(f"  Frequency: {cpu_freq.current:.0f} MHz")
        
        # Memory info
        memory = psutil.virtual_memory()
        print(f"\nRAM:")
        print(f"  Total: {memory.total/1e9:.1f} GB")
        print(f"  Used: {memory.used/1e9:.1f} GB ({memory.percent:.1f}%)")
        print(f"  Available: {memory.available/1e9:.1f} GB")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        print(f"\nDisk:")
        print(f"  Total: {disk.total/1e9:.1f} GB")
        print(f"  Used: {disk.used/1e9:.1f} GB ({disk.percent:.1f}%)")
        print(f"  Free: {disk.free/1e9:.1f} GB")
        
    except Exception as e:
        print(f"System monitoring error: {e}")
    
    print("="*50)


def estimate_training_time(num_epochs: int, samples_per_epoch: int, 
                          batch_size: int, time_per_batch: float) -> float:
    """Estimate total training time with detailed breakdown"""
    batches_per_epoch = (samples_per_epoch + batch_size - 1) // batch_size
    total_batches = num_epochs * batches_per_epoch
    estimated_seconds = total_batches * time_per_batch
    
    # Add overhead estimate (validation, checkpointing, etc.)
    overhead_factor = 1.2  # 20% overhead
    estimated_seconds *= overhead_factor
    
    print(f"\nTraining Time Estimation:")
    print(f"  Samples per epoch: {samples_per_epoch:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {batches_per_epoch:,}")
    print(f"  Total batches: {total_batches:,}")
    print(f"  Time per batch: {time_per_batch:.3f}s")
    print(f"  Estimated time: {format_time(estimated_seconds)}")
    print(f"  (includes {int((overhead_factor-1)*100)}% overhead for validation, etc.)")
    
    return estimated_seconds


def cleanup_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3, 
                       keep_best: bool = True) -> List[Path]:
    """Clean up old checkpoints to save disk space"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []
    
    # Get all checkpoint files
    checkpoint_files = list(checkpoint_path.glob("checkpoint_epoch_*.pth"))
    
    if len(checkpoint_files) <= keep_last_n:
        return []
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Files to remove
    files_to_remove = checkpoint_files[keep_last_n:]
    removed_files = []
    
    for file in files_to_remove:
        # Skip best model if requested
        if keep_best and 'best' in file.name:
            continue
        
        try:
            file_size = file.stat().st_size / (1024 * 1024)  # MB
            file.unlink()
            removed_files.append(file)
            print(f"Removed checkpoint: {file.name} ({file_size:.1f} MB)")
        except Exception as e:
            print(f"Error removing {file.name}: {e}")
    
    # Report total space saved
    if removed_files:
        total_saved = sum(f.stat().st_size for f in removed_files if f.exists()) / (1024 * 1024)
        print(f"Total space saved: {total_saved:.1f} MB")
    
    return removed_files


class MetricTracker:
    """Track metrics during training with enhanced functionality"""
    
    def __init__(self, metrics_names: List[str]):
        self.metrics_names = metrics_names
        self.history = {name: [] for name in metrics_names}
        self.best_values = {}
        self.best_epochs = {}
    
    def update(self, metrics_dict: Dict[str, float], epoch: Optional[int] = None):
        """Update metrics"""
        for name in self.metrics_names:
            if name in metrics_dict:
                value = metrics_dict[name]
                self.history[name].append(value)
                
                # Track best values
                if name not in self.best_values:
                    self.best_values[name] = value
                    self.best_epochs[name] = epoch if epoch is not None else len(self.history[name]) - 1
                else:
                    # Update best if improved (assume higher is better except for 'loss')
                    if 'loss' in name.lower():
                        if value < self.best_values[name]:
                            self.best_values[name] = value
                            self.best_epochs[name] = epoch if epoch is not None else len(self.history[name]) - 1
                    else:
                        if value > self.best_values[name]:
                            self.best_values[name] = value
                            self.best_epochs[name] = epoch if epoch is not None else len(self.history[name]) - 1
    
    def get_best(self, metric_name: str, mode: str = 'max') -> Optional[float]:
        """Get best value for a metric"""
        if metric_name not in self.history:
            return None
        
        values = self.history[metric_name]
        if not values:
            return None
        
        if mode == 'max':
            return max(values)
        else:
            return min(values)
    
    def get_current(self, metric_name: str) -> Optional[float]:
        """Get current (latest) value for a metric"""
        if metric_name not in self.history or not self.history[metric_name]:
            return None
        return self.history[metric_name][-1]
    
    def get_average(self, metric_name: str, last_n: int = 5) -> Optional[float]:
        """Get average of last n values for a metric"""
        if metric_name not in self.history or not self.history[metric_name]:
            return None
        
        values = self.history[metric_name][-last_n:]
        return np.mean(values) if values else None
    
    def save(self, filepath: Path):
        """Save metrics history"""
        try:
            data = {
                'history': self.history,
                'best_values': self.best_values,
                'best_epochs': self.best_epochs,
                'metrics_names': self.metrics_names
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Metrics saved to: {filepath}")
            
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    def load(self, filepath: Path):
        """Load metrics history"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.history = data.get('history', {})
            self.best_values = data.get('best_values', {})
            self.best_epochs = data.get('best_epochs', {})
            
            # Update metrics names
            if 'metrics_names' in data:
                self.metrics_names = data['metrics_names']
            
        except Exception as e:
            print(f"Error loading metrics: {e}")
    
    def print_summary(self):
        """Print summary of metrics"""
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        
        # Current values
        print("\nCurrent Values:")
        print("-" * 40)
        for metric_name in self.metrics_names:
            if metric_name in self.history and self.history[metric_name]:
                current = self.history[metric_name][-1]
                print(f"  {metric_name}: {current:.4f}")
        
        # Best values
        print("\nBest Values:")
        print("-" * 40)
        for metric_name, best_value in self.best_values.items():
            best_epoch = self.best_epochs.get(metric_name, 'N/A')
            print(f"  {metric_name}: {best_value:.4f} (epoch {best_epoch})")
        
        # Statistics
        print("\nStatistics (last 10 epochs):")
        print("-" * 40)
        for metric_name in self.metrics_names:
            if metric_name in self.history and len(self.history[metric_name]) > 0:
                recent_values = self.history[metric_name][-10:]
                mean_val = np.mean(recent_values)
                std_val = np.std(recent_values)
                print(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("="*60)
    
    def plot_metrics(self, save_path: Optional[Path] = None, show: bool = True):
        """Plot metrics history"""
        try:
            import matplotlib.pyplot as plt
            
            # Create subplots
            n_metrics = len(self.metrics_names)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            # Plot each metric
            for idx, metric_name in enumerate(self.metrics_names):
                if metric_name not in self.history or not self.history[metric_name]:
                    continue
                
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                values = self.history[metric_name]
                epochs = range(1, len(values) + 1)
                
                ax.plot(epochs, values, label=metric_name)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(metric_name)
                ax.grid(True, alpha=0.3)
                
                # Mark best value
                if metric_name in self.best_values:
                    best_epoch = self.best_epochs[metric_name]
                    best_value = self.best_values[metric_name]
                    ax.plot(best_epoch + 1, best_value, 'r*', markersize=10)
            
            # Hide empty subplots
            for idx in range(n_metrics, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Metrics plot saved to: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error plotting metrics: {e}")


def format_time(seconds: float) -> str:
    """Format seconds into readable time string"""
    if seconds < 0:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_training_header():
    """Print training header with system info"""
    print("\n" + "="*60)
    print("STUTTERING DETECTION TRAINING")
    print("="*60)
    
    # System info
    device = get_device()
    
    # Python and PyTorch versions
    print(f"\nEnvironment:")
    print(f"  Python: {os.sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    
    # Training info
    print(f"\nTraining Information:")
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Process ID: {os.getpid()}")
    print(f"  Working Directory: {os.getcwd()}")
    
    # Resource monitoring
    monitor_resources()
    
    return device


def create_experiment_id() -> str:
    """Create unique experiment ID"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    random_suffix = ''.join(random.choices('0123456789abcdef', k=4))
    return f"{timestamp}_{random_suffix}"


def save_training_config(config: Dict[str, Any], save_dir: Path):
    """Save training configuration for reproducibility"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = save_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save environment info
    env_info = {
        'python_version': os.sys.version,
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hostname': os.uname().nodename if hasattr(os, 'uname') else 'N/A',
        'platform': os.sys.platform
    }
    
    env_path = save_dir / 'environment_info.json'
    with open(env_path, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    print(f"Configuration saved to: {save_dir}")


def validate_config(config: Dict[str, Any], required_keys: List[Tuple[str, Any]]) -> bool:
    """Validate configuration has required keys with defaults"""
    valid = True
    
    for key_path, default_value in required_keys:
        keys = key_path.split('.')
        current = config
        
        try:
            for i, key in enumerate(keys):
                if i == len(keys) - 1:
                    # Last key - check if exists, set default if not
                    if key not in current:
                        print(f"Missing config key '{key_path}', using default: {default_value}")
                        current[key] = default_value
                else:
                    # Intermediate key - create dict if needed
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                    
        except Exception as e:
            print(f"Error validating config key '{key_path}': {e}")
            valid = False
    
    return valid


# Testing utilities
def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Test time formatting
    print("\nTesting format_time:")
    print(f"  45 seconds: {format_time(45)}")
    print(f"  125 seconds: {format_time(125)}")
    print(f"  7384 seconds: {format_time(7384)}")
    
    # Test early stopping
    print("\nTesting EarlyStopping:")
    early_stop = EarlyStopping(patience=3, verbose=True)
    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62]
    for i, score in enumerate(scores):
        should_stop = early_stop(score, epoch=i)
        print(f"  Epoch {i}: score={score:.2f}, stop={should_stop}")
    
    # Test metric tracker
    print("\nTesting MetricTracker:")
    tracker = MetricTracker(['loss', 'accuracy'])
    for i in range(5):
        tracker.update({
            'loss': 1.0 - i * 0.1,
            'accuracy': 0.5 + i * 0.1
        }, epoch=i)
    tracker.print_summary()
    
    # Test resource monitoring
    print("\nTesting resource monitoring:")
    monitor_resources()
    
    print("\nUtils test complete!")


if __name__ == "__main__":
    test_utils()