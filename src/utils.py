import torch
import numpy as np
from pathlib import Path
import json
import random
import os
import time
import psutil

# GPU monitoring imports
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, mode='max', delta=0.0001):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class CostOptimizer:
    """Monitor training costs and performance for early stopping"""
    
    def __init__(self, max_hours=8.0, target_performance=0.85):
        self.max_hours = max_hours
        self.target_performance = target_performance
        self.start_time = time.time()
        
    def should_stop_training(self, current_performance):
        """Decide if training should stop based on cost and performance"""
        elapsed_hours = (time.time() - self.start_time) / 3600
        
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
        
        return False
    
    def get_elapsed_time(self):
        """Get elapsed training time in hours"""
        return (time.time() - self.start_time) / 3600


def save_checkpoint(model, optimizer, epoch, score, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'score': score,
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['score']


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def monitor_resources():
    """Monitor system resources"""
    print("\n" + "="*50)
    print("RESOURCE MONITORING")
    print("="*50)
    
    # GPU monitoring
    if HAS_GPUTIL and torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"GPU Utilization: {gpu.load*100:.1f}%")
                print(f"GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil*100:.1f}%)")
                print(f"GPU Temperature: {gpu.temperature}°C")
            else:
                # Fallback to PyTorch CUDA info
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1e9
                    cached = torch.cuda.memory_reserved() / 1e9
                    total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"GPU Memory Allocated: {allocated:.2f} GB")
                    print(f"GPU Memory Cached: {cached:.2f} GB")
                    print(f"GPU Memory Total: {total:.2f} GB")
        except Exception as e:
            print(f"GPU monitoring error: {e}")
    else:
        print("GPU monitoring not available")
    
    # CPU and RAM monitoring
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"CPU Usage: {cpu_percent:.1f}%")
        print(f"RAM Usage: {memory.percent:.1f}% ({memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB)")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        print(f"Disk Usage: {disk.percent:.1f}% ({disk.used/1e9:.1f}/{disk.total/1e9:.1f} GB)")
    except Exception as e:
        print(f"System monitoring error: {e}")
    
    print("="*50)


def estimate_training_time(num_epochs, samples_per_epoch, batch_size, time_per_batch):
    """Estimate total training time"""
    batches_per_epoch = samples_per_epoch // batch_size
    total_batches = num_epochs * batches_per_epoch
    estimated_seconds = total_batches * time_per_batch
    estimated_hours = estimated_seconds / 3600
    
    print(f"\nTraining Time Estimation:")
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Total batches: {total_batches}")
    print(f"Estimated time: {estimated_hours:.2f} hours")
    
    return estimated_hours


def cleanup_checkpoints(checkpoint_dir, keep_last_n=3):
    """Clean up old checkpoints to save disk space"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return
    
    # Get all checkpoint files (excluding best_model.pth)
    checkpoint_files = []
    for file in checkpoint_path.glob("checkpoint_epoch_*.pth"):
        checkpoint_files.append(file)
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
    
    # Remove old checkpoints
    files_to_remove = checkpoint_files[keep_last_n:]
    for file in files_to_remove:
        try:
            file.unlink()
            print(f"Removed old checkpoint: {file.name}")
        except Exception as e:
            print(f"Error removing {file.name}: {e}")


class MetricTracker:
    """Track metrics during training"""
    
    def __init__(self, metrics_names):
        self.metrics_names = metrics_names
        self.history = {name: [] for name in metrics_names}
        
    def update(self, metrics_dict):
        """Update metrics"""
        for name in self.metrics_names:
            if name in metrics_dict:
                self.history[name].append(metrics_dict[name])
    
    def get_best(self, metric_name, mode='max'):
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
    
    def get_current(self, metric_name):
        """Get current (latest) value for a metric"""
        if metric_name not in self.history or not self.history[metric_name]:
            return None
        return self.history[metric_name][-1]
    
    def save(self, filepath):
        """Save metrics history"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def print_summary(self):
        """Print summary of metrics"""
        print("\n" + "="*40)
        print("METRICS SUMMARY")
        print("="*40)
        
        for metric_name in self.metrics_names:
            if metric_name in self.history and self.history[metric_name]:
                values = self.history[metric_name]
                current = values[-1]
                best = max(values) if 'loss' not in metric_name.lower() else min(values)
                print(f"{metric_name}: {current:.4f} (best: {best:.4f})")
        
        print("="*40)


def format_time(seconds):
    """Format seconds into readable time string"""
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
    print("STUTTERING DETECTION TRAINING - VAST.AI")
    print("="*60)
    
    # System info
    device = get_device()
    
    # Training info
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Process ID: {os.getpid()}")
    
    # Resource monitoring
    monitor_resources()
    
    return device