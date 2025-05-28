#!/usr/bin/env python3
"""
Training monitoring script for Vast.ai
Use this to monitor training progress without attaching to tmux session
"""

import json
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(metrics_file):
    """Load training metrics from file"""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        return None
    except Exception as e:
        print(f"Unexpected error loading metrics: {e}")
        return None

def print_current_status(metrics_file, results_file):
    """Print current training status"""
    print("\n" + "="*60)
    print("TRAINING STATUS MONITOR")
    print("="*60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if training is complete
    if Path(results_file).exists():
        print("Status: TRAINING COMPLETED ✓")
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            test_metrics = results.get('test_metrics', {})
            training_info = results.get('training_info', {})
            
            print(f"\nFinal Test Results:")
            print(f"  Macro F1: {test_metrics.get('macro_f1', 0):.4f}")
            print(f"  Weighted F1: {test_metrics.get('weighted_f1', 0):.4f}")
            print(f"  UAR: {test_metrics.get('uar', 0):.4f}")
            
            if 'total_training_time_formatted' in training_info:
                print(f"  Total Training Time: {training_info['total_training_time_formatted']}")
            
        except Exception as e:
            print(f"Error reading results: {e}")
        
        return True
    
    # Load current metrics
    metrics = load_metrics(metrics_file)
    if metrics is None:
        print("Status: WAITING FOR TRAINING TO START...")
        print("No metrics file found yet.")
        return False
    
    print("Status: TRAINING IN PROGRESS...")
    
    # Get latest values
    train_loss = metrics.get('train_loss', [])
    val_macro_f1 = metrics.get('val_macro_f1', [])
    val_uar = metrics.get('val_uar', [])
    
    if not train_loss:
        print("No training data available yet.")
        return False
    
    current_epoch = len(train_loss)
    print(f"Current Epoch: {current_epoch}")
    
    if train_loss:
        print(f"Latest Training Loss: {train_loss[-1]:.4f}")
    
    if val_macro_f1:
        print(f"Latest Validation F1: {val_macro_f1[-1]:.4f}")
        print(f"Best Validation F1: {max(val_macro_f1):.4f}")
    
    if val_uar:
        print(f"Latest UAR: {val_uar[-1]:.4f}")
    
    # Show per-class F1 scores if available
    class_names = ['Prolongation', 'Interjection', 'Word Repetition', 'Sound Repetition', 'Blocks']
    print(f"\nPer-class F1 scores (latest):")
    for class_name in class_names:
        key = f'val_{class_name}_f1'
        if key in metrics and metrics[key]:
            print(f"  {class_name}: {metrics[key][-1]:.4f}")
    
    return False

def plot_training_curves(metrics_file, save_path=None):
    """Plot training curves"""
    metrics = load_metrics(metrics_file)
    if metrics is None:
        print("No metrics available for plotting.")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    epochs = range(1, len(metrics.get('train_loss', [])) + 1)
    
    # Loss plot
    if 'train_loss' in metrics and metrics['train_loss']:
        axes[0,0].plot(epochs, metrics['train_loss'], label='Train Loss', color='blue')
        if 'val_loss' in metrics and metrics['val_loss']:
            axes[0,0].plot(epochs, metrics['val_loss'], label='Val Loss', color='red')
        axes[0,0].set_title('Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
    
    # F1 Score plot
    if 'val_macro_f1' in metrics and metrics['val_macro_f1']:
        axes[0,1].plot(epochs, metrics['val_macro_f1'], label='Macro F1', color='green')
        if 'val_weighted_f1' in metrics and metrics['val_weighted_f1']:
            axes[0,1].plot(epochs, metrics['val_weighted_f1'], label='Weighted F1', color='orange')
        axes[0,1].set_title('F1 Scores')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('F1 Score')
        axes[0,1].legend()
        axes[0,1].grid(True)
    
    # UAR plot
    if 'val_uar' in metrics and metrics['val_uar']:
        axes[1,0].plot(epochs, metrics['val_uar'], label='UAR', color='purple')
        axes[1,0].set_title('Unweighted Average Recall')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('UAR')
        axes[1,0].legend()
        axes[1,0].grid(True)
    
    # Per-class F1 scores
    class_names = ['Prolongation', 'Interjection', 'Word Repetition', 'Sound Repetition', 'Blocks']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, class_name in enumerate(class_names):
        key = f'val_{class_name}_f1'
        if key in metrics and metrics[key]:
            axes[1,1].plot(epochs, metrics[key], label=class_name, color=colors[i])
    
    axes[1,1].set_title('Per-class F1 Scores')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('F1 Score')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def monitor_training(metrics_file, results_file, interval=30, plot=False):
    """Monitor training with automatic updates"""
    print("Starting training monitor...")
    print(f"Checking every {interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            training_complete = print_current_status(metrics_file, results_file)
            
            if training_complete:
                print("\nTraining completed! Monitor stopping.")
                break
            
            if plot:
                try:
                    plot_training_curves(metrics_file, 'training_progress.png')
                except Exception as e:
                    print(f"Error creating plot: {e}")
            
            print(f"\nNext update in {interval} seconds...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

def main():
    parser = argparse.ArgumentParser(description="Monitor stuttering detection training progress")
    
    parser.add_argument('--metrics-file', 
                       default='checkpoints/training_metrics.json',
                       help='Path to training metrics file')
    
    parser.add_argument('--results-file',
                       default='checkpoints/test_results.json', 
                       help='Path to final results file')
    
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds (default: 30)')
    
    parser.add_argument('--plot', action='store_true',
                       help='Generate training plots')
    
    parser.add_argument('--once', action='store_true',
                       help='Check status once and exit')
    
    args = parser.parse_args()
    
    if args.once:
        print_current_status(args.metrics_file, args.results_file)
        if args.plot:
            plot_training_curves(args.metrics_file, 'training_progress.png')
    else:
        monitor_training(args.metrics_file, args.results_file, args.interval, args.plot)

if __name__ == "__main__":
    main()