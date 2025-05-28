#!/usr/bin/env python3
"""Setup script for Vast.ai environment"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("requirements.txt not found!")
        sys.exit(1)

def setup_directories():
    """Create necessary directories"""
    dirs = ["data/raw", "data/processed", "data/splits", "checkpoints", "logs"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("No GPU available")
    except ImportError:
        print("PyTorch not installed")

def main():
    print("Setting up Vast.ai environment...")
    
    # Install dependencies
    install_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Check GPU
    check_gpu()
    
    print("Setup completed!")

if __name__ == "__main__":
    main()