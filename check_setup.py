#!/usr/bin/env python3
"""
Debug script to verify your setup
Save as: check_setup.py
"""

import os
import sys
from pathlib import Path
import json
import torch

def check_setup():
    print("="*60)
    print("CHECKING STUTTERING DETECTION SETUP")
    print("="*60)
    
    # Check new files exist
    print("\n1. Checking required files:")
    files_to_check = [
        ('feature_preprocessing.py', 'Feature preprocessing module'),
        ('fast_trainer.py', 'Fast trainer module'),
        ('config/config.yaml', 'Configuration file'),
    ]
    
    all_files_exist = True
    for file, desc in files_to_check:
        exists = Path(file).exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {file} - {desc}")
        if not exists:
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Missing required files! Please add them to your project.")
        return False
    
    # Check if features are extracted
    print("\n2. Checking pre-extracted features:")
    features_path = Path('data/processed/features')
    
    if features_path.exists():
        feature_files = list(features_path.glob('*.npy'))
        info_file = features_path / 'feature_info.json'
        
        if info_file.exists():
            with open(info_file, 'r') as f:
                info = json.load(f)
            print(f"   ✓ Features found: {len(feature_files)} files")
            print(f"   ✓ Feature dimension: {info.get('feature_dim', 'unknown')}")
            print(f"   ✓ Device used: {info.get('device', 'unknown')}")
            print(f"   ✓ Extraction date: {info.get('extraction_date', 'unknown')}")
        else:
            print(f"   ⚠ Found {len(feature_files)} feature files but no info.json")
    else:
        print("   ✗ No pre-extracted features found")
        print("   → Run: python main.py --mode extract-features")
    
    # Check GPU
    print("\n3. Checking GPU:")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ✗ No GPU available - training will be slow!")
    
    # Check if old trainer is being imported
    print("\n4. Checking imports:")
    sys.path.insert(0, os.getcwd())
    
    try:
        import feature_preprocessing
        print("   ✓ feature_preprocessing module can be imported")
    except ImportError as e:
        print(f"   ✗ Cannot import feature_preprocessing: {e}")
    
    try:
        import fast_trainer
        print("   ✓ fast_trainer module can be imported")
    except ImportError as e:
        print(f"   ✗ Cannot import fast_trainer: {e}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if not features_path.exists() or len(list(features_path.glob('*.npy'))) == 0:
        print("1. Extract features first:")
        print("   python main.py --mode extract-features")
        print("")
    
    print("2. Run training with:")
    print("   python main.py --mode train --fast")
    print("")
    print("3. Or run everything:")
    print("   python main.py --mode all --fast")
    
    return True

if __name__ == "__main__":
    check_setup()