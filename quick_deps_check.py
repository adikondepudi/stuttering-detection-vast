#!/usr/bin/env python3
"""
Quick dependency checker - tests if basic imports work
without needing the heavy ML libraries
"""

def check_basic_imports():
    """Check if basic Python imports work"""
    print("Checking basic Python imports...")
    
    required_basic = [
        'json', 'yaml', 'pathlib', 'numpy', 'torch'
    ]
    
    optional_heavy = [
        'transformers', 'librosa', 'torchaudio', 'soundfile'
    ]
    
    # Test basic imports
    failed_basic = []
    for module in required_basic:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - REQUIRED")
            failed_basic.append(module)
    
    # Test optional imports (these can fail, we'll mock them)
    print("\nChecking optional heavy imports (can be mocked)...")
    for module in optional_heavy:
        try:
            __import__(module)
            print(f"✓ {module}")
        except (ImportError, OSError) as e:
            print(f"⚠ {module} - will be mocked in tests ({type(e).__name__})")
    
    if failed_basic:
        print(f"\n❌ Missing required basic imports: {failed_basic}")
        print("Install with: pip install torch numpy pyyaml")
        return False
    else:
        print(f"\n✅ All basic imports work! Heavy ML libraries will be mocked.")
        return True

def check_file_structure():
    """Check if the project file structure exists"""
    print("\nChecking project file structure...")
    
    from pathlib import Path
    
    required_files = [
        'src/__init__.py',
        'src/model.py',
        'src/utils.py',
        'config/config.yaml'
    ]
    
    optional_files = [
        'src/data_preprocessing.py',
        'src/feature_extraction.py',
        'src/dataset.py',
        'src/train.py'
    ]
    
    missing_required = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - REQUIRED")
            missing_required.append(file_path)
    
    print("\nOptional files:")
    for file_path in optional_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"⚠ {file_path} - will be mocked")
    
    if missing_required:
        print(f"\n❌ Missing required files: {missing_required}")
        return False
    else:
        print(f"\n✅ Required project structure exists!")
        return True

if __name__ == "__main__":
    print("="*50)
    print("QUICK DEPENDENCIES & STRUCTURE CHECK")
    print("="*50)
    
    deps_ok = check_basic_imports()
    structure_ok = check_file_structure()
    
    print("\n" + "="*50)
    if deps_ok and structure_ok:
        print("✅ Ready to run debug tests!")
        print("Run: python debug_test.py")
    else:
        print("❌ Fix the issues above first")