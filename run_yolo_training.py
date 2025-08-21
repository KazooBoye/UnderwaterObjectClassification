#!/usr/bin/env python3
"""
YOLO Training Runner Script
Simple interface for training YOLO model with underwater object detection dataset
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import tensorflow as tf
        import cv2
        import numpy as np
        import yaml
        print(f"✓ TensorFlow {tf.__version__}")
        print(f"✓ OpenCV {cv2.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Missing requirement: {e}")
        print("Please install requirements: pip install -r yolo_requirements.txt")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            return True
        else:
            print("ℹ No GPU found, will use CPU")
            return False
    except Exception as e:
        print(f"⚠ GPU check failed: {e}")
        return False

def check_dataset(data_path):
    """Validate dataset structure"""
    if not os.path.exists(data_path):
        print(f"✗ Dataset not found: {data_path}")
        return False
    
    required_files = [
        'data.yaml',
        'train/images',
        'train/labels', 
        'valid/images',
        'valid/labels'
    ]
    
    for file_path in required_files:
        full_path = os.path.join(data_path, file_path)
        if not os.path.exists(full_path):
            print(f"✗ Missing: {full_path}")
            return False
    
    print(f"✓ Dataset structure valid: {data_path}")
    
    # Count samples
    train_images = len(list(Path(data_path, 'train/images').glob('*.jpg')))
    val_images = len(list(Path(data_path, 'valid/images').glob('*.jpg')))
    
    print(f"  Training images: {train_images}")
    print(f"  Validation images: {val_images}")
    
    return True

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "yolo_requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install requirements")
        return False

def run_training(data_path, output_dir, epochs, batch_size):
    """Run YOLO training"""
    cmd = [
        sys.executable,
        "yolo_tensorflow.py",
        "--data", data_path,
        "--output", output_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size)
    ]
    
    print(f"Starting training with command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✓ Training completed successfully!")
        print(f"Results saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="YOLO Training Runner for Underwater Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_yolo_training.py                              # Use defaults
  python run_yolo_training.py --epochs 50 --batch-size 8   # Custom settings
  python run_yolo_training.py --install-deps               # Install dependencies first
        """
    )
    
    parser.add_argument('--data', type=str, default='aquarium_pretrain',
                       help='Path to dataset directory (default: aquarium_pretrain)')
    parser.add_argument('--output', type=str, default='yolo_training_results',
                       help='Output directory for training results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size (default: 16)')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install required dependencies before training')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check system requirements without training')
    
    args = parser.parse_args()
    
    print("YOLO Underwater Object Detection Training")
    print("=" * 50)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_requirements():
            return 1
        print()
    
    # Check requirements
    print("Checking system requirements...")
    if not check_requirements():
        print("\nPlease install missing requirements and try again.")
        return 1
    
    # Check GPU
    print("\nChecking GPU availability...")
    has_gpu = check_gpu()
    
    # Adjust batch size for CPU
    if not has_gpu and args.batch_size > 8:
        print(f"⚠ Reducing batch size from {args.batch_size} to 8 for CPU training")
        args.batch_size = 8
    
    # Check dataset
    print(f"\nChecking dataset: {args.data}")
    if not check_dataset(args.data):
        print("\nPlease ensure your dataset follows YOLO format structure.")
        return 1
    
    # Exit if check-only mode
    if args.check_only:
        print("\n✓ All checks passed! System is ready for training.")
        return 0
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Start training
    print(f"\nStarting YOLO training...")
    print(f"Dataset: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hardware: {'GPU' if has_gpu else 'CPU'}")
    
    if run_training(args.data, args.output, args.epochs, args.batch_size):
        print(f"\n🎉 Training completed!")
        print(f"📁 Results: {args.output}/")
        print(f"📊 Best model: {args.output}/best_model.h5")
        return 0
    else:
        return 1

if __name__ == '__main__':
    sys.exit(main())
