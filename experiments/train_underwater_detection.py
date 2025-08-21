#!/usr/bin/env python3
"""
Unified training script for all three underwater object detection approaches
Run specific approaches or compare all three
"""

import argparse
import os
import sys
import subprocess
import json
import yaml
from datetime import datetime
from pathlib import Path

def setup_environment():
    """Setup the training environment"""
    print("Setting up environment for underwater object detection training...")
    
    # Check GPU availability
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
        else:
            print("⚠ No GPU found. Training will use CPU (slow).")
    except ImportError:
        print("⚠ TensorFlow not installed. Please install requirements first.")
        return False
    
    return True

def run_approach(approach: str, config_path: str, extra_args: list = None):
    """Run training for a specific approach"""
    if extra_args is None:
        extra_args = []
    
    approach_scripts = {
        'yolov8': 'approach1_yolov8/yolov8_underwater.py',
        'faster_rcnn': 'approach2_faster_rcnn/faster_rcnn_underwater.py', 
        'detr': 'approach3_detr/detr_underwater.py'
    }
    
    if approach not in approach_scripts:
        raise ValueError(f"Unknown approach: {approach}. Choose from {list(approach_scripts.keys())}")
    
    script_path = os.path.join(os.path.dirname(__file__), approach_scripts[approach])
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    # Build command
    cmd = [sys.executable, script_path, '--config', config_path]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*50}")
    print(f"Starting {approach.upper()} training...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ {approach.upper()} training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {approach.upper()} training failed with error: {e}")
        return False

def run_comparative_experiment():
    """Run all three approaches and compare results"""
    print("\n" + "="*60)
    print("COMPARATIVE EXPERIMENT: All Three Approaches")
    print("="*60)
    
    approaches = {
        'yolov8': 'configs/yolov8_config.yaml',
        'faster_rcnn': 'configs/faster_rcnn_config.yaml',
        'detr': 'configs/detr_config.yaml'
    }
    
    results = {}
    
    for approach, config in approaches.items():
        print(f"\n{'*'*40}")
        print(f"Training Approach: {approach.upper()}")
        print(f"{'*'*40}")
        
        config_path = os.path.join(os.path.dirname(__file__), config)
        success = run_approach(approach, config_path)
        results[approach] = {'success': success}
        
        if not success:
            print(f"⚠ Skipping remaining approaches due to {approach} failure")
            break
    
    # Generate comparative report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"comparative_results_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("COMPARATIVE EXPERIMENT COMPLETED")
    print(f"Results saved to: {report_path}")
    print("="*60)

def validate_config(config_path: str):
    """Validate configuration file"""
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['approach', 'dataset', 'training']
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing required key in config: {key}")
                return False
        
        print(f"✓ Config file validated: {config_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error reading config file: {e}")
        return False

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if not os.path.exists(requirements_path):
        print("✗ requirements.txt not found")
        return False
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', requirements_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Unified training script for underwater object detection approaches'
    )
    
    parser.add_argument(
        '--approach', 
        choices=['yolov8', 'faster_rcnn', 'detr', 'all'], 
        default='all',
        help='Which approach to train (default: all)'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to configuration file (auto-selected if not provided)'
    )
    
    parser.add_argument(
        '--install-requirements', 
        action='store_true',
        help='Install required packages before training'
    )
    
    parser.add_argument(
        '--setup-only', 
        action='store_true',
        help='Only setup environment, do not run training'
    )
    
    parser.add_argument(
        '--visualize-attention', 
        action='store_true',
        help='Visualize attention maps (for DETR approach)'
    )
    
    args = parser.parse_args()
    
    # Install requirements if requested
    if args.install_requirements:
        if not install_requirements():
            return 1
    
    # Setup environment
    if not setup_environment():
        return 1
    
    if args.setup_only:
        print("Environment setup completed.")
        return 0
    
    # Determine configuration files
    if args.config:
        config_paths = {args.approach: args.config}
    else:
        base_dir = os.path.dirname(__file__)
        config_paths = {
            'yolov8': os.path.join(base_dir, 'configs/yolov8_config.yaml'),
            'faster_rcnn': os.path.join(base_dir, 'configs/faster_rcnn_config.yaml'),
            'detr': os.path.join(base_dir, 'configs/detr_config.yaml')
        }
    
    # Run training
    if args.approach == 'all':
        run_comparative_experiment()
    else:
        config_path = config_paths[args.approach]
        
        if not validate_config(config_path):
            return 1
        
        extra_args = []
        if args.visualize_attention and args.approach == 'detr':
            extra_args.append('--visualize_attention')
        
        success = run_approach(args.approach, config_path, extra_args)
        return 0 if success else 1

if __name__ == '__main__':
    exit(main())
