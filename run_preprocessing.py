#!/usr/bin/env python3
"""
Simple script to run the underwater dataset preprocessing with optimal settings.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All packages installed successfully!")
    except subprocess.CalledProcessError:
        print("Failed to install packages. Please install manually:")
        print("pip install -r requirements.txt")
        return False
    return True

def run_preprocessing():
    """Run the preprocessing pipeline."""
    
    # Paths
    current_dir = Path(__file__).parent
    input_path = current_dir / "aquarium_pretrain"
    output_path = current_dir / "preprocessed_dataset"
    config_path = current_dir / "preprocessing_config.json"
    
    # Check if input dataset exists
    if not input_path.exists():
        print(f"Input dataset not found at: {input_path}")
        print("Please ensure the aquarium_pretrain folder exists in the current directory.")
        return False
    
    print(f"Input dataset found: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    # Run preprocessing
    cmd = [
        sys.executable, "preprocess_dataset.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--config", str(config_path),
        "--suffix", "balanced_v1"
    ]
    
    print("🚀 Starting preprocessing pipeline...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\nPreprocessing completed successfully!")
        
        # Show output structure
        print("\nOutput structure:")
        for item in sorted(output_path.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(output_path)
                print(f"  {rel_path}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Preprocessing failed with error: {e}")
        return False

def main():
    """Main function."""
    print("Underwater Dataset Preprocessing Runner")
    print("=" * 50)
    
    # Install requirements if needed
    if not install_requirements():
        return
    
    # Run preprocessing
    success = run_preprocessing()
    
    if success:
        print("\nAll done! Your balanced dataset is ready for training.")
        print("\nNext steps:")
        print("1. Use the data_balanced_v1.yaml file for training")
        print("2. Check the statistics folder for detailed analysis")
        print("3. Review the preprocessing report for training recommendations")
    else:
        print("\nPreprocessing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
