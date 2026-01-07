"""
Dataset Download Script
Downloads ASL Alphabet and FER-2013 datasets from Kaggle
"""

import os
import subprocess
import zipfile
from pathlib import Path


DATASETS_DIR = Path("datasets")


def check_kaggle_credentials():
    """Check if Kaggle API credentials are set up."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("ERROR: Kaggle API credentials not found!")
        print("\nTo set up Kaggle API:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token' to download kaggle.json")
        print(f"  3. Place kaggle.json in: {kaggle_dir}")
        print("  4. Run this script again")
        return False
    
    # Ensure file permissions are correct (Windows compatible)
    print(f"✓ Found Kaggle credentials at {kaggle_json}")
    return True


def download_dataset(dataset_name: str, output_name: str):
    """Download and extract a Kaggle dataset."""
    output_path = DATASETS_DIR / output_name
    
    if output_path.exists():
        print(f"✓ Dataset already exists: {output_path}")
        return True
    
    print(f"\nDownloading {dataset_name}...")
    
    try:
        # Create datasets directory
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download using Kaggle API
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(DATASETS_DIR)],
            check=True
        )
        
        # Find and extract zip file
        zip_name = dataset_name.split("/")[-1] + ".zip"
        zip_path = DATASETS_DIR / zip_name
        
        if zip_path.exists():
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            
            # Remove zip file
            zip_path.unlink()
            print(f"✓ Dataset extracted to: {output_path}")
            return True
        else:
            print(f"ERROR: Zip file not found: {zip_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"ERROR downloading dataset: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Download all required datasets."""
    print("=" * 60)
    print("BridgeComm Dataset Downloader")
    print("=" * 60)
    
    # Check Kaggle credentials
    if not check_kaggle_credentials():
        return
    
    datasets = [
        ("grassknoted/asl-alphabet", "asl-alphabet"),
        ("msambare/fer2013", "fer2013"),
    ]
    
    success = True
    for dataset_name, output_name in datasets:
        if not download_dataset(dataset_name, output_name):
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("All datasets downloaded successfully!")
        print("\nNext steps:")
        print("  1. Run: python train_asl_gestures.py")
        print("  2. Run: python train_emotion_model.py")
    else:
        print("Some datasets failed to download.")
        print("You can manually download from Kaggle and extract to datasets/")
    print("=" * 60)


if __name__ == "__main__":
    main()
