#!/usr/bin/env python3
import os
import requests
from tqdm import tqdm
import zipfile
import shutil
import argparse

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    print(f"Downloading {os.path.basename(destination)}...")
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))
    
    print(f"Downloaded to {destination}")
    return destination

def main():
    parser = argparse.ArgumentParser(description="Download and prepare a flower dataset for image classification")
    parser.add_argument("--output_dir", default="flower_dataset",
                        help="Output directory for the dataset (default: flower_dataset)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download a small flower dataset from Kaggle (publicly available)
    # This is a smaller dataset that's easier to download
    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    
    # Define paths
    temp_dir = os.path.join(args.output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    tgz_path = os.path.join(temp_dir, "flower_photos.tgz")
    
    # Download dataset
    if not os.path.exists(tgz_path):
        try:
            download_file(url, tgz_path)
        except Exception as e:
            print(f"Error downloading the dataset: {str(e)}")
            print("Please check your internet connection and try again.")
            return
    
    # Extract the dataset
    print("Extracting dataset...")
    try:
        import tarfile
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(path=args.output_dir)
        print(f"Dataset extracted to {args.output_dir}/flower_photos")
    except Exception as e:
        print(f"Error extracting the dataset: {str(e)}")
        return
    
    # The dataset is already structured correctly with class folders
    dataset_dir = os.path.join(args.output_dir, "flower_photos")
    
    # List the classes and count images
    class_dirs = [d for d in os.listdir(dataset_dir) 
                 if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')]
    
    print("\nDataset information:")
    print(f"Location: {dataset_dir}")
    print(f"Classes: {len(class_dirs)}")
    
    total_images = 0
    for class_dir in class_dirs:
        class_path = os.path.join(dataset_dir, class_dir)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
        total_images += len(images)
        print(f"  - {class_dir}: {len(images)} images")
    
    print(f"Total images: {total_images}")
    
    # Instructions for using the dataset
    print("\nTo use this dataset with the image classifier, run:")
    print(f"python scripts/run_classifier.py --data_dir {dataset_dir} --epochs 5 --batch_size 16")
    
    # Clean up temporary files
    try:
        shutil.rmtree(temp_dir)
        print("Temporary files cleaned up.")
    except Exception as e:
        print(f"Error cleaning up temporary files: {str(e)}")

if __name__ == "__main__":
    main()
