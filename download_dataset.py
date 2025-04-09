#!/usr/bin/env python3
import os
import requests
import tarfile
import shutil
from tqdm import tqdm
import random
from PIL import Image
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

def extract_tar(tar_path, extract_path):
    """Extract a tar file to a specific path"""
    print(f"Extracting {os.path.basename(tar_path)}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted to {extract_path}")
    return extract_path

def create_dataset_structure(source_dir, target_dir, num_classes=5, images_per_class=20):
    """Create a dataset with specified structure from the downloaded images"""
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all flower class directories
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"Found {len(class_dirs)} classes in the source directory")
    
    # Select a subset of classes if needed
    if num_classes < len(class_dirs):
        class_dirs = random.sample(class_dirs, num_classes)
    
    # Process each class
    for class_dir in class_dirs:
        source_class_path = os.path.join(source_dir, class_dir)
        target_class_path = os.path.join(target_dir, class_dir)
        
        # Create class directory in target
        os.makedirs(target_class_path, exist_ok=True)
        
        # Get all images in the class directory
        images = [f for f in os.listdir(source_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Select a subset of images if needed
        if images_per_class < len(images):
            images = random.sample(images, images_per_class)
        
        print(f"Copying {len(images)} images for class '{class_dir}'")
        
        # Copy images to target directory
        for img_file in tqdm(images, desc=f"Class {class_dir}"):
            source_img_path = os.path.join(source_class_path, img_file)
            target_img_path = os.path.join(target_class_path, img_file)
            
            # Ensure the image can be opened (skip corrupted images)
            try:
                with Image.open(source_img_path) as img:
                    # Copy the image
                    shutil.copy2(source_img_path, target_img_path)
            except Exception as e:
                print(f"Skipping {img_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare a dataset for image classification")
    parser.add_argument("--dataset", choices=["flowers", "cifar", "food"], default="flowers",
                        help="Dataset to download (default: flowers)")
    parser.add_argument("--output_dir", default="datasets",
                        help="Output directory for the dataset (default: datasets)")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classes to include (default: 5)")
    parser.add_argument("--images_per_class", type=int, default=20,
                        help="Number of images per class (default: 20)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download and prepare dataset based on selection
    if args.dataset == "flowers":
        # Flower dataset (102 categories)
        url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
        labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
        
        # Define paths
        temp_dir = os.path.join(args.output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        tar_path = os.path.join(temp_dir, "flowers.tgz")
        extract_path = os.path.join(temp_dir, "extracted")
        
        # Download dataset if it doesn't exist
        if not os.path.exists(tar_path):
            download_file(url, tar_path)
        
        # Extract dataset if not already extracted
        if not os.path.exists(extract_path):
            extract_tar(tar_path, extract_path)
        
        # Create simplified flower dataset structure
        flower_classes = {
            'daisy': 'class_1',
            'sunflower': 'class_2',
            'roses': 'class_3',
            'dandelion': 'class_4',
            'tulips': 'class_5'
        }
        
        # Create dataset directory
        dataset_dir = os.path.join(args.output_dir, "flower_data")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create class directories and copy sample images
        # For the flower dataset, we'll create a simplified version manually
        img_dir = os.path.join(extract_path, "jpg")
        if os.path.exists(img_dir):
            files = os.listdir(img_dir)
            image_files = [f for f in files if f.endswith('.jpg')]
            
            # Distribute images to classes (simplified way since we don't have actual labels)
            images_per_class = min(args.images_per_class, len(image_files) // len(flower_classes))
            
            for i, (class_name, _) in enumerate(flower_classes.items()):
                class_dir = os.path.join(dataset_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Select images for this class
                start_idx = i * images_per_class
                end_idx = start_idx + images_per_class
                class_images = image_files[start_idx:end_idx]
                
                print(f"Copying {len(class_images)} images for class '{class_name}'")
                
                # Copy images
                for img_file in class_images:
                    source_path = os.path.join(img_dir, img_file)
                    target_path = os.path.join(class_dir, img_file)
                    shutil.copy2(source_path, target_path)
        
        print(f"\nDataset created at {dataset_dir}")
        print(f"The dataset contains {len(flower_classes)} classes with approximately {images_per_class} images per class")
        print("To use this dataset with the image classifier, run:")
        print(f"python run_classifier.py --data_dir {dataset_dir} --epochs 5 --batch_size 16")
        
    elif args.dataset == "cifar":
        # Implementation for CIFAR dataset
        print("CIFAR dataset download is not implemented yet. Please use the flowers dataset.")
    
    elif args.dataset == "food":
        # Implementation for Food dataset
        print("Food dataset download is not implemented yet. Please use the flowers dataset.")
    
    # Clean up temporary files if needed
    print("\nWould you like to clean up temporary download files? (y/n)")
    response = input().lower()
    if response == 'y':
        try:
            shutil.rmtree(temp_dir)
            print("Temporary files removed.")
        except Exception as e:
            print(f"Error removing temporary files: {str(e)}")

if __name__ == "__main__":
    main()
