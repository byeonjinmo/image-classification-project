#!/usr/bin/env python3
"""
이미지 데이터셋 클래스 및 유틸리티
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    """폴더에서 이미지를 로드하는 커스텀 데이터셋"""
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(folder_path) 
                              if os.path.isdir(os.path.join(folder_path, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(folder_path, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_transforms():
    """
    학습 및 검증용 데이터 변환 함수 반환
    
    Returns:
        tuple: (학습용 변환, 검증용 변환)
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_data(data_dir, batch_size=32, val_split=0.2, num_workers=4):
    """
    데이터 로드 및 학습/검증 세트로 분할
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        batch_size (int): 배치 크기
        val_split (float): 검증 세트 비율 (0-1)
        num_workers (int): 데이터 로딩에 사용할 워커 수
        
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    train_transform, val_transform = get_data_transforms()
    
    # 전체 데이터셋 로드
    dataset = ImageDataset(data_dir, transform=train_transform)
    class_names = dataset.classes
    
    # 분할 크기 계산
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    # 분할 수행
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 학습 및 검증 데이터셋에 적절한 변환 적용
    train_dataset_transformed = ImageDataset(data_dir, transform=train_transform)
    val_dataset_transformed = ImageDataset(data_dir, transform=val_transform)
    
    # 분할에 기반하여 샘플 업데이트
    train_dataset_transformed.samples = [dataset.samples[i] for i in train_dataset.indices]
    val_dataset_transformed.samples = [dataset.samples[i] for i in val_dataset.indices]
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset_transformed, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset_transformed, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, class_names, train_dataset_transformed, val_dataset_transformed

def get_class_distribution(dataset):
    """
    데이터셋의 클래스 분포 계산
    
    Args:
        dataset: 데이터셋 객체
        
    Returns:
        dict: 클래스별 샘플 수
    """
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    return class_counts

def visualize_batch(dataloader, class_names, num_images=16):
    """
    데이터로더에서 배치를 시각화
    
    Args:
        dataloader: 데이터 로더 객체
        class_names (list): 클래스 이름 목록
        num_images (int): 표시할 이미지 수
    """
    import matplotlib.pyplot as plt
    
    # 배치 가져오기
    images, labels = next(iter(dataloader))
    
    # 이미지 표시
    plt.figure(figsize=(12, 8))
    for i in range(min(num_images, len(images))):
        # Tensor를 이미지로 변환
        img = images[i].permute(1, 2, 0).numpy()
        
        # 정규화 복원
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.subplot(4, 4, i+1)
        plt.imshow(img)
        plt.title(class_names[labels[i]])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 테스트 용도
    import argparse
    
    parser = argparse.ArgumentParser(description="데이터셋 테스트")
    parser.add_argument("--data_dir", required=True, help="데이터 디렉토리 경로")
    
    args = parser.parse_args()
    
    # 데이터 로드
    train_loader, val_loader, class_names, train_dataset, val_dataset = load_data(args.data_dir)
    
    # 정보 출력
    print(f"클래스: {class_names}")
    print(f"학습 샘플 수: {len(train_dataset)}")
    print(f"검증 샘플 수: {len(val_dataset)}")
    
    # 클래스 분포 확인
    class_dist = get_class_distribution(train_dataset)
    for cls_name, count in class_dist.items():
        print(f"  {cls_name}: {count} 샘플")
    
    # 배치 시각화
    visualize_batch(train_loader, class_names)
