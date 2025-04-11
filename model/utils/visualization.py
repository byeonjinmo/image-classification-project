
#!/usr/bin/env python3
"""
모델 시각화를 위한 유틸리티 모듈
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE

def plot_training_progress(history, output_path):
    """
    학습 진행 과정을 시각화
    
    Args:
        history (dict): 학습 히스토리 {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        output_path (str): 결과 이미지 저장 경로
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 두 개의 서브플롯이 있는 그림 생성
    plt.figure(figsize=(12, 5))
    
    # 학습 및 검증 손실 플롯
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 학습 및 검증 정확도 플롯
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_samples(dataset, class_names, num_samples=5, output_path=None):
    """
    각 클래스별 샘플 이미지 시각화
    
    Args:
        dataset: 시각화할 데이터셋
        class_names (list): 클래스 이름 리스트
        num_samples (int): 각 클래스당 표시할 샘플 수
        output_path (str, optional): 결과 이미지 저장 경로
    """
    fig, axes = plt.subplots(len(class_names), num_samples, 
                            figsize=(num_samples*2, len(class_names)*2))
    
    # 정규화 역변환 함수
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    for class_idx, class_name in enumerate(class_names):
        # 해당 클래스의 모든 이미지 가져오기
        class_images = [img_path for img_path, label in dataset.samples 
                       if label == dataset.class_to_idx[class_name]]
        
        # 랜덤 샘플 선택
        if len(class_images) >= num_samples:
            sample_paths = np.random.choice(class_images, num_samples, replace=False)
        else:
            sample_paths = class_images
        
        # 이미지 표시
        for j, img_path in enumerate(sample_paths):
            if j < num_samples:
                img = Image.open(img_path).convert('RGB')
                img = dataset.transform(img)  # 데이터셋의 변환 적용
                img = inv_normalize(img)
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                
                if len(class_names) > 1:
                    ax = axes[class_idx, j]
                else:
                    ax = axes[j]
                
                ax.imshow(img)
                ax.set_title(class_name)
                ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_tsne_visualization(features, labels, class_names, output_path=None):
    """
    t-SNE를 사용하여 특성 시각화
    
    Args:
        features (numpy.ndarray): 특성 벡터, shape [n_samples, n_features]
        labels (list): 레이블 리스트
        class_names (list): 클래스 이름 리스트
        output_path (str, optional): 결과 이미지 저장 경로
    """
    # 특성 차원 확인 및 플래튼
    features = features.reshape(features.shape[0], -1)
    
    # t-SNE 차원 축소 적용
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # t-SNE 플롯
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        mask = np.array(labels) == i
        plt.scatter(
            features_tsne[mask, 0], features_tsne[mask, 1],
            label=class_name, alpha=0.7
        )
    
    plt.title('t-SNE Visualization of Features')
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def generate_class_activation_maps(model, val_dataset, class_names, device, output_path=None, num_samples=5):
    """
    클래스 활성화 맵(CAM) 생성
    
    Args:
        model: ResNet 모델
        val_dataset: 검증 데이터셋
        class_names (list): 클래스 이름 리스트
        device: 연산 장치 (CPU/GPU)
        output_path (str, optional): 결과 이미지 저장 경로
        num_samples (int): 각 클래스당 표시할 샘플 수
    """
    # 최종 완전 연결 레이어의 가중치 행렬 가져오기
    fc_weights = model.fc.weight.data.cpu().numpy()
    
    # 특성 맵을 저장할 리스트
    features = []
    
    # 피처 맵을 가져오기 위한 훅 등록
    def hook_fn(module, input, output):
        features.append(output.cpu().detach())
    
    model.avgpool.register_forward_hook(hook_fn)
    
    # 각 클래스별 몇 개의 이미지에 대한 시각화
    fig, axes = plt.subplots(len(class_names), num_samples, 
                            figsize=(num_samples*3, len(class_names)*3))
    
    for class_idx, class_name in enumerate(class_names):
        # 검증 세트에서 이 클래스의 모든 이미지 가져오기
        class_samples = [(i, img_path) for i, (img_path, label) in enumerate(val_dataset.samples) 
                        if label == val_dataset.class_to_idx[class_name]]
        
        # 랜덤 샘플 선택
        if len(class_samples) >= num_samples:
            samples = np.random.choice(range(len(class_samples)), num_samples, replace=False)
            selected_samples = [class_samples[i] for i in samples]
        else:
            selected_samples = class_samples
        
        # 각 샘플에 대한 CAM 생성
        for j, (img_idx, img_path) in enumerate(selected_samples):
            if j < num_samples:
                # 이미지 로드 및 전처리
                img = Image.open(img_path).convert('RGB')
                img_tensor = val_dataset.transform(img).unsqueeze(0).to(device)
                
                # 순전파
                model.eval()
                with torch.no_grad():
                    output = model(img_tensor)
                    
                # 마지막 컨볼루션 레이어의 특성 맵 가져오기
                feature_maps = features[-1].squeeze(0)
                
                # 예측 클래스 가져오기
                _, pred_idx = torch.max(output, 1)
                pred_class = pred_idx.item()
                
                # 예측 클래스에 대한 가중치 가져오기
                class_weights = fc_weights[pred_class]
                
                # 클래스 활성화 맵 생성
                cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
                for k, w in enumerate(class_weights):
                    cam += w * feature_maps[k].numpy()
                
                # CAM 정규화
                cam = np.maximum(cam, 0)
                cam = cam / np.max(cam) if np.max(cam) > 0 else cam
                
                # 입력 이미지 크기에 맞게 CAM 리사이즈
                cam = np.uint8(255 * cam)
                cam = Image.fromarray(cam).resize((224, 224), Image.BICUBIC)
                cam = np.array(cam)
                
                # 입력 이미지를 numpy 배열로 변환
                img_np = np.array(img.resize((224, 224)))
                
                # CAM을 이미지에 오버레이
                heatmap = cm.jet(cam)[..., :3]
                cam_img = heatmap * 0.4 + img_np / 255.0 * 0.6
                
                # 이미지 표시
                if len(class_names) > 1:
                    ax = axes[class_idx, j]
                else:
                    ax = axes[j]
                
                ax.imshow(cam_img)
                ax.set_title(f"{class_name}\nPred: {class_names[pred_class]}")
                ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
