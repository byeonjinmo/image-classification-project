#!/usr/bin/env python3
"""
이미지 분류 모델의 테스트 스크립트

이 스크립트는 훈련된 모델을 사용하여 테스트 이미지에 대한 예측을 수행합니다.
훈련된 모델이 없는 경우 새 모델을 초기화하고 이를 간단히 훈련시킵니다.
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model.classifier import ImageClassifier

def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="이미지 분류 모델 테스트")
    parser.add_argument('--data_dir', type=str, default=os.path.join(project_root, 'flower_dataset'),
                        help='데이터셋 디렉토리 경로 (기본값: flower_dataset)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='훈련된 모델 파일 경로 (.pth). 지정하지 않으면 새 모델을 훈련합니다.')
    parser.add_argument('--test_image', type=str, default=None, 
                        help='테스트 할 이미지 파일 경로. 지정하지 않으면 검증 데이터셋에서 무작위로 선택합니다.')
    parser.add_argument('--model_type', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='모델 타입 (기본값: resnet18)')
    parser.add_argument('--num_epochs', type=int, default=2, 
                        help='모델 훈련 에폭 수 (기본값: 2)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='배치 크기 (기본값: 32)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'test_results'),
                        help='결과 저장 디렉토리 (기본값: test_results)')
    
    return parser.parse_args()

def train_model(classifier, args):
    """모델 학습을 위한 함수"""
    print(f"데이터셋 로드 중: {args.data_dir}")
    # 데이터 로드
    classifier.load_data()
    
    print(f"모델 초기화 중: {args.model_type}")
    # 모델 초기화
    classifier.create_model()
    
    print(f"{args.num_epochs}에폭 동안 모델 훈련 중...")
    # 모델 훈련
    classifier.train()
    
    # 모델 저장
    model_path = os.path.join(args.output_dir, 'test_model.pth')
    classifier.save_model(model_path)
    print(f"훈련된 모델 저장됨: {model_path}")
    
    return classifier

def test_model(classifier, test_image_path=None):
    """
    모델 테스트를 위한 함수
    
    Args:
        classifier: 초기화된 ImageClassifier 객체
        test_image_path: 테스트 이미지 경로 (None인 경우 무작위 이미지 사용)
    """
    # 테스트 이미지가 지정되지 않은 경우 검증 데이터셋에서 무작위로 선택
    if test_image_path is None:
        # 검증 데이터셋에서 무작위 이미지 선택
        if hasattr(classifier, 'val_dataset') and classifier.val_dataset:
            idx = np.random.randint(0, len(classifier.val_dataset))
            test_image_path, true_label = classifier.val_dataset.samples[idx]
            true_class = classifier.class_names[true_label]
            print(f"무작위 이미지 선택됨: {test_image_path}")
            print(f"참 클래스: {true_class}")
        else:
            print("오류: 테스트 이미지가 지정되지 않았고 검증 데이터셋이 없습니다.")
            return
    
    # 이미지 예측
    try:
        result = classifier.predict(test_image_path)
        
        # 결과 출력
        print("\n예측 결과:")
        print(f"예측 클래스: {result['class']}")
        print(f"신뢰도: {result['probability']*100:.2f}%")
        
        # 모든 클래스별 확률 출력
        print("\n클래스별 신뢰도:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob*100:.2f}%")
        
        # 이미지와 예측 결과 시각화
        img = Image.open(test_image_path)
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("테스트 이미지")
        plt.axis('off')
        
        # 바 차트로 클래스별 확률 표시
        plt.subplot(1, 2, 2)
        classes = list(result['all_probabilities'].keys())
        probs = [result['all_probabilities'][c] for c in classes]
        colors = ['blue' if c != result['class'] else 'red' for c in classes]
        y_pos = np.arange(len(classes))
        
        plt.barh(y_pos, probs, color=colors)
        plt.yticks(y_pos, classes)
        plt.xlabel('신뢰도')
        plt.title('클래스별 예측 신뢰도')
        
        plt.tight_layout()
        
        # 결과 저장
        output_path = os.path.join(classifier.output_dir, 'prediction_result.png')
        plt.savefig(output_path)
        print(f"\n결과 저장됨: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")

def main():
    """메인 함수"""
    args = parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 분류기 객체 생성
    classifier = ImageClassifier(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        model_type=args.model_type
    )
    
    # 모델 파일이 지정되었으면 로드, 그렇지 않으면 새 모델 훈련
    if args.model_path and os.path.exists(args.model_path):
        print(f"모델 로드 중: {args.model_path}")
        # 데이터 로드
        classifier.load_data()
        # 모델 로드
        classifier.load_model(args.model_path)
    else:
        print("훈련된 모델이 지정되지 않았습니다. 새 모델을 훈련합니다.")
        classifier = train_model(classifier, args)
    
    # 모델 테스트
    test_model(classifier, args.test_image)

if __name__ == "__main__":
    main()
