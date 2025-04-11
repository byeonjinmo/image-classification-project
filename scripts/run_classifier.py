#!/usr/bin/env python3
"""
커맨드 라인에서 이미지 분류 모델 학습 및 평가 실행
"""
import sys
import os
import argparse

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model.classifier import ImageClassifier

def main():
    """명령행 인자 파싱 및 모델 학습/평가 실행"""
    parser = argparse.ArgumentParser(description="이미지 분류 모델 학습 및 평가")
    parser.add_argument("--data_dir", required=True, help="이미지 데이터가 있는 디렉토리")
    parser.add_argument("--output_dir", default="results", help="결과를 저장할 디렉토리")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=10, help="학습 에폭 수")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    parser.add_argument("--model", default="resnet18", 
                      choices=["resnet18", "resnet34", "resnet50"],
                      help="사용할 ResNet 모델 변종")
    parser.add_argument("--val_split", type=float, default=0.2, 
                      help="검증 데이터 분할 비율 (0-1)")
    parser.add_argument("--mode", default="train", 
                      choices=["train", "evaluate", "predict"],
                      help="실행 모드")
    parser.add_argument("--predict_image", help="예측할 이미지 경로 (--mode predict와 함께 사용)")
    
    args = parser.parse_args()
    
    # 분류기 생성
    classifier = ImageClassifier(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_type=args.model
    )
    
    if args.mode == "train":
        # 데이터 로드, 모델 생성 및 학습
        classifier.load_data(val_split=args.val_split)
        classifier.create_model()
        classifier.train()
        classifier.evaluate()
        classifier.save_model()
        
    elif args.mode == "evaluate":
        # 데이터 로드 및 사전 학습된 모델 평가
        classifier.load_data(val_split=args.val_split)
        classifier.load_model()
        classifier.evaluate()
        
    elif args.mode == "predict":
        if not args.predict_image:
            parser.error("--predict_image는 --mode predict와 함께 사용해야 합니다")
        
        # 사전 학습된 모델 로드 및 예측 실행
        classifier.load_model()
        result = classifier.predict(args.predict_image)
        print(f"예측: {result['class']} (확률: {result['probability']:.2%})")

if __name__ == "__main__":
    main()
