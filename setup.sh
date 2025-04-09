#!/bin/bash

# 실행 권한 설정
chmod +x download_and_train.py
chmod +x create_flower_dataset.py
chmod +x run_classifier.py

# 환경 설정 메시지 출력
echo "==========================================="
echo "이미지 분류 프로젝트 환경이 설정되었습니다!"
echo "==========================================="
echo "사용 가능한 명령어:"
echo "1. 데이터셋 다운로드:"
echo "   python create_flower_dataset.py"
echo ""
echo "2. 데이터셋 다운로드 및 모델 학습:"
echo "   python download_and_train.py --epochs 5 --batch_size 16"
echo ""
echo "3. 학습된 모델로 이미지 예측:"
echo "   python run_classifier.py --mode predict --predict_image <이미지_경로> --model_path flower_results/final_model.pth"
echo "==========================================="
