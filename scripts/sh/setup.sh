#!/bin/bash
# 프로젝트 기본 설정 스크립트

# 스크립트 위치 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

echo "프로젝트 루트 디렉토리: $PROJECT_ROOT"

# 실행 권한 설정
chmod +x $PROJECT_ROOT/scripts/download_and_train.py
chmod +x $PROJECT_ROOT/data/create_flower_dataset.py
chmod +x $PROJECT_ROOT/scripts/run_classifier.py
chmod +x $PROJECT_ROOT/scripts/run_ui.py
chmod +x $PROJECT_ROOT/scripts/run_app.py

# 환경 설정 메시지 출력
echo "==========================================="
echo "이미지 분류 프로젝트 환경이 설정되었습니다!"
echo "==========================================="
echo "사용 가능한 명령어:"
echo "1. 데이터셋 다운로드:"
echo "   python $PROJECT_ROOT/data/download_dataset.py"
echo ""
echo "2. 데이터셋 다운로드 및 모델 학습:"
echo "   python $PROJECT_ROOT/scripts/download_and_train.py --epochs 5 --batch_size 16"
echo ""
echo "3. UI 실행:"
echo "   python $PROJECT_ROOT/scripts/run_ui.py"
echo ""
echo "4. 학습된 모델로 이미지 예측:"
echo "   python $PROJECT_ROOT/scripts/run_classifier.py --mode predict --predict_image <이미지_경로>"
echo "==========================================="
