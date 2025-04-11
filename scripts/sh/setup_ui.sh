#!/bin/bash
# UI 모듈에 필요한 패키지 설치 스크립트

# 스크립트 위치 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

echo "프로젝트 루트 디렉토리: $PROJECT_ROOT"

# UI 관련 필요한 패키지 설치
echo "UI 관련 패키지 설치 중..."
pip install PyQt5>=5.15.0 pyqtgraph>=0.12.0 numpy>=1.19.0 torch>=1.7.0 torchvision>=0.8.0 matplotlib>=3.3.0 scikit-learn>=0.23.0 pandas>=1.1.0 tqdm>=4.47.0 Pillow>=7.2.0 seaborn>=0.11.0

echo "필요한 패키지 설치 완료"
echo "UI 프로그램을 실행하려면 다음 명령어를 실행하세요:"
echo "python $PROJECT_ROOT/scripts/run_ui.py"
