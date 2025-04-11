#!/bin/bash
# 패키지 설치 및 실행을 위한 스크립트

# 스크립트 위치 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

echo "프로젝트 루트 디렉토리: $PROJECT_ROOT"

# 필요한 패키지 설치
echo "필요한 패키지 설치 중..."
pip install -r $PROJECT_ROOT/requirements.txt

# 디렉토리 구조 확인
echo "디렉토리 구조 확인 중..."
ls -la $PROJECT_ROOT

# 애플리케이션 실행
echo "이미지 분류기 UI 실행 중..."
python $PROJECT_ROOT/scripts/run_app.py
