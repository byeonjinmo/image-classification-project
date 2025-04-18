# 딥러닝 이미지 분류 시스템

ResNet 아키텍처 기반의 딥러닝 이미지 분류 모델 학습 및 평가, 직관적인 사용자 인터페이스를 제공하는 종합 시스템입니다.

## 프로젝트 개선사항

최근 개선된 부분:
- UI 버그 수정 및 기능 향상 (누락된 메서드 추가)
- 데이터 분석 기능 강화 및 시각화 개선
- 체크포인트 관리 시스템 추가 
- 테스트 및 훈련 스크립트 추가
- 프로젝트 디렉토리 구조 정리 및 코드 경로 수정

## 소개

이 프로젝트는 컴퓨터 비전에서 가장 기본적이면서도 중요한 작업인 이미지 분류를 위한 딥러닝 파이프라인을 구현합니다. ResNet(Residual Network) 모델을 활용하여 다양한 이미지 데이터셋에 대한 분류 작업을 수행할 수 있으며, 특히 꽃 이미지 데이터셋을 기본 예제로 제공합니다.

### 주요 특징

- **다양한 ResNet 모델 지원**: ResNet18, ResNet34, ResNet50 등 다양한 깊이의 모델 선택 가능
- **데이터 관리**: 데이터셋 다운로드, 전처리, 증강(augmentation) 기능 내장
- **시각화 도구**: 학습 과정, 성능 평가, 모델 해석을 위한 다양한 시각화 도구 제공
- **직관적인 UI**: PyQt5 기반의 사용자 친화적 인터페이스
- **유연한 실행 방식**: 명령줄 또는 UI를 통한 실행 지원
- **향상된 모델 관리**: 체크포인트 저장/불러오기 및 관리 시스템

## 설치 및 실행 방법

### 필요 조건

- Python 3.6 이상
- PyTorch 1.7 이상
- CUDA 지원 GPU (권장)

### 설치 단계

1. 저장소 클론:
   ```bash
   git clone https://github.com/username/image-classification-project.git
   cd image-classification-project
   ```

2. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

### 실행 방법

#### 1. UI 실행하기

```bash
python scripts/run_ui.py
```

또는 업데이트된 런처를 사용할 수 있습니다:

```bash
python scripts/updated_launcher.py
```

#### 2. 명령줄로 모델 학습하기

```bash
python scripts/run_training.py --data_dir flower_dataset --num_epochs 10 --batch_size 32 --model_type resnet18
```

#### 3. 학습된 모델로 이미지 예측하기

```bash
python scripts/test_model.py --data_dir flower_dataset --model_path results/final_model.pth --test_image path/to/image.jpg
```

## 프로젝트 구성요소

### UI 컴포넌트

- **데이터 및 모델 설정 탭**: 데이터셋 로드, 모델 유형 선택, 하이퍼파라미터 설정
- **모델 학습 탭**: 학습 진행 상황 모니터링, 실시간 손실/정확도 그래프, 뉴런 활성화 시각화
- **학습 결과 탭**: 혼동 행렬, ROC 커브, t-SNE 시각화, 클래스 활성화 맵 등 
- **이미지 예측 탭**: 개별 이미지 예측 및 클래스별 신뢰도 확인
- **체크포인트 관리 탭**: 학습 체크포인트 저장, 불러오기 및 관리
- **데이터 분석 탭**: 데이터셋 분석 및 클래스 분포, 이미지 크기 분포 등 시각화

## 프로젝트 구조

```
jin/
├── data/                   # 데이터 관련 모듈
│   ├── __init__.py         # 패키지 초기화
│   ├── create_flower_dataset.py  # 꽃 데이터셋 생성
│   ├── dataset.py          # 데이터셋 클래스 및 유틸리티
│   └── download_dataset.py # 데이터셋 다운로드 스크립트
│
├── model/                  # 모델 관련 모듈
│   ├── __init__.py         # 패키지 초기화
│   ├── classifier.py       # 이미지 분류 모델 클래스
│   ├── image_utils.py      # 이미지 유틸리티 함수
│   └── utils/              # 모델 유틸리티
│
├── scripts/                # 실행 스크립트
│   ├── run_app.py          # 애플리케이션 실행 스크립트
│   ├── run_ui.py           # UI 실행 스크립트
│   ├── run_training.py     # 모델 학습 스크립트
│   ├── test_model.py       # 모델 테스트 스크립트
│   ├── updated_launcher.py # 업데이트된 UI 런처
│   └── improved_patch.py   # UI 패치 스크립트
│
├── ui/                     # UI 관련 모듈
│   ├── __init__.py         # 패키지 초기화
│   ├── classifier_ui.py    # 이미지 분류기 UI 클래스
│   ├── components/         # UI 컴포넌트
│   │   ├── __init__.py     # 패키지 초기화
│   │   ├── checkpoint_manager.py # 체크포인트 관리자
│   │   ├── data_analyzer.py # 데이터 분석기
│   │   └── widgets/        # UI 위젯
│   └── utils/              # UI 유틸리티
│
├── flower_dataset/         # 꽃 이미지 데이터셋 (예시)
│
├── results/                # 학습 결과 저장 디렉토리
│
└── requirements.txt        # 필수 패키지 목록
```

## 사용법 튜토리얼

### UI를 통한 사용

1. **데이터 및 모델 설정**:
   - '데이터 디렉토리' 찾아보기 버튼을 클릭하여 이미지 데이터셋 폴더 선택
   - '데이터 로드 및 미리보기' 버튼을 클릭하여 데이터셋 정보 확인
   - 모델 유형 및 하이퍼파라미터 설정
   - '모델 초기화' 버튼 클릭

2. **모델 학습**:
   - '학습 시작' 버튼을 클릭하여 훈련 시작
   - 실시간 손실/정확도 그래프 및 진행 상황 모니터링
   - 필요시 '학습 중단' 버튼으로 중단 가능

3. **결과 확인**:
   - 학습 완료 후 '학습 결과' 탭으로 자동 이동
   - 혼동 행렬, ROC 커브 등 다양한 성능 지표 확인
   - '모델 평가' 버튼으로 추가 평가 진행 가능
   - '결과 저장' 버튼으로 모델 및 결과 저장

4. **이미지 예측**:
   - '모델 파일' 선택 및 테스트할 이미지 선택
   - '예측 실행' 버튼 클릭
   - 예측 클래스 및 클래스별 신뢰도 확인

### 명령줄을 통한 사용

1. **모델 학습**: 
   ```bash
   python scripts/run_training.py --data_dir flower_dataset --output_dir results --model_type resnet18 --num_epochs 10
   ```

2. **학습 재개**: 
   ```bash
   python scripts/run_training.py --data_dir flower_dataset --checkpoint results/checkpoint_epoch_5.pth
   ```

3. **테스트 이미지 예측**: 
   ```bash
   python scripts/test_model.py --model_path results/final_model.pth --test_image flower.jpg
   ```

## 모델 성능

꽃 데이터셋에 대한 ResNet 모델의 대표적인 성능:

| 모델      | 정확도   | 훈련 시간 | 파라미터 수 |
|-----------|---------|----------|------------|
| ResNet18  | ~92%    | 빠름     | 11M        |
| ResNet34  | ~94%    | 중간     | 21M        |
| ResNet50  | ~95%    | 느림     | 25M        |

## 라이선스

MIT License
