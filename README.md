# 딥러닝 이미지 분류 시스템

ResNet 아키텍처 기반의 딥러닝 이미지 분류 모델 학습 및 평가, 직관적인 사용자 인터페이스를 제공하는 종합 시스템입니다.

## 소개

이 프로젝트는 컴퓨터 비전에서 가장 기본적이면서도 중요한 작업인 이미지 분류를 위한 딥러닝 파이프라인을 구현합니다. ResNet(Residual Network) 모델을 활용하여 다양한 이미지 데이터셋에 대한 분류 작업을 수행할 수 있으며, 특히 꽃 이미지 데이터셋을 기본 예제로 제공합니다.

### 주요 특징

- **다양한 ResNet 모델 지원**: ResNet18, ResNet34, ResNet50 등 다양한 깊이의 모델 선택 가능
- **데이터 관리**: 데이터셋 다운로드, 전처리, 증강(augmentation) 기능 내장
- **시각화 도구**: 학습 과정, 성능 평가, 모델 해석을 위한 다양한 시각화 도구 제공
- **직관적인 UI**: PyQt5 기반의 사용자 친화적 인터페이스
- **유연한 실행 방식**: 명령줄 또는 UI를 통한 실행 지원

## 기술적 개요

### 이미지 분류 구현

본 시스템은 다음과 같은 이미지 분류 파이프라인을 구현합니다:

1. **데이터 준비**
   - 데이터셋 다운로드 및 구조화
   - 이미지 전처리: 크기 조정, 정규화, 증강
   - 학습/검증 데이터 분할

2. **모델 아키텍처**
   - ResNet: 이미지 분류를 위한 잔차 학습(Residual Learning) 네트워크
   - 전이 학습(Transfer Learning): ImageNet 사전 학습 가중치 활용
   - 다양한 모델 깊이 지원: 18, 34, 50 레이어 버전

3. **학습 과정**
   - 교차 엔트로피 손실 함수
   - Adam 최적화 알고리즘
   - 학습률 스케줄러 (ReduceLROnPlateau)
   - 배치 정규화 및 드롭아웃을 통한 정규화

4. **평가 및 분석**
   - 정확도, 정밀도, 재현율, F1 점수 등 다양한 평가 지표
   - 혼동 행렬 및 ROC 커브 분석
   - t-SNE를 통한 특성 시각화
   - 클래스 활성화 맵(CAM)을 통한 모델 해석

### 시각화 기능

- **학습 모니터링**: 학습/검증 손실, 정확도의 실시간 그래프 시각화
- **성능 평가**: 혼동 행렬, ROC 커브, 정밀도-재현율 곡선
- **모델 해석**: 클래스 활성화 맵으로 모델이 이미지의 어떤 부분에 주목하는지 시각화
- **특성 시각화**: t-SNE를 활용한 고차원 특성의 2D 시각화
- **신경망 활성화**: 네트워크 레이어 활성화 패턴 시각화

## 프로젝트 구조

```
jin/
├── data/                   # 데이터 관련 모듈
│   ├── __init__.py         # 패키지 초기화
│   ├── create_flower_dataset.py  # 꽃 데이터셋 생성
│   ├── dataset.py          # 데이터셋 클래스 및 유틸리티
│   ├── download_dataset.py # 데이터셋 다운로드 스크립트
│   ├── flower_data/        # 꽃 데이터셋 저장 디렉토리
│   └── samples/            # 샘플 데이터
│
├── model/                  # 모델 관련 모듈
│   ├── __init__.py         # 패키지 초기화
│   ├── classifier.py       # 이미지 분류 모델 클래스
│   └── utils/              # 모델 유틸리티
│       ├── __init__.py     # 패키지 초기화
│       ├── metrics.py      # 모델 평가 메트릭 유틸리티
│       └── visualization.py # 모델 시각화 유틸리티
│
├── scripts/                # 실행 스크립트
│   ├── download_and_train.py # 데이터 다운로드 및 학습 통합 스크립트
│   ├── run_app.py          # 애플리케이션 실행 스크립트
│   ├── run_classifier.py   # 분류기 실행 스크립트
│   ├── run_ui.py           # UI 실행 스크립트
│   └── sh/                 # 셸 스크립트
│       ├── setup.sh        # 기본 설치 스크립트
│       ├── setup_and_run.sh # 설치 및 실행 스크립트
│       └── setup_ui.sh     # UI 관련 설치 스크립트
│
├── ui/                     # UI 관련 모듈
│   ├── __init__.py         # 패키지 초기화
│   ├── classifier_ui.py    # 이미지 분류기 UI 클래스
│   ├── components/         # UI 컴포넌트
│   │   ├── __init__.py     # 패키지 초기화
│   │   └── widgets/        # UI 위젯
│   │       ├── __init__.py  # 패키지 초기화
│   │       └── visualization.py # 시각화 위젯
│   ├── main_window.py      # 메인 UI 창
│   ├── main_window_init.py # 메인 창 초기화
│   ├── main_window_methods.py # 메인 창 메서드
│   └── utils/              # UI 유틸리티
│       ├── __init__.py     # 패키지 초기화
│       └── file_utils.py   # 파일 처리 유틸리티
│
└── requirements.txt        # 필수 패키지 목록
```

## 설치 방법

### 필요 조건

- Python 3.6 이상
- PyTorch 1.7 이상
- CUDA 지원 GPU (권장)

### 설치 단계

1. 저장소 클론:
   ```bash
   git clone https://github.com/byeonjinmo/image-classification-project.git
   cd image-classification-project
   ```

2. 필요한 패키지 설치:
   ```bash
   bash scripts/sh/setup.sh
   ```

## 사용 방법

### UI를 통한 사용

1. UI 실행:
   ```bash
   python scripts/run_ui.py
   ```

2. UI 내에서:
   - '데이터 및 모델 설정' 탭에서 데이터셋 선택
   - 모델 유형 및 하이퍼파라미터 설정
   - '모델 학습' 탭에서 학습 시작
   - '학습 결과' 탭에서 다양한 성능 지표 및 시각화 확인
   - '이미지 예측' 탭에서 새 이미지 분류 테스트

### 명령줄을 통한 사용

1. 데이터셋 다운로드:
   ```bash
   python data/download_dataset.py --dataset flowers --output_dir datasets
   ```

2. 모델 학습:
   ```bash
   python scripts/run_classifier.py --data_dir data/flower_data --epochs 10 --batch_size 32 --model resnet18
   ```

3. 이미지 예측:
   ```bash
   python scripts/run_classifier.py --mode predict --predict_image path/to/image.jpg
   ```

4. 다운로드 및 학습 자동화:
   ```bash
   python scripts/download_and_train.py --dataset flowers --epochs 10
   ```

## 모델 성능

꽃 데이터셋에 대한 ResNet 모델의 대표적인 성능:

| 모델      | 정확도   | 훈련 시간 | 파라미터 수 |
|-----------|---------|----------|------------|
| ResNet18  | ~92%    | 빠름     | 11M        |
| ResNet34  | ~94%    | 중간     | 21M        |
| ResNet50  | ~95%    | 느림     | 25M        |

## 시각화 예시

- **혼동 행렬**: 각 클래스의 예측 정확도와 오류 패턴을 시각화
- **ROC 커브**: 이진 분류기의 성능을 다양한 임계값에서 평가
- **t-SNE 시각화**: 고차원 특성을 2D 공간에 매핑하여 클래스 간 관계 시각화
- **클래스 활성화 맵**: 모델이 이미지에서 주목하는 영역을 히트맵으로 시각화

## 확장 및 커스터마이징

- **새 데이터셋 추가**: `data/dataset.py` 모듈 확장
- **다른 모델 아키텍처 지원**: `model/classifier.py`에 다른 모델 통합
- **추가 시각화**: `model/utils/visualization.py`에 새 시각화 기법 추가
- **UI 사용자 정의**: `ui/` 디렉토리의 구성 요소 수정

## 라이선스

MIT License
