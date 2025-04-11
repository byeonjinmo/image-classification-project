# 딥러닝 이미지 분류 시스템

딥러닝 기반 이미지 분류 모델 학습 및 UI 시스템입니다.

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

## 사용 방법

1. 설치 및 실행:
   ```bash
   bash scripts/sh/setup_and_run.sh
   ```

2. UI 실행:
   ```bash
   python scripts/run_ui.py
   ```

3. 커맨드 라인으로 모델 학습:
   ```bash
   python scripts/run_classifier.py --data_dir data/flower_data --epochs 10 --batch_size 32
   ```

4. 데이터 다운로드 및 학습 자동화:
   ```bash
   python scripts/download_and_train.py --dataset flowers --epochs 10
   ```

## 주요 기능

- 데이터셋 로드 및 미리보기
- 다양한 ResNet 모델 선택 및 하이퍼파라미터 설정
- 실시간 학습 모니터링 및 신경망 시각화
- 혼동 행렬, ROC 커브, t-SNE, 클래스 활성화 맵 등 다양한 시각화
- 학습된 모델을 이용한 새 이미지 분류
