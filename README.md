# GitHub Codespaces에서 이미지 분류기 실행하기

이 프로젝트는 GitHub Codespaces에서 쉽게 실행할 수 있도록 설정되었습니다.

## Codespaces에서 실행하는 방법

1. GitHub 저장소에서 "Code" 버튼을 클릭하고 "Codespaces" 탭을 선택합니다.
2. "Create codespace on main" 버튼을 클릭해 새 Codespace를 생성합니다.
3. Codespace가 시작되면 터미널에서 다음 명령어를 실행합니다:
   ```bash
   bash setup.sh
   ```
4. 설정 스크립트가 완료되면 다음 명령어로 데이터셋을 다운로드하고 모델을 학습합니다:
   ```bash
   python download_and_train.py --epochs 5 --batch_size 16
   ```

## 유의사항

- 학습에는 GPU가 없어도 되지만, 있으면 효율적으로 학습할 수 있습니다.
- Codespaces의 무료 할당량은 매월 60 코어 시간이며, 사용하지 않을 때는 Codespace를 중지하여 시간을 절약하세요.
- 학습된 모델은 `flower_results` 폴더에 저장됩니다. 중요한 결과는 리포지토리에 커밋하여 보존하세요.

# Automated Image Classification Framework

A comprehensive framework for automating image classification using ResNet backbone models with professional visualization of the entire machine learning pipeline.

## Features

- **Automated Data Processing**: Loads and preprocesses image data from folder structure
- **Visualization**: Professional visualizations for each step of the ML process
- **ResNet Backbone**: Uses pretrained ResNet models (18, 34, or 50) for transfer learning
- **Comprehensive Evaluation**: Generates confusion matrices, t-SNE visualizations, CAM heatmaps
- **Easy-to-use CLI**: Simple command-line interface for training, evaluation, and prediction

## Requirements

- Python 3.6+
- PyTorch and torchvision
- scikit-learn
- matplotlib
- seaborn
- pandas
- tqdm
- PIL (Pillow)

Install requirements with:

```
pip install torch torchvision scikit-learn matplotlib seaborn pandas tqdm pillow
```

## Data Structure

Your images should be organized in a folder structure where each class has its own folder:

```
data_directory/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Usage

### Training a Model

```bash
python run_classifier.py --data_dir /path/to/image/data --output_dir results --epochs 20 --model resnet34
```

### Evaluating a Trained Model

```bash
python run_classifier.py --data_dir /path/to/image/data --mode evaluate --model_path results/final_model.pth
```

### Making Predictions

```bash
python run_classifier.py --data_dir /path/to/image/data --mode predict --predict_image /path/to/single/image.jpg
```

### Resuming Training from a Checkpoint

```bash
python run_classifier.py --mode resume --data_dir /path/to/image/data --checkpoint_path results/checkpoint_epoch_3.pth --epochs 10
```

이 기능은 학습 중단 후 체크포인트에서 이어서 학습할 수 있게 해줍니다. 체크포인트는 각 에폭 후에 자동으로 저장됩니다.

## Output Visualizations

The framework generates the following visualizations:

1. **Sample Images**: Displays sample images from each class
2. **Training Progress**: Plots of training/validation loss and accuracy
3. **Confusion Matrix**: Visualization of model predictions across classes
4. **ROC Curves and AUC**: Receiver Operating Characteristic curves with AUC scores for each class
5. **t-SNE Visualization**: 2D projection of high-dimensional feature space
6. **Class Activation Maps**: Heatmaps showing important regions for classification

All visualizations are saved to the specified output directory.

## Advanced Options

```
--batch_size INT       Batch size for training (default: 32)
--lr FLOAT             Learning rate (default: 0.001)
--val_split FLOAT      Validation split ratio (default: 0.2)
--model STRING         ResNet variant: resnet18, resnet34, resnet50 (default: resnet18)
--mode STRING          Operation mode: train, evaluate, predict, resume
--checkpoint_path PATH Path to checkpoint file for resuming training (required for resume mode)
```

## 데이터셋 다운로드 및 모델 학습 자동화

이 프로젝트에는 데이터셋 다운로드와 모델 학습을 자동화하는 스크립트가 추가되었습니다:

### 꽃 데이터셋 자동 다운로드

```bash
python create_flower_dataset.py --output_dir flower_dataset
```

이 명령어는 TensorFlow Flowers 데이터셋을 다운로드하고 적절한 폴더 구조로 구성합니다.

### 데이터셋 다운로드 및 모델 학습 한번에 실행

```bash
python download_and_train.py --epochs 5 --batch_size 16
```

이 스크립트는 다음 작업을 순차적으로 수행합니다:
1. 꽃 데이터셋 다운로드
2. 이미지 분류기 모델 학습
3. 모델 평가 및 시각화

데이터셋 다운로드를 건너뛰려면 `--skip_download` 옵션을 사용하세요.

## Example Workflow

1. 자동으로 데이터셋 다운로드 및 모델 학습:
   ```
   python download_and_train.py
   ```

2. 또는 직접 이미지를 클래스별 폴더에 구성한 후 학습:
   ```
   python run_classifier.py --data_dir my_images/ --epochs 15
   ```

3. 학습 결과 및 시각화를 확인 (flower_results 또는 results 폴더)

4. 학습이 중단된 경우 체크포인트에서 이어서 학습:
   ```
   python run_classifier.py --mode resume --data_dir my_images/ --checkpoint_path flower_results/checkpoint_epoch_3.pth --epochs 15
   ```

5. 새 이미지로 예측:
   ```
   python run_classifier.py --mode predict --predict_image new_image.jpg --model_path flower_results/final_model.pth
   ```

## License

MIT License