#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys

def run_command(command):
    """Run a shell command and print output"""
    print(f"실행 명령어: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="데이터셋 다운로드 및 이미지 분류기 실행")
    parser.add_argument("--output_dir", default="flower_dataset",
                        help="데이터셋 저장 디렉토리 (기본값: flower_dataset)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="학습 에포크 수 (기본값: 5)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="배치 크기 (기본값: 16)")
    parser.add_argument("--skip_download", action="store_true",
                        help="데이터셋 다운로드 과정 건너뛰기")
    
    args = parser.parse_args()
    
    # Step 1: Download dataset (if not skipped)
    if not args.skip_download:
        print("=== 꽃 데이터셋 다운로드 시작 ===")
        download_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_flower_dataset.py")
        
        download_cmd = [sys.executable, download_script, "--output_dir", args.output_dir]
        if run_command(download_cmd) != 0:
            print("데이터셋 다운로드 중 오류가 발생했습니다.")
            return
    
    # Step 2: Train the model
    print("\n=== 이미지 분류기 학습 시작 ===")
    dataset_dir = os.path.join(args.output_dir, "flower_photos")
    
    if not os.path.exists(dataset_dir):
        print(f"오류: 데이터셋 디렉토리 '{dataset_dir}'가 존재하지 않습니다.")
        return
    
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_classifier.py")
    
    train_cmd = [
        sys.executable, train_script,
        "--data_dir", dataset_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--output_dir", "flower_results"
    ]
    
    if run_command(train_cmd) != 0:
        print("모델 학습 중 오류가 발생했습니다.")
        return
    
    print("\n=== 작업 완료 ===")
    print("모델 학습 결과는 'flower_results' 디렉토리에 저장되었습니다.")
    print("추가로 모델을 사용하여 새 이미지를 예측하려면 다음 명령어를 실행하세요:")
    print(f"{sys.executable} {train_script} --mode predict --predict_image <이미지_경로> --model_path flower_results/final_model.pth")

if __name__ == "__main__":
    main()
