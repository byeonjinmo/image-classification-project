#!/usr/bin/env python3
"""
데이터셋 다운로드, 준비 및 모델 학습을 위한 통합 스크립트
"""
import os
import sys
import argparse
import subprocess

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def main():
    """통합 스크립트 실행"""
    parser = argparse.ArgumentParser(description="데이터셋 다운로드 및 모델 학습")
    parser.add_argument("--dataset", choices=["flowers", "cifar", "food"], default="flowers",
                      help="다운로드할 데이터셋 (기본값: flowers)")
    parser.add_argument("--output_dir", default="datasets",
                      help="데이터셋 저장 디렉토리 (기본값: datasets)")
    parser.add_argument("--num_classes", type=int, default=5,
                      help="포함할 클래스 수 (기본값: 5)")
    parser.add_argument("--images_per_class", type=int, default=20,
                      help="클래스당 이미지 수 (기본값: 20)")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="학습 배치 크기 (기본값: 32)")
    parser.add_argument("--epochs", type=int, default=10,
                      help="학습 에폭 수 (기본값: 10)")
    parser.add_argument("--model", choices=["resnet18", "resnet34", "resnet50"],
                      default="resnet18", help="사용할 모델 (기본값: resnet18)")
    parser.add_argument("--skip_download", action="store_true",
                      help="데이터셋 다운로드 단계 건너뛰기")
    
    args = parser.parse_args()
    
    # 1. 데이터셋 다운로드 및 준비
    if not args.skip_download:
        print("\n=== 데이터셋 다운로드 및 준비 ===")
        download_cmd = [
            "python", os.path.join(project_root, "data", "download_dataset.py"),
            "--dataset", args.dataset,
            "--output_dir", args.output_dir,
            "--num_classes", str(args.num_classes),
            "--images_per_class", str(args.images_per_class)
        ]
        
        try:
            subprocess.run(download_cmd, check=True)
        except subprocess.CalledProcessError:
            print("데이터셋 다운로드 중 오류가 발생했습니다.")
            return
    
    # 2. 모델 학습
    print("\n=== 모델 학습 시작 ===")
    # 다운로드된 데이터셋의 경로 결정
    data_dir = os.path.join(args.output_dir, f"{args.dataset}_data")
    
    train_cmd = [
        "python", os.path.join(project_root, "scripts", "run_classifier.py"),
        "--data_dir", data_dir,
        "--output_dir", f"{args.dataset}_results",
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--model", args.model,
        "--mode", "train"
    ]
    
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError:
        print("모델 학습 중 오류가 발생했습니다.")
        return
    
    print("\n=== 처리 완료 ===")
    print(f"학습된 모델은 '{args.dataset}_results' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
