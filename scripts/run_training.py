#!/usr/bin/env python3
"""
딥러닝 이미지 분류 모델 학습 실행 스크립트

이 스크립트는 데이터 준비, 모델 학습, 평가 및 시각화를 자동화합니다.
"""
import os
import sys
import argparse
import time

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model.classifier import ImageClassifier

def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="딥러닝 이미지 분류 모델 학습")
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='데이터셋 디렉토리 경로 (필수)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'results'),
                        help='결과 저장 디렉토리 (기본값: results)')
    parser.add_argument('--model_type', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='모델 타입 (기본값: resnet18)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='배치 크기 (기본값: 32)')
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help='학습 에폭 수 (기본값: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='학습률 (기본값: 0.001)')
    parser.add_argument('--val_split', type=float, default=0.2, 
                        help='검증 데이터 비율 (기본값: 0.2)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='체크포인트 파일 경로 (.pth). 지정하면 체크포인트에서 학습 재개')
    parser.add_argument('--no_eval', action='store_true',
                        help='학습 후 모델 평가를 건너뜁니다.')
    
    return parser.parse_args()

def print_section(title):
    """섹션 제목 출력"""
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")

def main():
    """메인 함수"""
    args = parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print_section("설정 정보")
    print(f"데이터셋 디렉토리: {args.data_dir}")
    print(f"결과 저장 디렉토리: {args.output_dir}")
    print(f"모델 타입: {args.model_type}")
    print(f"배치 크기: {args.batch_size}")
    print(f"학습 에폭 수: {args.num_epochs}")
    print(f"학습률: {args.learning_rate}")
    print(f"검증 데이터 비율: {args.val_split}")
    print(f"체크포인트 파일: {args.checkpoint if args.checkpoint else '없음'}")
    
    # 분류기 객체 생성
    classifier = ImageClassifier(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        model_type=args.model_type
    )
    
    print_section("데이터 로드")
    # 데이터 로드
    train_loader, val_loader = classifier.load_data(val_split=args.val_split)
    print(f"학습 데이터: {len(classifier.train_dataset)}개 이미지, {len(classifier.train_loader)} 배치")
    print(f"검증 데이터: {len(classifier.val_dataset)}개 이미지, {len(classifier.val_loader)} 배치")
    print(f"클래스 목록: {classifier.class_names}")
    
    print_section("모델 초기화")
    # 모델 초기화
    model = classifier.create_model()
    
    # 체크포인트에서 학습 재개
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"체크포인트 로드 중: {args.checkpoint}")
        import torch
        checkpoint = torch.load(args.checkpoint, map_location=classifier.device)
        
        if 'model_state_dict' in checkpoint:
            classifier.model.load_state_dict(checkpoint['model_state_dict'])
            print("모델 상태 로드 완료")
            
        if 'optimizer_state_dict' in checkpoint:
            classifier.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("옵티마이저 상태 로드 완료")
            
        if 'scheduler_state_dict' in checkpoint and hasattr(classifier, 'scheduler'):
            classifier.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("스케줄러 상태 로드 완료")
            
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"에폭 {start_epoch}부터 학습 재개")
            
        if 'history' in checkpoint:
            classifier.history = checkpoint['history']
            print("학습 히스토리 로드 완료")
    
    print_section("모델 학습")
    # 학습 시작 시간 기록
    start_time = time.time()
    
    # 모델 학습
    classifier.train(start_epoch=start_epoch)
    
    # 학습 종료 시간 계산
    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print(f"학습 완료! 소요 시간: {hours}시간 {minutes}분 {seconds}초")
    
    # 모델 저장
    model_path = os.path.join(args.output_dir, 'final_model.pth')
    classifier.save_model(model_path)
    print(f"최종 모델 저장됨: {model_path}")
    
    # 모델 평가
    if not args.no_eval:
        print_section("모델 평가")
        results = classifier.evaluate()
        print("평가 완료!")
        print(f"결과 이미지 저장 디렉토리: {args.output_dir}")
    
    print_section("처리 완료")
    print(f"모든 결과는 {args.output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
