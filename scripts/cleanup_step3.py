#!/usr/bin/env python3
"""
프로젝트 정리 스크립트 (3단계)

불필요한 백업 파일들(.bak)을 정리합니다:
- model/image_classifier.py.bak
- ui/classifier_ui.py.bak
- ui/components/neuron_visualization.py.bak
- ui/components/training_thread.py.bak
"""
import os
import sys
import shutil

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def cleanup_backup_files():
    """백업 파일 정리 함수"""
    # 백업 디렉토리 생성
    backup_dir = os.path.join(project_root, '_backup')
    os.makedirs(backup_dir, exist_ok=True)
    
    # 백업 파일 목록
    backup_files = [
        os.path.join('model', 'image_classifier.py.bak'),
        os.path.join('ui', 'classifier_ui.py.bak'),
        os.path.join('ui', 'components', 'neuron_visualization.py.bak'),
        os.path.join('ui', 'components', 'training_thread.py.bak')
    ]
    
    for rel_path in backup_files:
        # 전체 경로 생성
        full_path = os.path.join(project_root, rel_path)
        if os.path.exists(full_path):
            try:
                # 백업 디렉토리에 동일한 상대 경로로 디렉토리 생성
                rel_dir = os.path.dirname(rel_path)
                backup_subdir = os.path.join(backup_dir, rel_dir)
                os.makedirs(backup_subdir, exist_ok=True)
                
                # 백업 후 삭제
                backup_path = os.path.join(backup_dir, rel_path)
                shutil.copy2(full_path, backup_path)
                os.remove(full_path)
                print(f"제거됨: {full_path} (백업: {backup_path})")
            except Exception as e:
                print(f"오류: {full_path} 제거 실패 - {str(e)}")
        else:
            print(f"정보: {full_path}가 존재하지 않습니다.")
    
    print("\n3단계 정리 완료! 백업 파일들이 정리되었습니다.")

if __name__ == "__main__":
    cleanup_backup_files()
