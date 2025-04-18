#!/usr/bin/env python3
"""
프로젝트 정리 스크립트 (1단계)

루트 디렉토리에 있는 중복 파일들을 scripts 디렉토리로 이미 이동한 후
루트에 남아있는 원본을 삭제합니다.
"""
import os
import sys
import shutil

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def cleanup_duplicate_files():
    """중복 파일 정리 함수"""
    # 이미 scripts/ 디렉토리로 이동된 파일들
    duplicate_files = [
        'improved_patch.py',
        'run_training.py',
        'test_model.py',
        'updated_launcher.py'
    ]
    
    # 백업 디렉토리 생성
    backup_dir = os.path.join(project_root, '_backup')
    os.makedirs(backup_dir, exist_ok=True)
    print(f"백업 디렉토리 생성됨: {backup_dir}")
    
    # 각 파일에 대해 루트 디렉토리에서 확인 후 제거
    for file_name in duplicate_files:
        root_file_path = os.path.join(project_root, file_name)
        script_file_path = os.path.join(project_root, 'scripts', file_name)
        
        # 루트와 scripts/ 디렉토리 모두에 파일이 존재하는지 확인
        if os.path.exists(root_file_path) and os.path.exists(script_file_path):
            try:
                # 백업 후 삭제
                backup_path = os.path.join(backup_dir, file_name)
                shutil.copy2(root_file_path, backup_path)
                os.remove(root_file_path)
                print(f"제거됨: {root_file_path} (백업: {backup_path})")
            except Exception as e:
                print(f"오류: {root_file_path} 제거 실패 - {str(e)}")
        elif os.path.exists(root_file_path):
            print(f"경고: {script_file_path}가 없습니다. {root_file_path}는 그대로 유지됩니다.")
        else:
            print(f"정보: {root_file_path}가 이미 제거되었습니다.")
    
    print("\n1단계 정리 완료! 중복 파일들이 정리되었습니다.")

if __name__ == "__main__":
    cleanup_duplicate_files()
