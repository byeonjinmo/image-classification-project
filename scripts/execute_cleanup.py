#!/usr/bin/env python3
"""
루트 디렉토리의 중복 파일 정리 스크립트

이 스크립트는 백업한 후 루트 디렉토리의 중복 파일들을 삭제합니다.
"""
import os
import sys
import shutil

# 프로젝트 루트 디렉토리
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 삭제할 파일 목록
files_to_delete = [
    'improved_patch.py',
    'run_training.py',
    'test_model.py',
    'updated_launcher.py',
    'README_UPDATED.md'
]

# 백업 디렉토리
backup_dir = os.path.join(project_root, '_backup')
os.makedirs(backup_dir, exist_ok=True)

# 각 파일을 백업하고 삭제
for filename in files_to_delete:
    file_path = os.path.join(project_root, filename)
    if os.path.exists(file_path):
        # 백업 경로
        backup_path = os.path.join(backup_dir, filename)
        
        # 백업 파일이 없으면 복사
        if not os.path.exists(backup_path):
            shutil.copy2(file_path, backup_path)
            print(f"백업 생성: {backup_path}")
        
        # 파일 삭제
        os.remove(file_path)
        print(f"삭제됨: {file_path}")
    else:
        print(f"파일을 찾을 수 없음: {file_path}")

print("정리 완료!")
