#!/usr/bin/env python3
"""
프로젝트 정리 스크립트

이 스크립트는 스크립트 폴더로 이동된 후에도 루트 디렉토리에 남아있는
중복 파일들을 제거하여 프로젝트를 깔끔하게 정리합니다.
"""
import os
import sys
import shutil

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def main():
    """중복 파일 정리 메인 함수"""
    # 이미 scripts/ 디렉토리로 이동된 파일들
    duplicate_files = [
        'test_model.py',
        'run_training.py',
        'updated_launcher.py',
        'improved_patch.py'
    ]
    
    # 각 파일에 대해 루트 디렉토리에서 확인 후 제거
    for file_name in duplicate_files:
        root_file_path = os.path.join(project_root, file_name)
        script_file_path = os.path.join(project_root, 'scripts', file_name)
        
        # 루트와 scripts/ 디렉토리 모두에 파일이 존재하는지 확인
        if os.path.exists(root_file_path) and os.path.exists(script_file_path):
            try:
                # 백업 디렉토리 생성
                backup_dir = os.path.join(project_root, '_backup')
                os.makedirs(backup_dir, exist_ok=True)
                
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
    
    # README_UPDATED.md를 README.md와 병합 또는 제거
    updated_readme = os.path.join(project_root, 'README_UPDATED.md')
    if os.path.exists(updated_readme):
        try:
            os.remove(updated_readme)
            print(f"제거됨: {updated_readme} (이미 README.md로 병합됨)")
        except Exception as e:
            print(f"오류: {updated_readme} 제거 실패 - {str(e)}")
    
    print("\n정리 완료! 프로젝트 구조가 깔끔하게 정리되었습니다.")

if __name__ == "__main__":
    main()
