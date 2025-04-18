#!/usr/bin/env python3
"""
프로젝트 정리 스크립트 (2단계)

정리가 필요한 파일들을 적절한 디렉토리로 이동하거나 삭제합니다:
- README_UPDATED.md 삭제
- add_browse_methods.py와 patch_classifier_ui.py를 scripts 디렉토리로 이동
- 문서 파일을 위한 docs 디렉토리 생성 및 UI_README.md 이동
"""
import os
import sys
import shutil

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def organize_files():
    """파일 정리 함수"""
    # 백업 디렉토리 생성
    backup_dir = os.path.join(project_root, '_backup')
    os.makedirs(backup_dir, exist_ok=True)
    
    # docs 디렉토리 생성
    docs_dir = os.path.join(project_root, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    print(f"문서 디렉토리 생성됨: {docs_dir}")
    
    # 1. README_UPDATED.md 삭제 (이미 내용이 README.md로 병합됨)
    readme_updated_path = os.path.join(project_root, 'README_UPDATED.md')
    if os.path.exists(readme_updated_path):
        try:
            # 백업 후 삭제
            backup_path = os.path.join(backup_dir, 'README_UPDATED.md')
            shutil.copy2(readme_updated_path, backup_path)
            os.remove(readme_updated_path)
            print(f"제거됨: {readme_updated_path} (백업: {backup_path})")
        except Exception as e:
            print(f"오류: {readme_updated_path} 제거 실패 - {str(e)}")
    
    # 2. 패치 관련 파일을 scripts 디렉토리로 이동
    patch_files = [
        'add_browse_methods.py',
        'patch_classifier_ui.py'
    ]
    
    for file_name in patch_files:
        source_path = os.path.join(project_root, file_name)
        dest_path = os.path.join(project_root, 'scripts', file_name)
        
        if os.path.exists(source_path):
            try:
                # 대상 경로에 이미 파일이 있는지 확인
                if os.path.exists(dest_path):
                    # 이미 존재하면 백업 후 대체
                    backup_path = os.path.join(backup_dir, file_name)
                    shutil.copy2(source_path, backup_path)
                    os.remove(source_path)
                    print(f"대상 경로에 이미 파일이 존재함: {dest_path}")
                    print(f"제거됨: {source_path} (백업: {backup_path})")
                else:
                    # 이동
                    shutil.move(source_path, dest_path)
                    print(f"이동됨: {source_path} -> {dest_path}")
            except Exception as e:
                print(f"오류: {source_path} 이동 실패 - {str(e)}")
    
    # 3. 문서 파일을 docs 디렉토리로 이동
    doc_files = [
        'UI_README.md'
    ]
    
    for file_name in doc_files:
        source_path = os.path.join(project_root, file_name)
        dest_path = os.path.join(docs_dir, file_name)
        
        if os.path.exists(source_path):
            try:
                # 이동
                shutil.move(source_path, dest_path)
                print(f"이동됨: {source_path} -> {dest_path}")
            except Exception as e:
                print(f"오류: {source_path} 이동 실패 - {str(e)}")
    
    print("\n2단계 정리 완료! 파일들이 적절한 디렉토리로 정리되었습니다.")

if __name__ == "__main__":
    organize_files()
