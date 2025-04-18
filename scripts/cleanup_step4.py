#!/usr/bin/env python3
"""
프로젝트 정리 스크립트 (4단계)

UI 디렉토리의 중복 파일들과 임시 파일들을 정리합니다:
- ui/classifier_ui_complete.py, ui/classifier_ui_part3.py 등의 중복 파일
- ui/browseDataDir.py, ui/browseDataDir_method.py 등의 패치 관련 파일
"""
import os
import sys
import shutil

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def cleanup_ui_directory():
    """UI 디렉토리 정리 함수"""
    # 백업 디렉토리 생성
    backup_dir = os.path.join(project_root, '_backup', 'ui')
    os.makedirs(backup_dir, exist_ok=True)
    
    # UI 디렉토리 경로
    ui_dir = os.path.join(project_root, 'ui')
    
    # 정리 대상 파일 목록 (패턴 일치)
    cleanup_patterns = [
        'classifier_ui_complete.py',
        'classifier_ui_part',
        'classifier_ui_simple.py',
        'classifier_ui_temp.py',
        'browseDataDir',
        'fix_needed_methods.py',
        'missing_methods.py',
        'training_finished_method.txt',
        'analysis_tab.py'
    ]
    
    # UI 디렉토리의 모든 파일에 대해 검사
    for filename in os.listdir(ui_dir):
        file_path = os.path.join(ui_dir, filename)
        
        # 디렉토리는 건너뛰기
        if os.path.isdir(file_path):
            continue
        
        # 패턴 일치 확인
        should_cleanup = False
        for pattern in cleanup_patterns:
            if pattern in filename:
                should_cleanup = True
                break
        
        # 정리 대상이면 백업 후 삭제
        if should_cleanup:
            try:
                backup_path = os.path.join(backup_dir, filename)
                shutil.copy2(file_path, backup_path)
                os.remove(file_path)
                print(f"제거됨: {file_path} (백업: {backup_path})")
            except Exception as e:
                print(f"오류: {file_path} 제거 실패 - {str(e)}")
    
    print("\n4단계 정리 완료! UI 디렉토리의 중복 파일들이 정리되었습니다.")

if __name__ == "__main__":
    cleanup_ui_directory()
