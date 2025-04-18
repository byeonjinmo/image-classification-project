#!/usr/bin/env python3
"""
딥러닝 이미지 분류 시스템 실행 스크립트

이 스크립트는 개선된 UI를 실행하고 필요한 경우 파일을 자동으로 복사합니다.
"""
import os
import sys
import shutil
from PyQt5.QtWidgets import QApplication, QMessageBox

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def setup_and_run():
    """UI 파일 설정 및 실행"""
    # UI 파일 경로
    ui_dir = os.path.join(project_root, 'ui')
    original_ui_file = os.path.join(ui_dir, 'classifier_ui.py')
    complete_ui_file = os.path.join(ui_dir, 'classifier_ui_complete.py')
    
    # 개선된 UI 파일이 존재하는지 확인
    if os.path.exists(complete_ui_file):
        # 기존 파일 백업 (아직 백업이 없는 경우)
        backup_file = os.path.join(ui_dir, 'classifier_ui.py.original')
        if not os.path.exists(backup_file):
            try:
                shutil.copy2(original_ui_file, backup_file)
                print(f"원본 UI 파일 백업 생성: {backup_file}")
            except Exception as e:
                print(f"백업 생성 실패: {str(e)}")
        
        # 개선된 UI 파일을 실제 UI 파일로 복사
        try:
            shutil.copy2(complete_ui_file, original_ui_file)
            print(f"개선된 UI 파일 적용 완료")
        except Exception as e:
            print(f"UI 파일 복사 실패: {str(e)}")
            sys.exit(1)
    
    # UI 모듈 임포트 (파일 변경 후)
    try:
        # UI 모듈 임포트
        from ui.classifier_ui import ImageClassifierUI
        
        # 애플리케이션 실행
        app = QApplication(sys.argv)
        ui = ImageClassifierUI()
        ui.show()
        
        print("UI 실행 중...")
        sys.exit(app.exec_())
        
    except Exception as e:
        error_msg = f"UI 실행 실패: {str(e)}"
        print(error_msg)
        
        # PyQt가 초기화되었다면 메시지 박스 표시
        if QApplication.instance():
            QMessageBox.critical(None, "오류", error_msg)
        
        sys.exit(1)

if __name__ == "__main__":
    setup_and_run()
