#!/usr/bin/env python3
"""
classifier_ui.py 파일에 browseDataDir 메서드를 추가하는 간단한 패치 스크립트
"""
import os
import shutil

def add_methods():
    # 파일 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ui_dir = os.path.join(current_dir, 'ui')
    
    # 원본 파일과 백업 파일 경로
    original_file = os.path.join(ui_dir, 'classifier_ui.py')
    backup_file = os.path.join(ui_dir, 'classifier_ui.py.bak')
    temp_file = os.path.join(ui_dir, 'classifier_ui.py.temp')
    
    # 백업 생성
    shutil.copy2(original_file, backup_file)
    print(f"원본 파일 백업 생성: {backup_file}")
    
    # 필요한 메서드 정의
    browse_methods = """
    def browseDataDir(self):
        \"\"\"데이터 디렉토리 선택 대화상자 표시\"\"\"
        dir_path = QFileDialog.getExistingDirectory(self, "데이터 디렉토리 선택")
        if dir_path:
            self.data_dir_edit.setText(dir_path)
            self.status_bar.showMessage(f"데이터 디렉토리 선택됨: {dir_path}")
            
            # 데이터 로드 버튼 활성화
            self.load_data_btn.setEnabled(True)
            
    def browseOutputDir(self):
        \"\"\"결과 저장 디렉토리 선택 대화상자 표시\"\"\"
        dir_path = QFileDialog.getExistingDirectory(self, "결과 저장 디렉토리 선택")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
            self.status_bar.showMessage(f"결과 저장 디렉토리 선택됨: {dir_path}")
    """
    
    # 원본 파일 수정
    with open(original_file, 'r', encoding='utf-8') as f_in:
        with open(temp_file, 'w', encoding='utf-8') as f_out:
            # 파일을 한 줄씩 읽어서 클래스 정의 끝 부분 찾기
            class_end = False
            showAbout_found = False
            
            for line in f_in:
                # showAbout 메서드 다음에 추가
                if 'def showAbout' in line:
                    showAbout_found = True
                    f_out.write(line)
                elif showAbout_found and line.strip() == '':
                    # showAbout 메서드 다음 빈 줄에 새 메서드 추가
                    f_out.write(line)
                    f_out.write(browse_methods)
                    showAbout_found = False  # 한 번만 추가하도록 플래그 리셋
                else:
                    f_out.write(line)
    
    # 임시 파일을 원본 파일로 교체
    shutil.move(temp_file, original_file)
    print(f"browseDataDir 및 browseOutputDir 메서드가 {original_file}에 추가되었습니다.")

if __name__ == "__main__":
    add_methods()
    print("수정 완료. 이제 'python scripts/run_app.py'를 실행하세요.")
