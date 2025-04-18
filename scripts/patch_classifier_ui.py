#!/usr/bin/env python3
"""
classifier_ui.py 파일에 누락된 메서드를 추가하는 패치 스크립트

이 스크립트는 기존 classifier_ui.py 파일을 백업하고,
누락된 'browseDataDir' 메서드와 관련 메서드들을 추가합니다.
"""
import os
import shutil

def apply_patch():
    # 스크립트 실행 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 파일 경로
    ui_dir = os.path.join(script_dir, 'ui')
    original_file = os.path.join(ui_dir, 'classifier_ui.py')
    backup_file = os.path.join(ui_dir, 'classifier_ui.py.bak')
    fixed_simple_file = os.path.join(ui_dir, 'classifier_ui_simple.py')
    missing_methods_file = os.path.join(ui_dir, 'fix_needed_methods.py')
    
    # 1. 기존 파일 백업
    if os.path.exists(original_file):
        print(f"백업 생성: {backup_file}")
        shutil.copy2(original_file, backup_file)
    
    # 2. 간단한 수정: 누락된 메서드 추가
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 간단한 수정 방법: 파일 끝에 누락된 메서드들을 추가
    if 'def browseDataDir' not in content:
        # 누락된 메서드 파일 읽기
        with open(missing_methods_file, 'r', encoding='utf-8') as f:
            missing_methods = f.read()
        
        # 파일에서 class 정의를 찾음
        class_line = content.find("class ImageClassifierUI")
        if class_line != -1:
            # 파일의 내용을 조각으로 나누기
            before_class = content[:class_line]
            class_def = content[class_line:]
            
            # 메서드들이 위치할 곳을 찾기 (클래스의 끝이나 마지막 메서드 후)
            last_method_pos = class_def.rfind("def ")
            if last_method_pos != -1:
                # 마지막 메서드의 끝 위치 찾기
                end_pos = class_def.find("\n\n", last_method_pos)
                if end_pos == -1:  # 파일 끝까지 마지막 메서드가 있는 경우
                    end_pos = len(class_def)
                
                # 필요한 메서드 추가
                with open(missing_methods_file, 'r', encoding='utf-8') as f:
                    methods_to_add = ""
                    for line in f:
                        if line.startswith("def browseDataDir"):
                            methods_to_add += "\n    def browseDataDir(self):\n"
                        elif line.startswith("def"):
                            method_name = line.split("(")[0].split()[1]
                            methods_to_add += f"\n    def {method_name}(self):\n"
                        else:
                            # 들여쓰기 추가
                            if line.strip() and not line.startswith("#"):
                                methods_to_add += "    " + line
                
                # 파일 재구성
                modified_class_def = class_def[:end_pos] + methods_to_add + class_def[end_pos:]
                modified_content = before_class + modified_class_def
                
                # 수정된 내용을 원본 파일에 저장
                with open(original_file, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                print(f"누락된 메서드들이 {original_file}에 성공적으로 추가되었습니다.")
            else:
                print("클래스에서 메서드를 찾을 수 없습니다.")
        else:
            print("ImageClassifierUI 클래스를 찾을 수 없습니다.")
    else:
        print("browseDataDir 메서드가 이미 존재합니다.")

    # 3. 더 단순한 대안: 간단한 버전으로 교체
    # 이 부분은 주석 처리했습니다. 위의 방법으로 해결이 안 될 경우 활성화하세요.
    """
    print(f"기존 파일을 간단한 버전으로 교체: {fixed_simple_file} -> {original_file}")
    shutil.copy2(fixed_simple_file, original_file)
    """

if __name__ == "__main__":
    apply_patch()
    print("패치 완료. 이제 'python scripts/run_app.py'를 실행해 보세요.")
