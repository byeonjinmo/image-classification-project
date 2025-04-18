#!/usr/bin/env python3
"""
개선된 classifier_ui.py 패치 스크립트

이 스크립트는 classifier_ui.py 파일에 누락된 메서드들을 안정적으로 추가합니다.
"""
import os
import sys
import shutil
import re

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def apply_improved_patch():
    """패치 적용 함수"""
    # 파일 경로
    ui_dir = os.path.join(project_root, 'ui')
    original_file = os.path.join(ui_dir, 'classifier_ui.py')
    backup_file = os.path.join(ui_dir, 'classifier_ui.py.bak.improved')
    fix_methods_file = os.path.join(ui_dir, 'fix_needed_methods.py')
    
    # 기존 파일 백업
    print(f"기존 파일 백업: {backup_file}")
    shutil.copy2(original_file, backup_file)
    
    # 필요한 메서드 로드
    with open(fix_methods_file, 'r', encoding='utf-8') as f:
        methods_content = f.read()
    
    # 메서드 파싱
    method_pattern = r'def (\w+)\(self.*?\):'
    method_blocks = re.findall(r'def (\w+)\(self.*?\):(?:.*?)(?=def \w+\(self|\Z)', methods_content, re.DOTALL)
    
    # classifier_ui.py 파일 읽기
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 각 메서드에 대해 파일에 없는지 확인하고 추가
    for method_name in method_blocks:
        method_def_pattern = rf'def {method_name}\(self.*?\):'
        if not re.search(method_def_pattern, content):
            print(f"누락된 메서드 추가: {method_name}")
            
            # 메서드 블록 추출
            method_block_pattern = rf'def {method_name}\(self.*?\):.*?(?=def \w+\(self|\Z)'
            method_block_match = re.search(method_block_pattern, methods_content, re.DOTALL)
            
            if method_block_match:
                method_block = method_block_match.group(0)
                
                # 메서드 블록에 적절한 들여쓰기 추가
                indented_method = "\n    " + method_block.replace("\n", "\n    ")
                
                # 클래스 끝에 메서드 추가
                class_pattern = r'class ImageClassifierUI\(.*?\):.*'
                class_match = re.search(class_pattern, content, re.DOTALL)
                
                if class_match:
                    class_content = class_match.group(0)
                    # 클래스 끝에 메서드 추가
                    updated_content = content.replace(class_content, class_content + indented_method)
                    content = updated_content
                else:
                    print("경고: ImageClassifierUI 클래스를 찾을 수 없습니다.")
            else:
                print(f"경고: {method_name} 메서드 블록을 추출할 수 없습니다.")
        else:
            print(f"메서드 이미 존재함: {method_name}")
    
    # 수정된 내용 저장
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("패치 완료!")

if __name__ == "__main__":
    apply_improved_patch()
