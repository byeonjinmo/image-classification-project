#!/usr/bin/env python3
"""
프로젝트 정리 통합 스크립트

이 스크립트는 프로젝트 디렉토리 구조를 깔끔하게 정리하는
모든 단계를 순차적으로 실행합니다.
"""
import os
import sys
import shutil
import importlib.util
from datetime import datetime

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def print_section(title):
    """섹션 제목 출력"""
    separator = "=" * 80
    print(f"\n{separator}\n{title}\n{separator}")

def import_and_run_module(module_path, function_name):
    """모듈을 동적으로 임포트하고 지정된 함수 실행"""
    try:
        # 모듈 경로에서 스펙 로드
        spec = importlib.util.spec_from_file_location("cleanup_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 지정된 함수 실행
        if hasattr(module, function_name):
            getattr(module, function_name)()
            return True
        else:
            print(f"오류: {module_path}에서 {function_name} 함수를 찾을 수 없습니다.")
            return False
    except Exception as e:
        print(f"오류: {module_path} 실행 중 예외 발생 - {str(e)}")
        return False

def master_cleanup():
    """모든 정리 단계를 순차적으로 실행"""
    print_section("프로젝트 디렉토리 정리 시작")
    
    # 시작 시간 기록
    start_time = datetime.now()
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 백업 디렉토리 생성
    backup_base = os.path.join(project_root, '_backup')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(backup_base, f"cleanup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)
    print(f"백업 디렉토리 생성됨: {backup_dir}")
    
    # 스크립트 디렉토리
    scripts_dir = os.path.join(project_root, 'scripts')
    
    # 각 정리 단계 실행
    steps = [
        {"file": "cleanup_step1.py", "function": "cleanup_duplicate_files", "desc": "중복 파일 정리"},
        {"file": "cleanup_step2.py", "function": "organize_files", "desc": "파일 재구성"},
        {"file": "cleanup_step3.py", "function": "cleanup_backup_files", "desc": "백업 파일 정리"},
        {"file": "cleanup_step4.py", "function": "cleanup_ui_directory", "desc": "UI 디렉토리 정리"}
    ]
    
    for i, step in enumerate(steps, 1):
        step_file = os.path.join(scripts_dir, step["file"])
        print_section(f"단계 {i}: {step['desc']}")
        
        if os.path.exists(step_file):
            success = import_and_run_module(step_file, step["function"])
            if success:
                print(f"단계 {i} 완료!")
            else:
                print(f"단계 {i} 실패...")
        else:
            print(f"오류: 단계 {i}의 스크립트를 찾을 수 없습니다: {step_file}")
    
    # 최종 정리: 정리 스크립트들도 백업 후 삭제
    print_section("최종 정리: 정리 스크립트 백업")
    
    for step in steps:
        step_file = os.path.join(scripts_dir, step["file"])
        if os.path.exists(step_file):
            try:
                # 백업 디렉토리에 복사
                shutil.copy2(step_file, os.path.join(backup_dir, step["file"]))
                # 스크립트 삭제 (선택적)
                # os.remove(step_file)
                print(f"백업됨: {step_file} -> {os.path.join(backup_dir, step['file'])}")
            except Exception as e:
                print(f"오류: {step_file} 백업 실패 - {str(e)}")
    
    # 종료 시간 및 소요 시간 계산
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_section("프로젝트 디렉토리 정리 완료")
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 소요 시간: {duration}")
    print("\n프로젝트 디렉토리가 성공적으로 정리되었습니다!")

if __name__ == "__main__":
    master_cleanup()
