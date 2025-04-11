#!/usr/bin/env python3
"""
파일 처리 관련 유틸리티 함수
"""
import os
import shutil
from PyQt5.QtWidgets import QFileDialog, QMessageBox

def select_directory(parent, caption="디렉토리 선택"):
    """
    디렉토리 선택 대화상자 표시
    
    Args:
        parent: 부모 위젯
        caption (str): 대화상자 제목
    
    Returns:
        str: 선택한 디렉토리 경로 (취소시 빈 문자열)
    """
    directory = QFileDialog.getExistingDirectory(
        parent, caption, os.path.expanduser("~"),
        QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
    )
    return directory

def select_file(parent, caption="파일 선택", file_filter="모든 파일 (*.*)"):
    """
    파일 선택 대화상자 표시
    
    Args:
        parent: 부모 위젯
        caption (str): 대화상자 제목
        file_filter (str): 파일 필터 (예: "이미지 파일 (*.jpg *.png)")
    
    Returns:
        str: 선택한 파일 경로 (취소시 빈 문자열)
    """
    file_path, _ = QFileDialog.getOpenFileName(
        parent, caption, os.path.expanduser("~"), file_filter
    )
    return file_path

def save_file(parent, caption="파일 저장", file_filter="모든 파일 (*.*)", default_name=""):
    """
    파일 저장 대화상자 표시
    
    Args:
        parent: 부모 위젯
        caption (str): 대화상자 제목
        file_filter (str): 파일 필터 (예: "이미지 파일 (*.jpg *.png)")
        default_name (str): 기본 파일명
    
    Returns:
        str: 저장할 파일 경로 (취소시 빈 문자열)
    """
    file_path, _ = QFileDialog.getSaveFileName(
        parent, caption, os.path.join(os.path.expanduser("~"), default_name), file_filter
    )
    return file_path

def ensure_directory(directory):
    """
    디렉토리가 존재하는지 확인하고, 없으면 생성
    
    Args:
        directory (str): 확인/생성할 디렉토리 경로
        
    Returns:
        bool: 디렉토리가 존재하거나 생성됐으면 True, 그렇지 않으면 False
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"디렉토리 생성 오류: {str(e)}")
        return False

def get_file_count(directory, extensions=None):
    """
    디렉토리 내 파일 개수 반환
    
    Args:
        directory (str): 검색할 디렉토리 경로
        extensions (list, optional): 파일 확장자 목록 (예: ['.jpg', '.png'])
    
    Returns:
        int: 파일 개수
    """
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for root, dirs, files in os.walk(directory):
        if extensions:
            count += sum(1 for f in files if os.path.splitext(f)[1].lower() in extensions)
        else:
            count += len(files)
    
    return count

def get_image_files(directory, recursive=False):
    """
    디렉토리 내 이미지 파일 목록 반환
    
    Args:
        directory (str): 검색할 디렉토리 경로
        recursive (bool): 하위 디렉토리도 검색할지 여부
        
    Returns:
        list: 이미지 파일 경로 목록
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)) and \
               os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(directory, file))
    
    return image_files

def copy_file(source, destination, overwrite=False):
    """
    파일 복사
    
    Args:
        source (str): 원본 파일 경로
        destination (str): 대상 파일 경로
        overwrite (bool): 대상 파일이 이미 존재할 경우 덮어쓸지 여부
        
    Returns:
        bool: 복사 성공 여부
    """
    try:
        if os.path.exists(destination) and not overwrite:
            return False
        
        # 대상 디렉토리가 없으면 생성
        dest_dir = os.path.dirname(destination)
        if dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        print(f"파일 복사 오류: {str(e)}")
        return False

def show_error_message(parent, title, message):
    """
    오류 메시지 대화상자 표시
    
    Args:
        parent: 부모 위젯
        title (str): 대화상자 제목
        message (str): 오류 메시지
    """
    QMessageBox.critical(parent, title, message)

def show_info_message(parent, title, message):
    """
    정보 메시지 대화상자 표시
    
    Args:
        parent: 부모 위젯
        title (str): 대화상자 제목
        message (str): 정보 메시지
    """
    QMessageBox.information(parent, title, message)

def show_confirmation_message(parent, title, message):
    """
    확인 메시지 대화상자 표시
    
    Args:
        parent: 부모 위젯
        title (str): 대화상자 제목
        message (str): 확인 메시지
        
    Returns:
        bool: '예'를 선택하면 True, 그렇지 않으면 False
    """
    reply = QMessageBox.question(
        parent, title, message,
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No
    )
    return reply == QMessageBox.Yes
