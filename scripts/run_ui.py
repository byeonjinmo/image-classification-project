#!/usr/bin/env python3
"""
이미지 분류 UI 실행 스크립트
"""
import sys
import os

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from PyQt5.QtWidgets import QApplication
from ui.classifier_ui import ImageClassifierUI

def main():
    """UI 애플리케이션 실행"""
    app = QApplication(sys.argv)
    window = ImageClassifierUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
