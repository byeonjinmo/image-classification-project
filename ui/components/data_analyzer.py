#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QComboBox, QGroupBox, QGridLayout, QTabWidget,
                             QCheckBox, QSpinBox, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

class DatasetAnalyzer(QWidget):
    """데이터셋 분석 및 시각화 위젯"""
    
    def __init__(self, data_loader=None):
        super().__init__()
        self.data_loader = data_loader
        self.class_counts = {}
        self.image_sizes = []
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # 제목
        title = QLabel("데이터셋 분석")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 탭 위젯 생성
        self.tabs = QTabWidget()
        
        # 클래스 분포 탭
        class_dist_tab = QWidget()
        class_dist_layout = QVBoxLayout(class_dist_tab)
        
        # 클래스 분포 그래프
        class_dist_group = QGroupBox("클래스 분포")
        class_dist_inner_layout = QVBoxLayout(class_dist_group)
        
        self.class_dist_fig = plt.Figure(figsize=(8, 6))
        self.class_dist_canvas = FigureCanvas(self.class_dist_fig)
        self.class_dist_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        class_dist_inner_layout.addWidget(self.class_dist_canvas)
        
        # 클래스 불균형 정보
        self.class_balance_info = QLabel("데이터셋을 분석하면 클래스 불균형 정보가 여기에 표시됩니다.")
        class_dist_inner_layout.addWidget(self.class_balance_info)
        
        class_dist_layout.addWidget(class_dist_group)
        
        # 버튼 영역
        class_dist_btn_layout = QHBoxLayout()
        
        self.analyze_class_btn = QPushButton("클래스 분포 분석")
        self.analyze_class_btn.clicked.connect(self.analyze_class_distribution)
        
        class_dist_btn_layout.addWidget(self.analyze_class_btn)
        class_dist_layout.addLayout(class_dist_btn_layout)
        
        # 이미지 크기 탭
        img_size_tab = QWidget()
        img_size_layout = QVBoxLayout(img_size_tab)
        
        # 이미지 크기 그래프
        img_size_group = QGroupBox("이미지 크기 분포")
        img_size_inner_layout = QVBoxLayout(img_size_group)
        
        self.img_size_fig = plt.Figure(figsize=(8, 6))
        self.img_size_canvas = FigureCanvas(self.img_size_fig)
        self.img_size_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        img_size_inner_layout.addWidget(self.img_size_canvas)
        
        # 이미지 크기 정보
        self.img_size_info = QLabel("데이터셋을 분석하면 이미지 크기 정보가 여기에 표시됩니다.")
        img_size_inner_layout.addWidget(self.img_size_info)
        
        img_size_layout.addWidget(img_size_group)
        
        # 버튼 영역
        img_size_btn_layout = QHBoxLayout()
        
        self.analyze_size_btn = QPushButton("이미지 크기 분석")
        self.analyze_size_btn.clicked.connect(self.analyze_image_sizes)
        
        img_size_btn_layout.addWidget(self.analyze_size_btn)
        img_size_layout.addLayout(img_size_btn_layout)
        
        # 탭 추가
        self.tabs.addTab(class_dist_tab, "클래스 분포")
        self.tabs.addTab(img_size_tab, "이미지 크기 분포")
        
        layout.addWidget(self.tabs)
    
    def set_data_loader(self, data_loader):
        """데이터 로더 설정"""
        self.data_loader = data_loader
    
    def analyze_class_distribution(self):
        """클래스 분포 분석"""
        if not self.data_loader:
            return
        
        # 클래스별 이미지 개수 계산
        class_counts = {}
        for img_path, label_idx in self.data_loader.dataset.samples:
            class_name = list(self.data_loader.dataset.class_to_idx.keys())[list(self.data_loader.dataset.class_to_idx.values()).index(label_idx)]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        self.class_counts = class_counts
        
        # 그래프 그리기
        self.class_dist_fig.clear()
        ax = self.class_dist_fig.add_subplot(111)
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # 내림차순 정렬
        sorted_indices = np.argsort(counts)[::-1]
        classes = [classes[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # 막대 그래프
        bars = ax.bar(classes, counts, color='skyblue')
        
        # 불균형 강조 - 평균보다 크게 차이나는 클래스 강조
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        for i, count in enumerate(counts):
            if count > mean_count + std_count:
                bars[i].set_color('darkgreen')
            elif count < mean_count - std_count:
                bars[i].set_color('crimson')
        
        # 그래프 설정
        ax.set_title('클래스별 이미지 개수')
        ax.set_xlabel('클래스')
        ax.set_ylabel('이미지 개수')
        ax.tick_params(axis='x', rotation=45)
        
        # 각 막대 위에 숫자 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(counts[i]),
                    ha='center', va='bottom')
        
        self.class_dist_fig.tight_layout()
        self.class_dist_canvas.draw()
        
        # 클래스 불균형 정보 업데이트
        max_class = classes[0]
        min_class = classes[-1]
        max_count = counts[0]
        min_count = counts[-1]
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        info_text = f"클래스 수: {len(classes)}\n"
        info_text += f"총 이미지 수: {sum(counts)}\n"
        info_text += f"클래스당 평균 이미지 수: {np.mean(counts):.1f} (±{np.std(counts):.1f})\n"
        info_text += f"최대 이미지 클래스: {max_class} ({max_count}개)\n"
        info_text += f"최소 이미지 클래스: {min_class} ({min_count}개)\n"
        info_text += f"불균형 비율 (최대/최소): {imbalance_ratio:.2f}\n"
        
        if imbalance_ratio > 10:
            info_text += "\n경고: 심각한 클래스 불균형이 감지되었습니다. 데이터 증강 또는 가중치 부여를 고려하세요."
        elif imbalance_ratio > 3:
            info_text += "\n주의: 중간 정도의 클래스 불균형이 감지되었습니다. 데이터 증강을 고려해 보세요."
        
        self.class_balance_info.setText(info_text)
    
    def analyze_image_sizes(self):
        """이미지 크기 분포 분석"""
        if not self.data_loader:
            return
        
        # 이미지 크기 분석 (샘플링)
        from PIL import Image
        import random
        
        size_counts = {}
        aspect_ratios = []
        
        # 최대 100개 이미지만 분석 (속도 향상)
        samples = self.data_loader.dataset.samples
        if len(samples) > 100:
            samples = random.sample(samples, 100)
        
        for img_path, _ in samples:
            try:
                with Image.open(img_path) as img:
                    size = img.size  # (width, height)
                    size_str = f"{size[0]}x{size[1]}"
                    
                    if size_str not in size_counts:
                        size_counts[size_str] = 0
                    size_counts[size_str] += 1
                    
                    # 종횡비 계산
                    aspect_ratio = size[0] / size[1]
                    aspect_ratios.append(aspect_ratio)
            except Exception as e:
                print(f"이미지 로드 실패: {img_path} - {str(e)}")
        
        self.image_sizes = size_counts
        
        # 그래프 그리기
        self.img_size_fig.clear()
        
        # 서브플롯 2개 생성
        ax1 = self.img_size_fig.add_subplot(211)
        ax2 = self.img_size_fig.add_subplot(212)
        
        # 이미지 크기 분포
        sizes = list(size_counts.keys())
        counts = list(size_counts.values())
        
        # 내림차순 정렬
        sorted_indices = np.argsort(counts)[::-1]
        sizes = [sizes[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # 상위 10개만 표시
        if len(sizes) > 10:
            sizes = sizes[:10]
            counts = counts[:10]
            
        ax1.bar(sizes, counts, color='lightblue')
        ax1.set_title('상위 이미지 크기 분포')
        ax1.set_xlabel('이미지 크기 (너비x높이)')
        ax1.set_ylabel('이미지 개수')
        ax1.tick_params(axis='x', rotation=45)
        
        # 종횡비 히스토그램
        ax2.hist(aspect_ratios, bins=20, color='lightgreen', alpha=0.7)
        ax2.set_title('이미지 종횡비 분포')
        ax2.set_xlabel('종횡비 (너비/높이)')
        ax2.set_ylabel('이미지 개수')
        ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)  # 정사각형 라인
        
        self.img_size_fig.tight_layout()
        self.img_size_canvas.draw()
        
        # 이미지 크기 정보 업데이트
        most_common_size = sizes[0] if sizes else "없음"
        
        # 종횡비 통계
        mean_ratio = np.mean(aspect_ratios)
        median_ratio = np.median(aspect_ratios)
        std_ratio = np.std(aspect_ratios)
        
        # 크기 통계
        all_sizes = []
        for size_str, count in size_counts.items():
            w, h = map(int, size_str.split('x'))
            all_sizes.extend([(w, h)] * count)
        
        all_widths = [s[0] for s in all_sizes]
        all_heights = [s[1] for s in all_sizes]
        
        avg_width = np.mean(all_widths)
        avg_height = np.mean(all_heights)
        
        info_text = f"분석한 이미지 수: {len(samples)}\n"
        info_text += f"서로 다른 이미지 크기: {len(size_counts)}개\n"
        info_text += f"가장 흔한 이미지 크기: {most_common_size} ({size_counts.get(most_common_size, 0)}개)\n"
        info_text += f"평균 이미지 크기: {avg_width:.1f} x {avg_height:.1f}\n"
        info_text += f"평균 종횡비: {mean_ratio:.2f}\n"
        info_text += f"중간값 종횡비: {median_ratio:.2f}\n"
        
        if len(size_counts) > 5:
            info_text += "\n주의: 데이터셋에 다양한 이미지 크기가 있습니다. 전처리 단계에서 크기를 통일하는 것이 좋습니다."
        
        self.img_size_info.setText(info_text)
