#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox, 
                           QTabWidget, QGridLayout, QScrollArea, QSizePolicy, QGroupBox,
                           QProgressBar, QLineEdit, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 커스텀 컴포넌트 임포트
from .components import NeuronVisualizationWidget, TrainingThread

# 모델 임포트 (경로는 실제 구조에 맞게 조정)
# 참고: 이 파일은 더 이상 사용되지 않습니다. UI는 ui/classifier_ui.py를 사용합니다.
# from model.classifier import ImageClassifier

class MainWindow(QMainWindow):
    """이미지 분류기 학습 및 시각화를 위한 메인 UI 창"""
    
    def __init__(self):
        super().__init__()
        self.classifier = None
        self.training_thread = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("딥러닝 이미지 분류기 UI")
        self.setGeometry(100, 100, 1200, 800)
        
        # 메인 위젯 및 레이아웃
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 상단 제목
        title_label = QLabel("딥러닝 이미지 분류 학습 시스템")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #2C3E50; margin: 10px;")
        self.main_layout.addWidget(title_label)
        
        # 탭 위젯 생성
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                background: #f0f0f0;
                border: 1px solid #c0c0c0;
                padding: 8px 16px;
                min-width: 100px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
            }
        """)
        
        # 각 탭 생성
        self.setup_tab = QWidget()
        self.training_tab = QWidget()
        self.results_tab = QWidget()
        self.predict_tab = QWidget()
        
        self.setupSetupTab()
        self.setupTrainingTab()
        self.setupResultsTab()
        self.setupPredictTab()
        
        # 탭 추가
        self.tabs.addTab(self.setup_tab, "1. 데이터 및 모델 설정")
        self.tabs.addTab(self.training_tab, "2. 모델 학습")
        self.tabs.addTab(self.results_tab, "3. 학습 결과")
        self.tabs.addTab(self.predict_tab, "4. 이미지 예측")
        
        self.main_layout.addWidget(self.tabs)
        
        # 하단 상태 표시줄
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("시스템이 준비되었습니다")
        
        # 메뉴바 설정
        self.setupMenuBar()
        
    def setupMenuBar(self):
        menubar = self.menuBar()
        
        # 파일 메뉴
        file_menu = menubar.addMenu("파일")
        
        open_action = file_menu.addAction("데이터 폴더 열기")
        open_action.triggered.connect(self.openDataFolder)
        
        save_action = file_menu.addAction("모델 저장")
        save_action.triggered.connect(self.saveModel)
        
        load_action = file_menu.addAction("모델 불러오기")
        load_action.triggered.connect(self.loadModel)
        
        exit_action = file_menu.addAction("종료")
        exit_action.triggered.connect(self.close)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu("도움말")
        
        about_action = help_menu.addAction("프로그램 정보")
        about_action.triggered.connect(self.showAbout)
        
    def setupSetupTab(self):
        layout = QVBoxLayout(self.setup_tab)
        
        # 데이터 선택 그룹
        data_group = QGroupBox("데이터 선택")
        data_layout = QGridLayout(data_group)
        
        data_dir_label = QLabel("데이터 디렉토리:")
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setReadOnly(True)
        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(self.browseDataDir)
        
        data_layout.addWidget(data_dir_label, 0, 0)
        data_layout.addWidget(self.data_dir_edit, 0, 1)
        data_layout.addWidget(browse_btn, 0, 2)
        
        # 검증 분할 설정
        val_split_label = QLabel("검증 데이터 비율:")
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.1, 0.5)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.2)
        data_layout.addWidget(val_split_label, 1, 0)
        data_layout.addWidget(self.val_split_spin, 1, 1)
        
        layout.addWidget(data_group)
        
        # 모델 설정 그룹
        model_group = QGroupBox("모델 설정")
        model_layout = QGridLayout(model_group)
        
        model_type_label = QLabel("모델 타입:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["ResNet18", "ResNet34", "ResNet50"])
        model_layout.addWidget(model_type_label, 0, 0)
        model_layout.addWidget(self.model_combo, 0, 1)
        
        batch_size_label = QLabel("배치 크기:")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(4, 128)
        self.batch_size_spin.setSingleStep(4)
        self.batch_size_spin.setValue(32)
        model_layout.addWidget(batch_size_label, 1, 0)
        model_layout.addWidget(self.batch_size_spin, 1, 1)
        
        epochs_label = QLabel("에폭 수:")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        model_layout.addWidget(epochs_label, 2, 0)
        model_layout.addWidget(self.epochs_spin, 2, 1)
        
        lr_label = QLabel("학습률:")
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        model_layout.addWidget(lr_label, 3, 0)
        model_layout.addWidget(self.lr_spin, 3, 1)
        
        output_dir_label = QLabel("결과 저장 디렉토리:")
        self.output_dir_edit = QLineEdit("results")
        output_dir_browse = QPushButton("찾아보기...")
        output_dir_browse.clicked.connect(self.browseOutputDir)
        
        model_layout.addWidget(output_dir_label, 4, 0)
        model_layout.addWidget(self.output_dir_edit, 4, 1)
        model_layout.addWidget(output_dir_browse, 4, 2)
        
        layout.addWidget(model_group)
        
        # 데이터 미리보기 그룹
        preview_group = QGroupBox("데이터 미리보기")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("데이터를 로드하면 샘플 이미지가 표시됩니다.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("color: #7f8c8d;")
        
        preview_layout.addWidget(self.preview_label)
        
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_content = QWidget()
        self.preview_grid = QGridLayout(self.preview_content)
        self.preview_scroll.setWidget(self.preview_content)
        self.preview_scroll.setVisible(False)
        
        preview_layout.addWidget(self.preview_scroll)
        
        layout.addWidget(preview_group)
        
        # 버튼 영역
        button_layout = QHBoxLayout()
        
        self.load_data_btn = QPushButton("데이터 로드 및 미리보기")
        self.load_data_btn.setStyleSheet("background-color: #3498db; color: white; padding: 8px;")
        self.load_data_btn.clicked.connect(self.loadData)
        
        self.init_model_btn = QPushButton("모델 초기화")
        self.init_model_btn.setStyleSheet("background-color: #2ecc71; color: white; padding: 8px;")
        self.init_model_btn.clicked.connect(self.initializeModel)
        self.init_model_btn.setEnabled(False)
        
        button_layout.addWidget(self.load_data_btn)
        button_layout.addWidget(self.init_model_btn)
        
        layout.addLayout(button_layout)
        
    def setupTrainingTab(self):
        layout = QVBoxLayout(self.training_tab)
        
        # 학습 설정 요약
        summary_group = QGroupBox("학습 설정 요약")
        summary_layout = QGridLayout(summary_group)
        
        self.model_summary_label = QLabel("모델: 아직 초기화되지 않음")
        self.data_summary_label = QLabel("데이터: 아직 로드되지 않음")
        self.params_summary_label = QLabel("파라미터: -")
        
        summary_layout.addWidget(self.model_summary_label, 0, 0)
        summary_layout.addWidget(self.data_summary_label, 1, 0)
        summary_layout.addWidget(self.params_summary_label, 2, 0)
        
        layout.addWidget(summary_group)
        
        # 학습 진행 상황
        progress_group = QGroupBox("학습 진행 상황")
        progress_layout = QVBoxLayout(progress_group)
        
        self.epoch_label = QLabel("에폭: 0/0")
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setValue(0)
        
        self.batch_label = QLabel("배치: 0/0")
        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        
        progress_layout.addWidget(self.epoch_label)
        progress_layout.addWidget(self.epoch_progress)
        progress_layout.addWidget(self.batch_label)
        progress_layout.addWidget(self.batch_progress)
        
        # 학습 지표 실시간 표시
        metrics_layout = QHBoxLayout()
        
        # 손실 그래프
        self.loss_plot = pg.PlotWidget()
        self.loss_plot.setBackground('w')
        self.loss_plot.setTitle("손실 (Loss)")
        self.loss_plot.setLabel('left', "손실")
        self.loss_plot.setLabel('bottom', "에폭")
        self.loss_plot.showGrid(x=True, y=True)
        self.loss_plot.addLegend()
        
        self.train_loss_curve = self.loss_plot.plot([], [], pen=pg.mkPen(color=(255, 0, 0), width=2), name="학습 손실")
        self.val_loss_curve = self.loss_plot.plot([], [], pen=pg.mkPen(color=(0, 0, 255), width=2), name="검증 손실")
        
        # 정확도 그래프
        self.acc_plot = pg.PlotWidget()
        self.acc_plot.setBackground('w')
        self.acc_plot.setTitle("정확도 (Accuracy)")
        self.acc_plot.setLabel('left', "정확도")
        self.acc_plot.setLabel('bottom', "에폭")
        self.acc_plot.showGrid(x=True, y=True)
        self.acc_plot.addLegend()
        
        self.train_acc_curve = self.acc_plot.plot([], [], pen=pg.mkPen(color=(255, 0, 0), width=2), name="학습 정확도")
        self.val_acc_curve = self.acc_plot.plot([], [], pen=pg.mkPen(color=(0, 0, 255), width=2), name="검증 정확도")
        
        metrics_layout.addWidget(self.loss_plot)
        metrics_layout.addWidget(self.acc_plot)
        
        progress_layout.addLayout(metrics_layout)
        
        layout.addWidget(progress_group)
        
        # 뉴런 시각화
        self.neuron_viz = NeuronVisualizationWidget()
        layout.addWidget(self.neuron_viz)
        
        # 제어 버튼
        control_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("학습 시작")
        self.train_btn.setStyleSheet("background-color: #3498db; color: white; padding: 8px;")
        self.train_btn.clicked.connect(self.startTraining)
        self.train_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("학습 중단")
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 8px;")
        self.stop_btn.clicked.connect(self.stopTraining)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.train_btn)
        control_layout.addWidget(self.stop_btn)
        
        layout.addLayout(control_layout)
        
    def setupResultsTab(self):
        layout = QVBoxLayout(self.results_tab)
        
        # 결과 요약
        summary_group = QGroupBox("학습 결과 요약")
        summary_layout = QGridLayout(summary_group)
        
        self.result_model_label = QLabel("모델: 아직 학습되지 않음")
        self.result_data_label = QLabel("데이터: -")
        self.result_time_label = QLabel("학습 시간: -")
        self.result_accuracy_label = QLabel("최종 정확도: -")
        
        summary_layout.addWidget(self.result_model_label, 0, 0)
        summary_layout.addWidget(self.result_data_label, 1, 0)
        summary_layout.addWidget(self.result_time_label, 2, 0)
        summary_layout.addWidget(self.result_accuracy_label, 3, 0)
        
        layout.addWidget(summary_group)
        
        # 결과 시각화 탭
        result_tabs = QTabWidget()
        
        # 혼동 행렬 탭
        conf_matrix_tab = QWidget()
        conf_matrix_layout = QVBoxLayout(conf_matrix_tab)
        
        self.conf_matrix_fig = plt.Figure(figsize=(8, 6))
        self.conf_matrix_canvas = FigureCanvas(self.conf_matrix_fig)
        self.conf_matrix_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        conf_matrix_layout.addWidget(self.conf_matrix_canvas)
        
        # ROC 커브 탭
        roc_tab = QWidget()
        roc_layout = QVBoxLayout(roc_tab)
        
        self.roc_fig = plt.Figure(figsize=(8, 6))
        self.roc_canvas = FigureCanvas(self.roc_fig)
        self.roc_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        roc_layout.addWidget(self.roc_canvas)
        
        # t-SNE 시각화 탭
        tsne_tab = QWidget()
        tsne_layout = QVBoxLayout(tsne_tab)
        
        self.tsne_fig = plt.Figure(figsize=(8, 6))
        self.tsne_canvas = FigureCanvas(self.tsne_fig)
        self.tsne_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        tsne_layout.addWidget(self.tsne_canvas)
        
        # CAM 시각화 탭
        cam_tab = QWidget()
        cam_layout = QVBoxLayout(cam_tab)
        
        self.cam_fig = plt.Figure(figsize=(8, 6))
        self.cam_canvas = FigureCanvas(self.cam_fig)
        self.cam_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        cam_layout.addWidget(self.cam_canvas)
        
        # 탭 추가
        result_tabs.addTab(conf_matrix_tab, "혼동 행렬")
        result_tabs.addTab(roc_tab, "ROC 커브")
        result_tabs.addTab(tsne_tab, "t-SNE 시각화")
        result_tabs.addTab(cam_tab, "클래스 활성화 맵")
        
        layout.addWidget(result_tabs)
        
        # 결과 제어 버튼
        result_btn_layout = QHBoxLayout()
        
        self.save_results_btn = QPushButton("결과 저장")
        self.save_results_btn.setStyleSheet("background-color: #2ecc71; color: white; padding: 8px;")
        self.save_results_btn.clicked.connect(self.saveResults)
        self.save_results_btn.setEnabled(False)
        
        self.open_results_dir_btn = QPushButton("결과 폴더 열기")
        self.open_results_dir_btn.clicked.connect(self.openResultsDir)
        self.open_results_dir_btn.setEnabled(False)
        
        result_btn_layout.addWidget(self.save_results_btn)
        result_btn_layout.addWidget(self.open_results_dir_btn)
        
        layout.addLayout(result_btn_layout)
        
    def setupPredictTab(self):
        layout = QVBoxLayout(self.predict_tab)
        
        # 모델 선택
        model_group = QGroupBox("모델 선택")
        model_layout = QHBoxLayout(model_group)
        
        self.predict_model_label = QLabel("모델 파일:")
        self.predict_model_path = QLineEdit()
        self.predict_model_path.setReadOnly(True)
        self.predict_model_browse = QPushButton("찾아보기...")
        self.predict_model_browse.clicked.connect(self.browsePredictModel)
        
        model_layout.addWidget(self.predict_model_label)
        model_layout.addWidget(self.predict_model_path)
        model_layout.addWidget(self.predict_model_browse)
        
        layout.addWidget(model_group)
        
        # 이미지 선택 및 표시
        image_group = QGroupBox("예측 이미지")
        image_layout = QGridLayout(image_group)
        
        self.predict_image_label = QLabel("이미지 파일:")
        self.predict_image_path = QLineEdit()
        self.predict_image_path.setReadOnly(True)
        self.predict_image_browse = QPushButton("찾아보기...")
        self.predict_image_browse.clicked.connect(self.browsePredictImage)
        
        image_layout.addWidget(self.predict_image_label, 0, 0)
        image_layout.addWidget(self.predict_image_path, 0, 1)
        image_layout.addWidget(self.predict_image_browse, 0, 2)
        
        # 이미지 표시
        self.image_display = QLabel("이미지를 선택하면 여기에 표시됩니다.")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("background-color: #f0f0f0; padding: 20px;")
        self.image_display.setMinimumHeight(300)
        
        image_layout.addWidget(self.image_display, 1, 0, 1, 3)
        
        layout.addWidget(image_group)
        
        # 예측 결과
        predict_group = QGroupBox("예측 결과")
        predict_layout = QVBoxLayout(predict_group)
        
        self.prediction_label = QLabel("예측 클래스: -")
        self.prediction_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.prediction_label.setAlignment(Qt.AlignCenter)
        
        self.confidence_label = QLabel("신뢰도: -")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        
        predict_layout.addWidget(self.prediction_label)
        predict_layout.addWidget(self.confidence_label)
        
        # 신뢰도 바 차트
        self.confidence_chart = pg.PlotWidget()
        self.confidence_chart.setBackground('w')
        self.confidence_chart.setTitle("클래스별 신뢰도")
        self.confidence_chart.setLabel('left', "신뢰도")
        self.confidence_chart.setLabel('bottom', "클래스")
        self.confidence_chart.showGrid(y=True)
        
        predict_layout.addWidget(self.confidence_chart)
        
        layout.addWidget(predict_group)
        
        # 예측 버튼
        self.predict_btn = QPushButton("예측 실행")
        self.predict_btn.setStyleSheet("background-color: #3498db; color: white; padding: 8px;")
        self.predict_btn.clicked.connect(self.runPrediction)
        self.predict_btn.setEnabled(False)
        
        layout.addWidget(self.predict_btn)
    
    # 이벤트 핸들러 및 기능 메서드들
    def browseDataDir(self):
        folder = QFileDialog.getExistingDirectory(self, "데이터 디렉토리 선택")
        if folder:
            self.data_dir_edit.setText(folder)
    
    def browseOutputDir(self):
        folder = QFileDialog.getExistingDirectory(self, "결과 저장 디렉토리 선택")
        if folder:
            self.output_dir_edit.setText(folder)
    
    def browsePredictModel(self):
        file, _ = QFileDialog.getOpenFileName(self, "모델 파일 선택", "", "PyTorch 모델 (*.pth)")
        if file:
            self.predict_model_path.setText(file)
            self.predict_btn.setEnabled(True)
    
    def browsePredictImage(self):
        file, _ = QFileDialog.getOpenFileName(self, "이미지 파일 선택", "", "이미지 파일 (*.jpg *.jpeg *.png)")
        if file:
            self.predict_image_path.setText(file)
            
            # 이미지 표시
            pixmap = QPixmap(file)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_display.setPixmap(pixmap)
