        # 데이터 증강 설정
        augmentation_group = QGroupBox("데이터 증강 설정")
        augmentation_layout = QGridLayout(augmentation_group)
        
        # 증강 사용 여부
        self.use_augmentation_check = QCheckBox("데이터 증강 사용")
        self.use_augmentation_check.setChecked(True)
        augmentation_layout.addWidget(self.use_augmentation_check, 0, 0, 1, 2)
        
        # 회전 각도 설정
        rotation_label = QLabel("회전 각도(\u00b0):")
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(0, 180)
        self.rotation_spin.setValue(15)
        augmentation_layout.addWidget(rotation_label, 1, 0)
        augmentation_layout.addWidget(self.rotation_spin, 1, 1)
        
        # 크기 변환 설정
        scale_label = QLabel("크기 변환 비율:")
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 0.5)
        self.scale_spin.setValue(0.1)
        self.scale_spin.setSingleStep(0.05)
        augmentation_layout.addWidget(scale_label, 2, 0)
        augmentation_layout.addWidget(self.scale_spin, 2, 1)
        
        # 반전 설정
        self.horizontal_flip_check = QCheckBox("수평 반전")
        self.horizontal_flip_check.setChecked(True)
        self.vertical_flip_check = QCheckBox("수직 반전")
        self.vertical_flip_check.setChecked(False)
        augmentation_layout.addWidget(self.horizontal_flip_check, 3, 0)
        augmentation_layout.addWidget(self.vertical_flip_check, 3, 1)
        
        # 밝기/대비 변환
        self.brightness_contrast_check = QCheckBox("밝기/대비 변환")
        self.brightness_contrast_check.setChecked(True)
        augmentation_layout.addWidget(self.brightness_contrast_check, 4, 0, 1, 2)
        
        layout.addWidget(augmentation_group)#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox, 
                             QTabWidget, QGridLayout, QScrollArea, QSizePolicy, QGroupBox,
                             QProgressBar, QLineEdit, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QFont
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
import time
import random
from model.classifier import ImageClassifier

# UI 컴포넌트 임포트
from ui.components import NeuronVisualizationWidget, TrainingThread
from ui.components.checkpoint_manager import CheckpointManager
from ui.components.data_analyzer import DatasetAnalyzer

# 프로그래스 색상 지정
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class ImageClassifierUI(QMainWindow):
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
        self.checkpoint_tab = QWidget()
        self.analysis_tab = QWidget()
        
        self.setupSetupTab()
        self.setupTrainingTab()
        self.setupResultsTab()
        self.setupPredictTab()
        self.setupCheckpointTab()
        self.setupAnalysisTab()
        
        # 탭 추가
        self.tabs.addTab(self.setup_tab, "1. 데이터 및 모델 설정")
        self.tabs.addTab(self.training_tab, "2. 모델 학습")
        self.tabs.addTab(self.results_tab, "3. 학습 결과")
        self.tabs.addTab(self.predict_tab, "4. 이미지 예측")
        self.tabs.addTab(self.checkpoint_tab, "5. 체크포인트 관리")
        self.tabs.addTab(self.analysis_tab, "6. 데이터 분석")
        
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
        
        load_checkpoint_action = file_menu.addAction("체크포인트 불러오기")
        load_checkpoint_action.triggered.connect(self.loadCheckpoint)
        
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
        self.model_combo.addItems(["ResNet18", "ResNet34", "ResNet50", "MobileNetV2", "EfficientNetB0", "VisionTransformer"])
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
        
        # 옵티맀이저 선택 옵션 추가
        optimizer_label = QLabel("옵티맀이저:")
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "AdamW", "RMSprop"])
        model_layout.addWidget(optimizer_label, 4, 0)
        model_layout.addWidget(self.optimizer_combo, 4, 1)
        
        # 학습률 스케줄러 옵션 추가
        scheduler_label = QLabel("학습률 스케줄러:")
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["없음", "StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"])
        model_layout.addWidget(scheduler_label, 5, 0)
        model_layout.addWidget(self.scheduler_combo, 5, 1)
        
        output_dir_label = QLabel("결과 저장 디렉토리:")
        self.output_dir_edit = QLineEdit("results")
        output_dir_browse = QPushButton("찾아보기...")
        output_dir_browse.clicked.connect(self.browseOutputDir)
        
        model_layout.addWidget(output_dir_label, 6, 0)
        model_layout.addWidget(self.output_dir_edit, 6, 1)
        model_layout.addWidget(output_dir_browse, 6, 2)
        
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
        
        # 체크포인트 불러오기 기능 추가
        checkpoint_layout = QHBoxLayout()
        self.checkpoint_label = QLabel("체크포인트:")
        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setReadOnly(True)
        self.checkpoint_browse = QPushButton("찾아보기...")
        self.checkpoint_browse.clicked.connect(self.browseCheckpoint)
        
        checkpoint_layout.addWidget(self.checkpoint_label)
        checkpoint_layout.addWidget(self.checkpoint_path)
        checkpoint_layout.addWidget(self.checkpoint_browse)
        
        summary_layout.addLayout(checkpoint_layout, 3, 0)
        
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
        
    def setupAnalysisTab(self):
        """데이터 분석 탭 설정"""
        layout = QVBoxLayout(self.analysis_tab)
        
        # 데이터 분석기 추가
        self.data_analyzer = DatasetAnalyzer()
        
        layout.addWidget(self.data_analyzer)
        
        # 변경 버튼
        btn_layout = QHBoxLayout()
        
        self.analyze_data_btn = QPushButton("현재 데이터셋 분석")
        self.analyze_data_btn.clicked.connect(self.analyzeCurrentDataset)
        self.analyze_data_btn.setEnabled(False)
        
        btn_layout.addWidget(self.analyze_data_btn)
        
        layout.addLayout(btn_layout)
        
    def setupCheckpointTab(self):
        """체크포인트 관리 탭 설정"""
        layout = QVBoxLayout(self.checkpoint_tab)
        
        # 체크포인트 관리자 추가
        self.checkpoint_manager = CheckpointManager(self.output_dir_edit.text() or "results")
        self.checkpoint_manager.load_checkpoint_signal.connect(self.loadCheckpointFromManager)
        
        layout.addWidget(self.checkpoint_manager)
    
    def analyzeCurrentDataset(self):
        """현재 로드된 데이터셋 분석"""
        if hasattr(self, 'classifier') and self.classifier and hasattr(self.classifier, 'train_loader'):
            # 데이터 로더 설정
            self.data_analyzer.set_data_loader(self.classifier.train_loader)
            
            # 클래스 분포 분석 실행
            self.data_analyzer.analyze_class_distribution()
            
            # 탭 이동
            self.tabs.setCurrentWidget(self.analysis_tab)
            
            # 상태 메시지 업데이트
            self.status_bar.showMessage("데이터셋 분석 완료")
        else:
            QMessageBox.warning(self, "경고", "먼저 데이터를 로드해주세요")
        # 체크포인트 경로 설정
        self.checkpoint_path.setText(checkpoint_path)
        
        # 학습 탭으로 이동
        self.tabs.setCurrentIndex(1)
        
        # 학습 재개 버튼으로 변경
        self.train_btn.setText("학습 재개")
        
        # 체크포인트 정보 불러오기
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            epoch = checkpoint.get('epoch', 0) + 1  # 0-based에서 1-based로 변환
            self.params_summary_label.setText(f"파라미터: 에폭 {epoch}\/{self.epochs_spin.value()} 에서 체크포인트 불러옴")
            
            # 체크포인트 학습 정보가 있으면 출력
            if 'history' in checkpoint and len(checkpoint['history'].get('train_loss', [])) > 0:
                val_acc = checkpoint['history']['val_acc'][-1] * 100 if checkpoint['history']['val_acc'] else 0
                self.status_bar.showMessage(f"체크포인트 불러오기 성공: 에폭 {epoch}, 정확도: {val_acc:.2f}%")
                
                # 학습 그래프 업데이트
                epochs = list(range(1, len(checkpoint['history']['train_loss'])+1))
                self.train_loss_curve.setData(epochs, checkpoint['history']['train_loss'])
                self.val_loss_curve.setData(epochs, checkpoint['history']['val_loss'])
                self.train_acc_curve.setData(epochs, checkpoint['history']['train_acc'])
                self.val_acc_curve.setData(epochs, checkpoint['history']['val_acc'])
            else:
                self.status_bar.showMessage(f"체크포인트 불러오기 성공: 에폭 {epoch}")
        except Exception as e:
            QMessageBox.warning(self, "경고", f"체크포인트 정보 불러오기 오류: {str(e)}")
        
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
        
    def browseCheckpoint(self):
        """체크포인트 파일 선택 대화상자 표시"""
        file, _ = QFileDialog.getOpenFileName(self, "체크포인트 파일 선택", "", "PyTorch 체크포인트 (*.pth)")
        if file:
            self.checkpoint_path.setText(file)
            # 파일이 선택되면 학습 재개 가능하도록 설정
            self.train_btn.setText("학습 재개")
            
            # 체크포인트 정보 불러오기
            try:
                checkpoint = torch.load(file, map_location=torch.device('cpu'))
                epoch = checkpoint.get('epoch', 0) + 1  # 0-based에서 1-based로 변환
                self.params_summary_label.setText(f"파라미터: 에폭 {epoch}\/{self.epochs_spin.value()} 에서 체크포인트 불러옴")
                
                # 체크포인트 학습 정보가 있으면 출력
                if 'history' in checkpoint and len(checkpoint['history'].get('train_loss', [])) > 0:
                    val_acc = checkpoint['history']['val_acc'][-1] * 100 if checkpoint['history']['val_acc'] else 0
                    self.status_bar.showMessage(f"체크포인트 불러오기 성공: 에폭 {epoch}, 정확도: {val_acc:.2f}%")
                else:
                    self.status_bar.showMessage(f"체크포인트 불러오기 성공: 에폭 {epoch}")
            except Exception as e:
                QMessageBox.warning(self, "경고", f"체크포인트 정보 불러오기 오류: {str(e)}")
    
    def browseDataDir(self):
        """데이터 디렉토리 선택 대화상자 표시"""
        folder = QFileDialog.getExistingDirectory(self, "데이터 디렉토리 선택")
        if folder:
            self.data_dir_edit.setText(folder)
    
    def browseOutputDir(self):
        """결과 저장 디렉토리 선택 대화상자 표시"""
        folder = QFileDialog.getExistingDirectory(self, "결과 저장 디렉토리 선택")
        if folder:
            self.output_dir_edit.setText(folder)
            
            # 체크포인트 관리자 경로 업데이트
            if hasattr(self, 'checkpoint_manager'):
                self.checkpoint_manager.set_output_dir(folder)
    
    def browsePredictModel(self):
        """모델 파일 선택 대화상자 표시"""
        file, _ = QFileDialog.getOpenFileName(self, "모델 파일 선택", "", "PyTorch 모델 (*.pth)")
        if file:
            self.predict_model_path.setText(file)
            self.predict_btn.setEnabled(True)
    
    def browsePredictImage(self):
        """예측할 이미지 파일 선택 대화상자 표시"""
        file, _ = QFileDialog.getOpenFileName(self, "이미지 파일 선택", "", "이미지 파일 (*.jpg *.jpeg *.png)")
        if file:
            self.predict_image_path.setText(file)
            
            # 이미지 표시
            pixmap = QPixmap(file)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_display.setPixmap(pixmap)
    def loadData(self):
        """데이터 로드 및 미리보기"""
        data_dir = self.data_dir_edit.text()
        if not data_dir:
            QMessageBox.warning(self, "경고", "데이터 디렉토리를 선택해주세요")
            return
        
        try:
            # 분류기 생성
            self.classifier = ImageClassifier(
                data_dir=data_dir,
                output_dir=self.output_dir_edit.text(),
                batch_size=self.batch_size_spin.value(),
                num_epochs=self.epochs_spin.value(),
                lr=self.lr_spin.value(),
                model_type=self.model_combo.currentText().lower()
            )
            
            # 데이터 로드
            self.classifier.load_data(val_split=self.val_split_spin.value())
            
            # 샘플 이미지 표시
            self.displaySampleImages()
            
            # 모델 초기화 버튼 활성화
            self.init_model_btn.setEnabled(True)
            
            # 데이터 분석 버튼 활성화
            self.analyze_data_btn.setEnabled(True)
            
            # 상태 메시지 업데이트
            self.status_bar.showMessage(f"데이터 로드 완료: {len(self.classifier.train_dataset)} 학습 이미지, {len(self.classifier.val_dataset)} 검증 이미지")
            
            # 데이터 요약 업데이트
            self.data_summary_label.setText(f"데이터: {len(self.classifier.train_dataset) + len(self.classifier.val_dataset)} 이미지, {len(self.classifier.class_names)} 클래스")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"데이터 로드 중 오류 발생: {str(e)}")
    
    def displaySampleImages(self):
        """샘플 이미지 표시"""
        # 기존 콘텐츠 정리
        for i in reversed(range(self.preview_grid.count())): 
            self.preview_grid.itemAt(i).widget().setParent(None)
        
        # 각 클래스별 샘플 이미지 표시
        for class_idx, class_name in enumerate(self.classifier.class_names):
            # 클래스 레이블 추가
            label = QLabel(class_name)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-weight: bold;")
            self.preview_grid.addWidget(label, class_idx, 0)
            
            # 해당 클래스의 이미지 가져오기
            class_images = [img_path for img_path, label in self.classifier.train_dataset.samples 
                           if label == self.classifier.train_dataset.class_to_idx[class_name]]
            
            # 최대 5개 이미지 표시
            num_samples = min(5, len(class_images))
            for j in range(num_samples):
                img_path = class_images[j]
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    img_label = QLabel()
                    img_label.setPixmap(pixmap)
                    img_label.setAlignment(Qt.AlignCenter)
                    self.preview_grid.addWidget(img_label, class_idx, j+1)
        
        # 미리보기 스크롤 영역 표시
        self.preview_scroll.setVisible(True)
        self.preview_label.setVisible(False)
    
    def initializeModel(self):
        """모델 초기화"""
        try:
            # 모델 생성
            self.classifier.create_model()
            
            # 모델 요약 업데이트
            model_info = f"모델: {self.model_combo.currentText()}"
            self.model_summary_label.setText(model_info)
            
            # 파라미터 요약 업데이트
            params_info = f"파라미터: 배치 크기={self.batch_size_spin.value()}, 에폭={self.epochs_spin.value()}, 학습률={self.lr_spin.value()}"  
            self.params_summary_label.setText(params_info)
            
            # 학습 버튼 활성화
            self.train_btn.setEnabled(True)
            self.train_btn.setText("학습 시작")  # 체크포인트 선택 후 "학습 재개"로 변경될 수 있음
            
            # 상태 메시지 업데이트
            self.status_bar.showMessage(f"{self.model_combo.currentText()} 모델 초기화 완료")
            
            # 체크포인트 경로 초기화 (새 모델을 시작하는 경우)
            if self.checkpoint_path.text():
                response = QMessageBox.question(
                    self, "체크포인트 초기화",
                    "새 모델을 초기화했습니다. 체크포인트 선택을 초기화할까요?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if response == QMessageBox.Yes:
                    self.checkpoint_path.setText("")
                    self.train_btn.setText("학습 시작")
                
        except Exception as e:
            QMessageBox.critical(self, "오류", f"모델 초기화 중 오류 발생: {str(e)}")
    
    def startTraining(self):
        """모델 학습 시작"""
        # 체크포인트 파일이 있는지 확인
        start_epoch = 0
        checkpoint_path = self.checkpoint_path.text()
        
        # 체크포인트가 있을 경우 체크포인트에서 모델 불러오기
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # 체크포인트 불러오기
                checkpoint = torch.load(checkpoint_path, map_location=self.classifier.device)
                
                # 모델 가중치 불러오기
                self.classifier.model.load_state_dict(checkpoint['model_state_dict'])
                
                # 옵티마이저 가중치 불러오기
                self.classifier.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # 스케줄러 가중치 불러오기
                if 'scheduler_state_dict' in checkpoint and hasattr(self.classifier, 'scheduler'):
                    self.classifier.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # 학습 히스토리 불러오기
                if 'history' in checkpoint:
                    self.classifier.history = checkpoint['history']
                
                # 시작 에폭 가져오기
                start_epoch = checkpoint.get('epoch', 0) + 1
                
                self.status_bar.showMessage(f"체크포인트에서 불러왔습니다. 에폭 {start_epoch}부터 학습을 재개합니다.")
            except Exception as e:
                QMessageBox.warning(self, "경고", f"체크포인트 불러오기 오류: {str(e)}")
                return
        
        # 체크포인트 설정 가져오기
        checkpoint_settings = None
        if hasattr(self, 'checkpoint_manager'):
            checkpoint_settings = self.checkpoint_manager.get_checkpoint_settings()
        
        # 학습 쓰레드 생성
        self.training_thread = TrainingThread(self.classifier, self.epochs_spin.value(), start_epoch, checkpoint_settings)
        
        # 시그널 연결
        self.training_thread.update_progress.connect(self.updateEpochProgress)
        self.training_thread.update_batch.connect(self.updateBatchProgress)
        self.training_thread.finished.connect(self.trainingFinished)
        self.training_thread.error.connect(self.handle_training_error)
        self.training_thread.error.connect(self.handle_training_error)
        
        # 버튼 상태 업데이트
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.init_model_btn.setEnabled(False)
        
        # 뉴런 시각화 시작
        self.neuron_viz.startVisualization()
        
        # 학습 시작 시간 기록
        self.training_start_time = time.time()
        
        # 학습 쓰레드 시작
        self.training_thread.start()
        
        # 상태 메시지 업데이트
        if start_epoch > 0:
            self.status_bar.showMessage(f"에폭 {start_epoch}부터 학습 재개 중...")
        else:
            self.status_bar.showMessage("모델 학습 시작...")
    
    def stopTraining(self):
        """모델 학습 중단"""
        if self.training_thread and self.training_thread.isRunning():
            # 학습 쓰레드 중단
            self.training_thread.stop()
            
            # 버튼 상태 업데이트
            self.stop_btn.setEnabled(False)
            self.train_btn.setEnabled(True)
            self.init_model_btn.setEnabled(True)
            
            # 뉴런 시각화 중단
            self.neuron_viz.stopVisualization()
            
            # 상태 메시지 업데이트
            self.status_bar.showMessage("모델 학습 중단됨")
    
    def updateEpochProgress(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """에폭 진행 상황 업데이트"""
        # 에폭 레이블 및 프로그레스바 업데이트
        self.epoch_label.setText(f"에폭: {epoch}/{self.epochs_spin.value()}")
        self.epoch_progress.setValue(int(epoch / self.epochs_spin.value() * 100))
        
        # 손실 그래프 업데이트
        epochs = list(range(1, epoch+1))
        self.train_loss_curve.setData(epochs, self.classifier.history['train_loss'])
        self.val_loss_curve.setData(epochs, self.classifier.history['val_loss'])
        
        # 정확도 그래프 업데이트
        self.train_acc_curve.setData(epochs, self.classifier.history['train_acc'])
        self.val_acc_curve.setData(epochs, self.classifier.history['val_acc'])
    
    def updateBatchProgress(self, epoch, batch, progress):
        """배치 진행 상황 업데이트"""
        total_batches = len(self.classifier.train_loader)
        self.batch_label.setText(f"배치: {batch+1}/{total_batches}")
        self.batch_progress.setValue(int(progress))
    
    def handle_training_error(self, error_message):
        """학습 중 오류 처리"""
        QMessageBox.critical(self, "학습 오류", f"학습 중 오류가 발생했습니다: {error_message}")
        
        # 뉴런 시각화 중지
        self.neuron_viz.stopVisualization()
        
        # 버튼 상태 업데이트
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.init_model_btn.setEnabled(True)
        
        # 상태 메시지 업데이트
        self.status_bar.showMessage(f"학습 오류: {error_message}")
    
    def trainingFinished(self, history):
        """학습 완료 처리"""
        # 뉴런 시각화 중지
        self.neuron_viz.stopVisualization()
        
        # 버튼 상태 업데이트
        self.train_btn.setEnabled(True)
        self.train_btn.setText("학습 시작")
        self.stop_btn.setEnabled(False)
        self.init_model_btn.setEnabled(True)
        
        # 체크포인트 경로 초기화
        self.checkpoint_path.setText("")
        
        # 학습 시간 계산
        training_time = time.time() - self.training_start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}시간 {int(minutes)}분 {int(seconds)}초"
        
        # 결과 탭 정보 업데이트
        self.result_model_label.setText(f"모델: {self.model_combo.currentText()}")
        self.result_data_label.setText(f"데이터: {len(self.classifier.train_dataset) + len(self.classifier.val_dataset)} 이미지, {len(self.classifier.class_names)} 클래스")
        self.result_time_label.setText(f"학습 시간: {time_str}")
        
        if history['val_acc']:
            final_acc = history['val_acc'][-1] * 100
            self.result_accuracy_label.setText(f"최종 정확도: {final_acc:.2f}%")
        
        # 모델 평가 및 결과 시각화
        try:
            # 모델 저장
            model_path = self.classifier.save_model()
            
            # 마지막 체크포인트 경로 - 일관된 형식 사용
            last_epoch = len(self.classifier.history['train_loss']) - 1
            last_checkpoint_path = os.path.join(self.classifier.output_dir, f'checkpoint_epoch_{last_epoch}.pth')
            
            self.classifier.evaluate()
            self.loadResultVisualizations()
            
            # 예측 탭의 모델 경로 설정
            self.predict_model_path.setText(model_path)
            self.predict_btn.setEnabled(True)
            
            # 결과 저장 버튼 활성화
            self.save_results_btn.setEnabled(True)
            self.open_results_dir_btn.setEnabled(True)
            
            self.status_bar.showMessage(f"모델 학습 완료. 정확도: {final_acc:.2f}%, 모델 저장됨: {model_path}")
        except Exception as e:
            QMessageBox.warning(self, "경고", f"결과 시각화 중 오류 발생: {str(e)}")
            self.status_bar.showMessage("모델 학습 완료, 결과 시각화 실패")
    
    def loadResultVisualizations(self):
        """결과 시각화 불러오기"""
        # 혼동 행렬 표시
        cm_path = os.path.join(self.classifier.output_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            self.conf_matrix_fig.clf()
            ax = self.conf_matrix_fig.add_subplot(111)
            img = plt.imread(cm_path)
            ax.imshow(img)
            ax.axis('off')
            self.conf_matrix_canvas.draw()
        
        # ROC 커브 표시
        roc_path = os.path.join(self.classifier.output_dir, 'roc_curves.png')
        if os.path.exists(roc_path):
            self.roc_fig.clf()
            ax = self.roc_fig.add_subplot(111)
            img = plt.imread(roc_path)
            ax.imshow(img)
            ax.axis('off')
            self.roc_canvas.draw()
        
        # t-SNE 시각화 표시
        tsne_path = os.path.join(self.classifier.output_dir, 'tsne_visualization.png')
        if os.path.exists(tsne_path):
            self.tsne_fig.clf()
            ax = self.tsne_fig.add_subplot(111)
            img = plt.imread(tsne_path)
            ax.imshow(img)
            ax.axis('off')
            self.tsne_canvas.draw()
        
        # CAM 시각화 표시
        cam_path = os.path.join(self.classifier.output_dir, 'class_activation_maps.png')
        if os.path.exists(cam_path):
            self.cam_fig.clf()
            ax = self.cam_fig.add_subplot(111)
            img = plt.imread(cam_path)
            ax.imshow(img)
            ax.axis('off')
            self.cam_canvas.draw()
    
    def saveResults(self):
        """결과 저장"""
        try:
            # 모델 저장
            model_path = self.classifier.save_model()
            
            # 저장 완료 메시지
            QMessageBox.information(self, "정보", f"모델 및 결과가 저장되었습니다.\n저장 위치: {self.classifier.output_dir}")
            
            # 상태 메시지 업데이트
            self.status_bar.showMessage(f"결과 저장 완료: {self.classifier.output_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"결과 저장 중 오류 발생: {str(e)}")
    
    def openResultsDir(self):
        """결과 폴더 열기"""
        import subprocess
        import platform
        
        # 운영체제별 폴더 열기 명령어
        output_dir = os.path.abspath(self.classifier.output_dir)
        
        try:
            if platform.system() == "Windows":
                os.startfile(output_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
                
            self.status_bar.showMessage(f"결과 폴더 열기: {output_dir}")
        except Exception as e:
            QMessageBox.warning(self, "경고", f"폴더를 열 수 없습니다: {str(e)}")
    
    def runPrediction(self):
        """이미지 예측 실행"""
        model_path = self.predict_model_path.text()
        image_path = self.predict_image_path.text()
        
        if not model_path or not image_path:
            QMessageBox.warning(self, "경고", "모델 파일과 이미지 파일을 모두 선택해주세요")
            return
        
        try:
            # 분류기 생성
            self.predictor = ImageClassifier(
                data_dir="",  # 예측 모드에서는 불필요
                output_dir="predictions"
            )
            
            # 모델 로드
            self.predictor.load_model(filename=model_path)
            
            # 예측 실행
            result = self.predictor.predict(image_path)
            
            # 예측 결과 표시
            self.prediction_label.setText(f"예측 클래스: {result['class']}")
            self.confidence_label.setText(f"신뢰도: {result['probability']:.2%}")
            
            # 신뢰도 차트 업데이트
            self.updateConfidenceChart(result['all_probabilities'])
            
            # 상태 메시지 업데이트
            self.status_bar.showMessage(f"예측 완료: {result['class']} ({result['probability']:.2%})")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"예측 중 오류 발생: {str(e)}")
    
    def updateConfidenceChart(self, probabilities):
        """신뢰도 차트 업데이트"""
        self.confidence_chart.clear()
        
        # X축 레이블 설정
        class_names = list(probabilities.keys())
        xDict = dict(enumerate(class_names))
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([xDict.items()])
        
        # 바 차트 데이터 준비
        x = np.arange(len(class_names))
        y = [probabilities[name] for name in class_names]
        
        # 바 차트 그리기
        bargraph = pg.BarGraphItem(x=x, height=y, width=0.6, brush='b')
        self.confidence_chart.addItem(bargraph)
        
        # Y축 범위 설정
        self.confidence_chart.setYRange(0, 1)
    
    def openDataFolder(self):
        """데이터 폴더 열기 메뉴 처리"""
        self.browseDataDir()
    
    def saveModel(self):
        """모델 저장 메뉴 처리"""
        if hasattr(self, 'classifier') and self.classifier and hasattr(self.classifier, 'model') and self.classifier.model:
            self.saveResults()
        else:
            QMessageBox.warning(self, "경고", "저장할 모델이 없습니다. 먼저 모델을 학습해주세요.")
    
    def loadCheckpoint(self):
        """체크포인트를 불러와 학습 재개"""
        file, _ = QFileDialog.getOpenFileName(self, "체크포인트 파일 선택", "", "PyTorch 체크포인트 (*.pth)")
        if file:
            try:
                # 체크포인트 파일 로드
                checkpoint = torch.load(file, map_location=torch.device('cpu'))
                
                # 현재 데이터가 로드되어 있는지 확인
                if not hasattr(self, 'classifier') or not self.classifier or not hasattr(self.classifier, 'class_names'):
                    QMessageBox.warning(self, "경고", "먼저 데이터를 로드해주세요.")
                    return
                
                # 기존 모델이 있는지 확인
                if not hasattr(self.classifier, 'model') or self.classifier.model is None:
                    # 모델 초기화
                    self.classifier.create_model()
                
                # 학습 과정 로드
                self.classifier.model.load_state_dict(checkpoint['model_state_dict'])
                self.classifier.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.classifier.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # 학습 히스토리 불러오기
                if 'history' in checkpoint:
                    self.classifier.history = checkpoint['history']
                    
                    # 학습 그래프 업데이트
                    if len(self.classifier.history['train_loss']) > 0:
                        epochs = list(range(1, len(self.classifier.history['train_loss'])+1))
                        self.train_loss_curve.setData(epochs, self.classifier.history['train_loss'])
                        self.val_loss_curve.setData(epochs, self.classifier.history['val_loss'])
                        self.train_acc_curve.setData(epochs, self.classifier.history['train_acc'])
                        self.val_acc_curve.setData(epochs, self.classifier.history['val_acc'])
                
                # 에폭 정보 추출
                start_epoch = checkpoint.get('epoch', -1) + 1  # 다음 에폭부터 시작
                
                # UI 상태 업데이트
                self.model_summary_label.setText(f"모델: {self.classifier.model_type}")
                self.data_summary_label.setText(f"데이터: 학습 {len(self.classifier.train_dataset)}, 검증 {len(self.classifier.val_dataset)}")
                self.params_summary_label.setText(f"파라미터: 에폭={start_epoch}/{self.epochs_spin.value()}")
                
                # 학습 버튼 활성화
                self.train_btn.setEnabled(True)
                
                # 에폭 프로그레스 업데이트
                self.epoch_label.setText(f"에폭: {start_epoch}/{self.epochs_spin.value()}")
                self.epoch_progress.setValue(int(start_epoch / self.epochs_spin.value() * 100))
                
                # 상태 메시지 업데이트
                self.status_bar.showMessage(f"체크포인트 불러오기 성공: {file}, 에폭 {start_epoch}부터 다시 시작")
                
                # 학습 계속할지 여부 물어보기
                reply = QMessageBox.question(self, "학습 재개", 
                    f"체크포인트가 성공적으로 불러왔습니다.\n\n에폭 {start_epoch}부터 학습을 계속하시겠습니까?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    # 학습 재개 시작
                    self.resumeTraining(start_epoch)
                
            except Exception as e:
                QMessageBox.critical(self, "오류", f"체크포인트 불러오기 실패: {str(e)}")
    
    def resumeTraining(self, start_epoch):
        """체크포인트부터 학습 재개"""
        # 체크포인트 설정 가져오기
        checkpoint_settings = None
        if hasattr(self, 'checkpoint_manager'):
            checkpoint_settings = self.checkpoint_manager.get_checkpoint_settings()
            
        # 학습 쓰레드 생성 (시작 에폭 지정)
        self.training_thread = TrainingThread(self.classifier, self.epochs_spin.value(), start_epoch, checkpoint_settings)
        
        # 시그널 연결
        self.training_thread.update_progress.connect(self.updateEpochProgress)
        self.training_thread.update_batch.connect(self.updateBatchProgress)
        self.training_thread.finished.connect(self.trainingFinished)
        self.training_thread.error.connect(self.handle_training_error)
        
        # 버튼 상태 업데이트
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.init_model_btn.setEnabled(False)
        
        # 뉴런 시각화 시작
        self.neuron_viz.startVisualization()
        
        # 학습 시작 시간 기록
        self.training_start_time = time.time()
        
        # 학습 쓰레드 시작
        self.training_thread.start()
        
        # 상태 메시지 업데이트
        self.status_bar.showMessage(f"모델 학습 재개... 에폭 {start_epoch}부터")
    
    def loadModel(self):
        """모델 불러오기 메뉴 처리"""
        file, _ = QFileDialog.getOpenFileName(self, "모델 파일 선택", "", "PyTorch 모델 (*.pth)")
        if file:
            try:
                # 기본 분류기 생성
                self.classifier = ImageClassifier(
                    data_dir="",  # 필요한 경우 데이터 디렉토리 설정
                    output_dir=self.output_dir_edit.text() or "results"
                )
                
                # 모델 불러오기
                self.classifier.load_model(filename=file)
                
                # UI 상태 업데이트
                self.model_summary_label.setText(f"모델: {self.classifier.model_type}")
                self.train_btn.setEnabled(False)  # 학습 버튼 비활성화 (이미 학습된 모델)
                self.save_results_btn.setEnabled(True)  # 결과 저장 버튼 활성화
                
                # 상태 메시지 업데이트
                self.status_bar.showMessage(f"모델을 불러왔습니다: {file}")
                
                # 평가할 데이터가 있으면 평가 버튼 활성화 (필요에 따라 구현)
                
            except Exception as e:
                QMessageBox.critical(self, "오류", f"모델 불러오기 실패: {str(e)}")
    def setupAnalysisTab(self):
        """데이터 분석 탭 설정"""
        layout = QVBoxLayout(self.analysis_tab)
        
        # 데이터 분석기 추가
        self.data_analyzer = DatasetAnalyzer()
        
        layout.addWidget(self.data_analyzer)
        
        # 분석 버튼
        btn_layout = QHBoxLayout()
        
        self.analyze_data_btn = QPushButton("현재 데이터셋 분석")
        self.analyze_data_btn.clicked.connect(self.analyzeCurrentDataset)
        self.analyze_data_btn.setEnabled(False)
        
        btn_layout.addWidget(self.analyze_data_btn)
        
        layout.addLayout(btn_layout)
        
    def analyzeCurrentDataset(self):
        """현재 데이터셋 분석"""
        if not hasattr(self, 'classifier') or not self.classifier:
            QMessageBox.warning(self, "경고", "먼저 데이터를 로드해주세요.")
            return
        
        if not hasattr(self.classifier, 'train_loader') or not self.classifier.train_loader:
            QMessageBox.warning(self, "경고", "데이터 로더가 초기화되지 않았습니다.")
            return
        
        # 데이터 분석기에 데이터 로더 전달
        self.data_analyzer.set_data_loader(self.classifier.train_loader)
        
        # 분석 시작
        self.tabs.setCurrentWidget(self.analysis_tab)
        self.data_analyzer.analyze_class_distribution()
        
        # 상태 메세지 업데이트
        self.status_bar.showMessage("데이터셋 분석 완료")
        
    def showAbout(self):
        """프로그램 정보 표시"""
        QMessageBox.about(self, "프로그램 정보",
            "<h3>딥러닝 이미지 분류 시스템</h3>"
            "<p>ResNet 기반 이미지 분류 모델의 학습, 평가, 시각화를 위한 UI 시스템</p>"
            "<p>버전: 1.0</p>"
            "<p>레퍼런스:</p>"
            "<ul>"
            "<li>PyTorch: <a href='https://pytorch.org'>https://pytorch.org</a></li>"
            "<li>ResNet: He, K., et al. (2016). Deep Residual Learning for Image Recognition</li>"
            "</ul>")
