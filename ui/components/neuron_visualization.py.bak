#!/usr/bin/env python3
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import random

class NeuronVisualizationWidget(QWidget):
    """신경망 뉴런 시각화를 위한 위젯"""
    
    def __init__(self):
        super().__init__()
        self.initUI()
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateNeurons)
        self.active_neurons = []
        self.layer_sizes = [3, 64, 128, 256, 512, 1000, 5]  # 예시 네트워크 구조
        
    def initUI(self):
        self.layout = QVBoxLayout(self)
        
        # 설명 라벨
        title = QLabel("신경망 뉴런 활성화 시각화")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Bold))
        self.layout.addWidget(title)
        
        # PyQtGraph 위젯
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground('w')
        self.graphWidget.setTitle("뉴런 활성화 패턴", color="k", size="12pt")
        self.graphWidget.setLabel('left', "레이어", color="k")
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setXRange(0, 1000)
        self.graphWidget.setYRange(0, len(self.layer_sizes))
        self.layout.addWidget(self.graphWidget)
        
        # 제어 버튼
        self.controlLayout = QHBoxLayout()
        
        self.startButton = QPushButton("시각화 시작")
        self.startButton.clicked.connect(self.startVisualization)
        
        self.stopButton = QPushButton("시각화 중지")
        self.stopButton.clicked.connect(self.stopVisualization)
        self.stopButton.setEnabled(False)
        
        self.controlLayout.addWidget(self.startButton)
        self.controlLayout.addWidget(self.stopButton)
        
        self.layout.addLayout(self.controlLayout)
        
    def startVisualization(self):
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.timer.start(100)  # 100ms 마다 업데이트
        
    def stopVisualization(self):
        self.stopButton.setEnabled(False)
        self.startButton.setEnabled(True)
        self.timer.stop()
        
    def updateNeurons(self):
        self.graphWidget.clear()
        
        # 기본 네트워크 구조 그리기
        cumulative_size = 0
        y_positions = []
        
        for i, size in enumerate(self.layer_sizes):
            y_pos = i + 0.5
            y_positions.append(y_pos)
            
            # 레이어 라벨
            if i == 0:
                layer_name = "입력 레이어"
            elif i == len(self.layer_sizes) - 1:
                layer_name = "출력 레이어"
            else:
                layer_name = f"은닉 레이어 {i}"
                
            text = pg.TextItem(layer_name, color=(0, 0, 0))
            text.setPos(0, y_pos)
            self.graphWidget.addItem(text)
            
            # 레이어간 연결 그리기
            if i > 0:
                for _ in range(min(40, size)):  # 최대 40개 연결만 표시
                    x_positions = np.linspace(10, 990, 40)
                    pen = pg.mkPen(color=(200, 200, 200), width=1)
                    conn_line = pg.PlotCurveItem(x=x_positions, 
                                              y=self.generateConnectionCurve(y_positions[i-1], y_positions[i], x_positions),
                                              pen=pen)
                    self.graphWidget.addItem(conn_line)
            
            # 각 레이어에 뉴런 그리기
            for j in range(min(40, size)):  # 최대 40개 뉴런만 표시
                x_pos = 10 + (990 - 10) * (j / min(40, size))
                
                # 랜덤으로 뉴런 활성화 결정
                is_active = random.random() > 0.7
                
                if is_active:
                    brush = pg.mkBrush(color=(255, 0, 0, 200))
                    size_factor = 15
                else:
                    brush = pg.mkBrush(color=(100, 100, 255, 100))
                    size_factor = 8
                    
                neuron = pg.ScatterPlotItem()
                neuron.addPoints([{'pos': (x_pos, y_pos), 'brush': brush, 'size': size_factor}])
                self.graphWidget.addItem(neuron)
        
        # Y축 범위 조정
        self.graphWidget.setYRange(0, len(self.layer_sizes))
        
    def generateConnectionCurve(self, y1, y2, x_range):
        # 두 레이어 사이의 연결 곡선 생성
        y = np.zeros_like(x_range)
        for i in range(len(x_range)):
            t = i / (len(x_range) - 1)  # 0에서 1 사이의 비율
            # 베지어 곡선 형태로 부드러운 곡선 생성
            y[i] = (1-t) * y1 + t * y2
        return y
