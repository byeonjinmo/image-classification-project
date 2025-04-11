#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import random
import time


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


class TrainingThread(QThread):
    """백그라운드에서 모델 학습을 수행하는 쓰레드"""
    
    # 시그널 정의
    update_progress = pyqtSignal(int, float, float, float, float)
    update_batch = pyqtSignal(int, int, float)
    finished = pyqtSignal(dict)
    
    def __init__(self, classifier, num_epochs):
        super().__init__()
        self.classifier = classifier
        self.num_epochs = num_epochs
        self.running = True
        
    def run(self):
        # 원래 train 메서드의 내용을 바탕으로 각 에폭과 배치에서 신호를 보냄
        for epoch in range(self.num_epochs):
            if not self.running:
                break
                
            # 학습 단계
            self.classifier.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i, (inputs, labels) in enumerate(self.classifier.train_loader):
                if not self.running:
                    break
                    
                inputs, labels = inputs.to(self.classifier.device), labels.to(self.classifier.device)
                
                # 그래디언트 초기화
                self.classifier.optimizer.zero_grad()
                
                # 순전파
                outputs = self.classifier.model(inputs)
                loss = self.classifier.criterion(outputs, labels)
                
                # 역전파 및 최적화
                loss.backward()
                self.classifier.optimizer.step()
                
                # 통계 추적
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # 배치 진행 상황 업데이트
                current_loss = train_loss / train_total
                current_acc = train_correct / train_total
                batch_progress = (i + 1) / len(self.classifier.train_loader) * 100
                self.update_batch.emit(epoch, i, batch_progress)
                
                # 뉴런 시각화용으로 약간의 딜레이
                time.sleep(0.01)
            
            # 학습 손실 및 정확도 계산
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # 검증 단계
            self.classifier.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in self.classifier.val_loader:
                    if not self.running:
                        break
                        
                    inputs, labels = inputs.to(self.classifier.device), labels.to(self.classifier.device)
                    
                    # 순전파
                    outputs = self.classifier.model(inputs)
                    loss = self.classifier.criterion(outputs, labels)
                    
                    # 통계 추적
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # 검증 손실 및 정확도 계산
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # 학습률 스케줄러 업데이트
            self.classifier.scheduler.step(val_loss)
            
            # 에폭 진행 상황 업데이트
            self.classifier.history['train_loss'].append(train_loss)
            self.classifier.history['val_loss'].append(val_loss)
            self.classifier.history['train_acc'].append(train_acc)
            self.classifier.history['val_acc'].append(val_acc)
            
            # 에폭 업데이트 시그널 발생
            progress = (epoch + 1) / self.num_epochs * 100
            self.update_progress.emit(epoch + 1, train_loss, train_acc, val_loss, val_acc)
            
            # 체크포인트 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.classifier.model.state_dict(),
                'optimizer_state_dict': self.classifier.optimizer.state_dict(),
                'scheduler_state_dict': self.classifier.scheduler.state_dict(),
                'history': self.classifier.history
            }, os.path.join(self.classifier.output_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        # 학습 완료 신호 보내기
        if self.running:
            self.finished.emit(self.classifier.history)
    
    def stop(self):
        self.running = False
