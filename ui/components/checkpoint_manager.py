#!/usr/bin/env python3
import os
import torch
import glob
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QListWidget, QGroupBox, QMessageBox, QFileDialog,
                             QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
                             QFormLayout, QLineEdit, QComboBox, QCheckBox, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

class CheckpointManager(QWidget):
    """체크포인트 관리 위젯"""
    
    # 체크포인트 로드 시그널
    load_checkpoint_signal = pyqtSignal(str)
    
    def __init__(self, output_dir="results"):
        super().__init__()
        self.output_dir = output_dir
        self.checkpoint_files = []
        self.checkpoint_info = {}
        self.initUI()
        
    def initUI(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 제목
        title = QLabel("체크포인트 관리")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 체크포인트 목록 테이블
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["파일명", "에폭", "학습 손실", "검증 손실", "검증 정확도"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        # 버튼 영역
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.refresh_checkpoints)
        
        self.load_btn = QPushButton("체크포인트 불러오기")
        self.load_btn.clicked.connect(self.load_selected_checkpoint)
        
        self.delete_btn = QPushButton("삭제")
        self.delete_btn.clicked.connect(self.delete_selected_checkpoint)
        
        self.import_btn = QPushButton("외부 체크포인트 가져오기")
        self.import_btn.clicked.connect(self.import_checkpoint)
        
        self.export_btn = QPushButton("선택한 체크포인트 내보내기")
        self.export_btn.clicked.connect(self.export_checkpoint)
        
        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addWidget(self.import_btn)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
        
        # 설정 영역
        settings_group = QGroupBox("체크포인트 설정")
        settings_layout = QFormLayout(settings_group)
        
        self.auto_save_cb = QCheckBox("자동 저장")
        self.auto_save_cb.setChecked(True)
        
        self.save_freq_spin = QSpinBox()
        self.save_freq_spin.setRange(1, 10)
        self.save_freq_spin.setValue(1)
        self.save_freq_spin.setPrefix("매 ")
        self.save_freq_spin.setSuffix(" 에폭마다")
        
        settings_layout.addRow("체크포인트 자동 저장:", self.auto_save_cb)
        settings_layout.addRow("저장 주기:", self.save_freq_spin)
        
        layout.addWidget(settings_group)
        
        # 초기 체크포인트 로드
        self.refresh_checkpoints()
    
    def set_output_dir(self, output_dir):
        """결과 디렉토리 설정"""
        self.output_dir = output_dir
        self.refresh_checkpoints()
    
    def refresh_checkpoints(self):
        """체크포인트 목록 새로고침"""
        # 기존 정보 초기화
        self.checkpoint_files = []
        self.checkpoint_info = {}
        
        # 체크포인트 파일 검색
        checkpoint_pattern = os.path.join(self.output_dir, "checkpoint_epoch_*.pth")
        self.checkpoint_files = sorted(glob.glob(checkpoint_pattern))
        
        # 테이블 초기화
        self.table.setRowCount(len(self.checkpoint_files))
        
        # 체크포인트 정보 로드 및 테이블 추가
        for i, file_path in enumerate(self.checkpoint_files):
            file_name = os.path.basename(file_path)
            self.table.setItem(i, 0, QTableWidgetItem(file_name))
            
            try:
                # CPU 메모리에 체크포인트 로드 (메타데이터만)
                checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
                
                # 정보 추출
                epoch = checkpoint.get('epoch', -1) + 1  # 0-based에서 1-based로 변환
                self.table.setItem(i, 1, QTableWidgetItem(str(epoch)))
                
                # 히스토리 데이터가 있는 경우
                if 'history' in checkpoint:
                    history = checkpoint['history']
                    
                    # 학습 손실
                    if 'train_loss' in history and len(history['train_loss']) > 0:
                        train_loss = history['train_loss'][-1]
                        self.table.setItem(i, 2, QTableWidgetItem(f"{train_loss:.4f}"))
                    
                    # 검증 손실
                    if 'val_loss' in history and len(history['val_loss']) > 0:
                        val_loss = history['val_loss'][-1]
                        self.table.setItem(i, 3, QTableWidgetItem(f"{val_loss:.4f}"))
                    
                    # 검증 정확도
                    if 'val_acc' in history and len(history['val_acc']) > 0:
                        val_acc = history['val_acc'][-1] * 100
                        self.table.setItem(i, 4, QTableWidgetItem(f"{val_acc:.2f}%"))
                
                # 체크포인트 정보 저장
                self.checkpoint_info[file_path] = {
                    'epoch': epoch,
                    'file_name': file_name
                }
                
                # 최고 성능 체크포인트 강조
                if 'history' in checkpoint and 'val_acc' in checkpoint['history'] and len(checkpoint['history']['val_acc']) > 0:
                    best_val_acc = max(item.get('val_acc', 0) for item in [
                        self.checkpoint_info.get(f, {'val_acc': 0}) for f in self.checkpoint_files
                    ])
                    
                    if checkpoint['history']['val_acc'][-1] == best_val_acc:
                        for col in range(self.table.columnCount()):
                            item = self.table.item(i, col)
                            if item:
                                item.setBackground(QColor(200, 255, 200))  # 밝은 녹색 배경
                
            except Exception as e:
                # 체크포인트 로드 실패 시
                self.table.setItem(i, 1, QTableWidgetItem("로드 실패"))
                for col in range(2, self.table.columnCount()):
                    self.table.setItem(i, col, QTableWidgetItem("-"))
    
    def load_selected_checkpoint(self):
        """선택된 체크포인트 로드"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "경고", "체크포인트를 선택해주세요")
            return
        
        # 선택된 행의 파일 경로 추출
        row = selected_items[0].row()
        if row < len(self.checkpoint_files):
            checkpoint_path = self.checkpoint_files[row]
            self.load_checkpoint_signal.emit(checkpoint_path)
    
    def delete_selected_checkpoint(self):
        """선택된 체크포인트 삭제"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "경고", "체크포인트를 선택해주세요")
            return
        
        # 선택된 행의 파일 경로 추출
        row = selected_items[0].row()
        if row < len(self.checkpoint_files):
            checkpoint_path = self.checkpoint_files[row]
            
            # 삭제 확인
            reply = QMessageBox.question(
                self, "체크포인트 삭제",
                f"선택한 체크포인트를 삭제하시겠습니까?\n{os.path.basename(checkpoint_path)}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    os.remove(checkpoint_path)
                    self.refresh_checkpoints()
                    QMessageBox.information(self, "정보", "체크포인트가 삭제되었습니다.")
                except Exception as e:
                    QMessageBox.critical(self, "오류", f"체크포인트 삭제 실패: {str(e)}")
    
    def import_checkpoint(self):
        """외부 체크포인트 가져오기"""
        file, _ = QFileDialog.getOpenFileName(self, "체크포인트 파일 선택", "", "PyTorch 체크포인트 (*.pth)")
        if file:
            try:
                # 체크포인트 파일 복사
                import shutil
                file_name = os.path.basename(file)
                dest_path = os.path.join(self.output_dir, file_name)
                
                # 대상 디렉토리가 없으면 생성
                os.makedirs(self.output_dir, exist_ok=True)
                
                # 파일 복사
                shutil.copy2(file, dest_path)
                
                # 체크포인트 목록 새로고침
                self.refresh_checkpoints()
                
                QMessageBox.information(self, "정보", f"체크포인트를 가져왔습니다: {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"체크포인트 가져오기 실패: {str(e)}")
    
    def export_checkpoint(self):
        """선택된 체크포인트 내보내기"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "경고", "체크포인트를 선택해주세요")
            return
        
        # 선택된 행의 파일 경로 추출
        row = selected_items[0].row()
        if row < len(self.checkpoint_files):
            checkpoint_path = self.checkpoint_files[row]
            file_name = os.path.basename(checkpoint_path)
            
            # 저장 경로 선택
            save_path, _ = QFileDialog.getSaveFileName(self, "체크포인트 저장", file_name, "PyTorch 체크포인트 (*.pth)")
            if save_path:
                try:
                    # 체크포인트 파일 복사
                    import shutil
                    shutil.copy2(checkpoint_path, save_path)
                    
                    QMessageBox.information(self, "정보", f"체크포인트를 내보냈습니다: {save_path}")
                except Exception as e:
                    QMessageBox.critical(self, "오류", f"체크포인트 내보내기 실패: {str(e)}")
    
    def get_checkpoint_settings(self):
        """체크포인트 설정 반환"""
        return {
            'auto_save': self.auto_save_cb.isChecked(),
            'save_frequency': self.save_freq_spin.value()
        }
