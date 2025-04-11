# 메인 윈도우 클래스의 메서드들
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QLabel, QFont, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from .components import TrainingThread
from .utils.file_utils import open_directory

def loadData(self):
    data_dir = self.data_dir_edit.text()
    if not data_dir or not os.path.exists(data_dir):
        self.status_bar.showMessage("유효한 데이터 디렉토리를 선택하세요.")
        return
    
    # 데이터 미리보기를 위한 이미지 로드
    try:
        # 클래스 폴더 찾기
        class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        if not class_folders:
            self.status_bar.showMessage("데이터 디렉토리에 클래스 폴더가 없습니다.")
            return
        
        # UI 업데이트
        self.preview_label.setVisible(False)
        self.preview_scroll.setVisible(True)
        
        # 이전 미리보기 지우기
        for i in reversed(range(self.preview_grid.count())): 
            widget = self.preview_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # 각 클래스별 샘플 이미지 표시
        row = 0
        for class_name in class_folders:
            class_dir = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                         and os.path.isfile(os.path.join(class_dir, f))][:5]  # 각 클래스당 최대 5개
            
            if image_files:
                # 클래스 라벨 추가
                class_label = QLabel(f"클래스: {class_name}")
                class_label.setFont(QFont("Arial", 10, QFont.Bold))
                self.preview_grid.addWidget(class_label, row, 0, 1, 5)
                row += 1
                
                # 샘플 이미지 표시
                for col, img_file in enumerate(image_files):
                    img_path = os.path.join(class_dir, img_file)
                    pixmap = QPixmap(img_path).scaled(
                        150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    img_label = QLabel()
                    img_label.setPixmap(pixmap)
                    img_label.setAlignment(Qt.AlignCenter)
                    img_label.setToolTip(img_file)
                    
                    self.preview_grid.addWidget(img_label, row, col)
                
                row += 1
        
        # 데이터 요약 업데이트
        self.data_summary_label.setText(
            f"데이터: {len(class_folders)} 클래스, 디렉토리: {data_dir}"
        )
        
        # 초기화 버튼 활성화
        self.init_model_btn.setEnabled(True)
        
        self.status_bar.showMessage(f"데이터 로드 완료: {len(class_folders)} 클래스 발견")
        
    except Exception as e:
        self.status_bar.showMessage(f"데이터 로드 오류: {str(e)}")

def initializeModel(self):
    try:
        data_dir = self.data_dir_edit.text()
        output_dir = self.output_dir_edit.text()
        batch_size = self.batch_size_spin.value()
        num_epochs = self.epochs_spin.value()
        lr = self.lr_spin.value()
        val_split = self.val_split_spin.value()
        
        # 모델 타입 결정
        model_type_map = {
            "ResNet18": "resnet18",
            "ResNet34": "resnet34",
            "ResNet50": "resnet50"
        }
        model_type = model_type_map[self.model_combo.currentText()]
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 분류기 초기화
        self.classifier = ImageClassifier(
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            model_type=model_type
        )
        
        # 데이터 로드
        self.classifier.load_data(val_split=val_split)
        
        # 모델 생성
        self.classifier.create_model()
        
        # UI 업데이트
        self.model_summary_label.setText(f"모델: {model_type}")
        self.params_summary_label.setText(
            f"파라미터: 배치 크기={batch_size}, 에폭={num_epochs}, 학습률={lr}"
        )
        
        # 학습 버튼 활성화
        self.train_btn.setEnabled(True)
        
        # 결과 탭 정보 업데이트
        self.result_model_label.setText(f"모델: {model_type}")
        self.result_data_label.setText(f"데이터: {self.data_summary_label.text().split(':')[1]}")
        
        self.status_bar.showMessage(f"{model_type} 모델 초기화 완료")
        
    except Exception as e:
        self.status_bar.showMessage(f"모델 초기화 오류: {str(e)}")

def startTraining(self):
    if not self.classifier:
        self.status_bar.showMessage("모델이 초기화되지 않았습니다.")
        return
    
    # UI 업데이트
    self.train_btn.setEnabled(False)
    self.stop_btn.setEnabled(True)
    self.init_model_btn.setEnabled(False)
    self.tabs.setTabEnabled(0, False)  # 설정 탭 비활성화
    
    # 에폭 진행 상황 초기화
    self.epoch_label.setText(f"에폭: 0/{self.classifier.num_epochs}")
    self.epoch_progress.setValue(0)
    
    # 그래프 초기화
    self.train_loss_curve.setData([], [])
    self.val_loss_curve.setData([], [])
    self.train_acc_curve.setData([], [])
    self.val_acc_curve.setData([], [])
    
    # 뉴런 시각화 시작
    self.neuron_viz.startVisualization()
    
    # 학습 시작 시간 기록
    self.training_start_time = time.time()
    
    # 학습 쓰레드 시작
    self.training_thread = TrainingThread(self.classifier, self.classifier.num_epochs)
    self.training_thread.update_progress.connect(self.updateTrainingProgress)
    self.training_thread.update_batch.connect(self.updateBatchProgress)
    self.training_thread.finished.connect(self.trainingFinished)
    self.training_thread.start()
    
    self.status_bar.showMessage("학습 시작...")

def updateTrainingProgress(self, epoch, train_loss, train_acc, val_loss, val_acc):
    # 에폭 진행 상황 업데이트
    self.epoch_label.setText(f"에폭: {epoch}/{self.classifier.num_epochs}")
    progress = int(epoch / self.classifier.num_epochs * 100)
    self.epoch_progress.setValue(progress)
    
    # 손실 및 정확도 그래프 업데이트
    epochs = list(range(1, epoch + 1))
    
    train_losses = self.classifier.history['train_loss']
    val_losses = self.classifier.history['val_loss']
    train_accs = self.classifier.history['train_acc']
    val_accs = self.classifier.history['val_acc']
    
    self.train_loss_curve.setData(epochs, train_losses)
    self.val_loss_curve.setData(epochs, val_losses)
    self.train_acc_curve.setData(epochs, train_accs)
    self.val_acc_curve.setData(epochs, val_accs)
    
    # 상태 업데이트
    self.status_bar.showMessage(
        f"에폭 {epoch}/{self.classifier.num_epochs} - "
        f"학습 손실: {train_loss:.4f}, 학습 정확도: {train_acc:.4f}, "
        f"검증 손실: {val_loss:.4f}, 검증 정확도: {val_acc:.4f}"
    )

def updateBatchProgress(self, epoch, batch_idx, batch_progress):
    # 배치 진행 상황 업데이트
    self.batch_label.setText(f"배치: {batch_idx+1}/{len(self.classifier.train_loader)}")
    self.batch_progress.setValue(int(batch_progress))

def stopTraining(self):
    if self.training_thread and self.training_thread.isRunning():
        self.training_thread.stop()
        self.status_bar.showMessage("학습 중단 중...")

def trainingFinished(self, history=None):
    # 뉴런 시각화 중지
    self.neuron_viz.stopVisualization()
    
    # UI 업데이트
    self.train_btn.setEnabled(True)
    self.stop_btn.setEnabled(False)
    self.init_model_btn.setEnabled(True)
    self.tabs.setTabEnabled(0, True)  # 설정 탭 활성화
    
    # 학습이 정상적으로 완료된 경우
    if history is not None:
        # 학습 종료 시간 계산
        training_time = time.time() - self.training_start_time
        training_time_str = f"{training_time//60:.0f}분 {training_time%60:.0f}초"
        
        self.status_bar.showMessage("학습 완료!")
        
        # 결과 탭 업데이트
        self.result_time_label.setText(f"학습 시간: {training_time_str}")
        
        # 최종 정확도 표시
        last_val_acc = self.classifier.history['val_acc'][-1] if self.classifier.history['val_acc'] else 0
        self.result_accuracy_label.setText(f"최종 정확도: {last_val_acc:.4f}")
        
        # 결과 시각화 생성
        self.updateResultsVisualization()
        
        # 결과 저장 버튼 활성화
        self.save_results_btn.setEnabled(True)
        self.open_results_dir_btn.setEnabled(True)
        
        # 탭 전환
        self.tabs.setCurrentIndex(2)  # 결과 탭으로 전환
    else:
        self.status_bar.showMessage("학습이 중단되었습니다.")

def updateResultsVisualization(self):
    if not self.classifier:
        return
    
    try:
        # 결과 폴더에서 이미지 파일 로드
        output_dir = self.classifier.output_dir
        
        # 혼동 행렬
        conf_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
        if os.path.exists(conf_matrix_path):
            self.conf_matrix_fig.clear()
            ax = self.conf_matrix_fig.add_subplot(111)
            img = plt.imread(conf_matrix_path)
            ax.imshow(img)
            ax.axis('off')
            self.conf_matrix_canvas.draw()
        
        # ROC 커브
        roc_curves_path = os.path.join(output_dir, 'roc_curves.png')
        if os.path.exists(roc_curves_path):
            self.roc_fig.clear()
            ax = self.roc_fig.add_subplot(111)
            img = plt.imread(roc_curves_path)
            ax.imshow(img)
            ax.axis('off')
            self.roc_canvas.draw()
        
        # t-SNE 시각화
        tsne_path = os.path.join(output_dir, 'tsne_visualization.png')
        if os.path.exists(tsne_path):
            self.tsne_fig.clear()
            ax = self.tsne_fig.add_subplot(111)
            img = plt.imread(tsne_path)
            ax.imshow(img)
            ax.axis('off')
            self.tsne_canvas.draw()
        
        # 클래스 활성화 맵
        cam_path = os.path.join(output_dir, 'class_activation_maps.png')
        if os.path.exists(cam_path):
            self.cam_fig.clear()
            ax = self.cam_fig.add_subplot(111)
            img = plt.imread(cam_path)
            ax.imshow(img)
            ax.axis('off')
            self.cam_canvas.draw()
        
    except Exception as e:
        self.status_bar.showMessage(f"결과 시각화 오류: {str(e)}")

def saveResults(self):
    if not self.classifier:
        return
    
    try:
        # 최종 모델 저장
        self.classifier.save_model("final_model.pth")
        self.status_bar.showMessage(f"모델 및 결과가 {self.classifier.output_dir}에 저장되었습니다.")
    except Exception as e:
        self.status_bar.showMessage(f"결과 저장 오류: {str(e)}")

def openResultsDir(self):
    if not self.classifier:
        return
    
    # 결과 폴더 열기
    output_dir = os.path.abspath(self.classifier.output_dir)
    if open_directory(output_dir):
        self.status_bar.showMessage(f"결과 폴더 열기: {output_dir}")
    else:
        self.status_bar.showMessage(f"결과 폴더를 열 수 없습니다: {output_dir}")

def runPrediction(self):
    model_path = self.predict_model_path.text()
    image_path = self.predict_image_path.text()
    
    if not model_path or not os.path.exists(model_path):
        self.status_bar.showMessage("유효한 모델 파일을 선택하세요.")
        return
    
    if not image_path or not os.path.exists(image_path):
        self.status_bar.showMessage("예측할 이미지 파일을 선택하세요.")
        return
    
    try:
        # 출력 디렉토리 설정
        output_dir = os.path.join(os.path.dirname(model_path), "predictions")
        os.makedirs(output_dir, exist_ok=True)
        
        # 데이터 디렉토리 추정
        if self.classifier and self.classifier.data_dir:
            data_dir = self.classifier.data_dir
        else:
            data_dir = os.path.dirname(os.path.dirname(model_path))
        
        # 임시 분류기 생성
        temp_classifier = ImageClassifier(
            data_dir=data_dir,
            output_dir=output_dir
        )
        
        # 모델 로드
        temp_classifier.load_model(model_path)
        
        # 예측 실행
        result = temp_classifier.predict(image_path)
        
        # 결과 표시
        self.prediction_label.setText(f"예측 클래스: {result['class']}")
        self.confidence_label.setText(f"신뢰도: {result['probability']:.2%}")
        
        # 신뢰도 바 차트 업데이트
        self.confidence_chart.clear()
        
        # 클래스 이름과 확률 추출
        class_names = list(result['all_probabilities'].keys())
        probabilities = list(result['all_probabilities'].values())
        
        # 내림차순 정렬
        sorted_data = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)
        class_names, probabilities = zip(*sorted_data)
        
        x = np.arange(len(class_names))
        y = np.array(probabilities)
        
        # 색상 설정
        colors = [(0, 0, 255, 100) if i > 0 else (255, 0, 0, 200) for i in range(len(class_names))]
        
        # 바 차트 생성
        bar_chart = pg.BarGraphItem(x=x, height=y, width=0.6, brush=colors)
        self.confidence_chart.addItem(bar_chart)
        
        # 축 설정
        ax = self.confidence_chart.getAxis('bottom')
        ax.setTicks([[(i, name) for i, name in enumerate(class_names)]])
        
        self.status_bar.showMessage(f"예측 완료: {result['class']} (신뢰도: {result['probability']:.2%})")
        
    except Exception as e:
        self.status_bar.showMessage(f"예측 오류: {str(e)}")

def openDataFolder(self):
    self.browseDataDir()

def saveModel(self):
    if not self.classifier:
        self.status_bar.showMessage("저장할 모델이 없습니다.")
        return
    
    file, _ = QFileDialog.getSaveFileName(self, "모델 저장", "", "PyTorch 모델 (*.pth)")
    if file:
        try:
            self.classifier.save_model(file)
            self.status_bar.showMessage(f"모델을 {file}에 저장했습니다.")
        except Exception as e:
            self.status_bar.showMessage(f"모델 저장 오류: {str(e)}")

def loadModel(self):
    file, _ = QFileDialog.getOpenFileName(self, "모델 불러오기", "", "PyTorch 모델 (*.pth)")
    if file:
        try:
            # 데이터 디렉토리 필요
            if not self.data_dir_edit.text():
                self.status_bar.showMessage("먼저 데이터 디렉토리를 선택하세요.")
                return
            
            data_dir = self.data_dir_edit.text()
            output_dir = self.output_dir_edit.text()
            
            # 분류기 초기화
            self.classifier = ImageClassifier(
                data_dir=data_dir,
                output_dir=output_dir
            )
            
            # 데이터 로드
            self.classifier.load_data()
            
            # 모델 로드
            self.classifier.load_model(file)
            
            # UI 업데이트
            self.model_summary_label.setText(f"모델: {self.classifier.model_type} (로드됨)")
            self.train_btn.setEnabled(True)
            self.predict_btn.setEnabled(True)
            
            self.status_bar.showMessage(f"모델을 {file}에서 불러왔습니다.")
            
        except Exception as e:
            self.status_bar.showMessage(f"모델 로드 오류: {str(e)}")

def showAbout(self):
    about_text = """
    <h2>딥러닝 이미지 분류기 UI</h2>
    <p>이 프로그램은 딥러닝 기반 이미지 분류 모델의 학습 및 평가를 위한 UI입니다.</p>
    <p>ResNet 기반 모델을 사용하여 다양한 이미지 분류 작업을 수행할 수 있습니다.</p>
    <p>개발자: AI 연구자</p>
    <p>버전: 1.0.0</p>
    """
    
    QMessageBox.about(self, "프로그램 정보", about_text)
