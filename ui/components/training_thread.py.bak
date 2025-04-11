#!/usr/bin/env python3
import os
import torch
from PyQt5.QtCore import QThread, pyqtSignal
import time

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
