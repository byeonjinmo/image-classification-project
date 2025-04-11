#!/usr/bin/env python3
import os
import sys
import time
from PyQt5.QtWidgets import QMainWindow
from image_classifier import ImageClassifier

# 메인 윈도우 클래스와 메서드들을 연결
def init_methods(MainWindowClass):
    """MainWindow 클래스에 메서드들을 추가합니다."""
    from .main_window_methods import (
        loadData, initializeModel, startTraining, updateTrainingProgress,
        updateBatchProgress, stopTraining, trainingFinished, updateResultsVisualization,
        saveResults, openResultsDir, runPrediction, openDataFolder, saveModel, 
        loadModel, showAbout
    )
    
    # 메서드 추가
    MainWindowClass.loadData = loadData
    MainWindowClass.initializeModel = initializeModel
    MainWindowClass.startTraining = startTraining
    MainWindowClass.updateTrainingProgress = updateTrainingProgress
    MainWindowClass.updateBatchProgress = updateBatchProgress
    MainWindowClass.stopTraining = stopTraining
    MainWindowClass.trainingFinished = trainingFinished
    MainWindowClass.updateResultsVisualization = updateResultsVisualization
    MainWindowClass.saveResults = saveResults
    MainWindowClass.openResultsDir = openResultsDir
    MainWindowClass.runPrediction = runPrediction
    MainWindowClass.openDataFolder = openDataFolder
    MainWindowClass.saveModel = saveModel
    MainWindowClass.loadModel = loadModel
    MainWindowClass.showAbout = showAbout
    
    return MainWindowClass
