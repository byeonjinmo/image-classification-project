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