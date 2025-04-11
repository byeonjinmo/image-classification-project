
#!/usr/bin/env python3
"""
모델 평가를 위한 메트릭 유틸리티 모듈
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(true_labels, pred_labels, class_names, output_path):
    """
    혼동 행렬을 계산하고 시각화
    
    Args:
        true_labels (list): 실제 레이블 리스트
        pred_labels (list): 예측 레이블 리스트
        class_names (list): 클래스 이름 리스트
        output_path (str): 결과 이미지 저장 경로
    """
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return cm

def plot_roc_curves(true_labels, pred_scores, class_names, output_path):
    """
    각 클래스별 ROC 커브 및 AUC 계산
    
    Args:
        true_labels (list): 실제 레이블 리스트
        pred_scores (numpy.ndarray): 클래스별 예측 확률 (shape: [n_samples, n_classes])
        class_names (list): 클래스 이름 리스트
        output_path (str): 결과 이미지 저장 경로
    
    Returns:
        dict: 각 클래스별 및 평균 AUC 값
    """
    from itertools import cycle
    
    # 데이터 준비
    n_classes = len(class_names)
    
    # OneHotEncoding으로 변환
    true_labels_onehot = np.zeros((len(true_labels), n_classes))
    for i, label in enumerate(true_labels):
        true_labels_onehot[i, label] = 1
    
    # 각 클래스별 ROC 커브 계산
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_onehot[:, i], pred_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # micro-average ROC 커브 계산 (각 예제를 독립적으로 고려)
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_onehot.ravel(), pred_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # macro-average ROC 커브 계산 (모든 클래스의 평균)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # ROC 커브 시각화
    plt.figure(figsize=(12, 10))
    
    # 매크로 및 마이크로 평균 그리기
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4)
            
    plt.plot(fpr["macro"], tpr["macro"],
            label=f'macro-average ROC (AUC = {roc_auc["macro"]:.2f})',
            color='navy', linestyle=':', linewidth=4)
    
    # 각 클래스별 ROC 커브 그리기
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # 그래프 꾸미기
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # AUC 점수 테이블 추가
    auc_table = pd.DataFrame({
        'Class': [class_names[i] for i in range(n_classes)] + ['Micro Average', 'Macro Average'],
        'AUC': [roc_auc[i] for i in range(n_classes)] + [roc_auc['micro'], roc_auc['macro']]
    })
    
    # 테이블 형태로 드로잉
    table_ax = plt.axes([0.15, 0.02, 0.7, 0.2], frameon=True)  # [left, bottom, width, height]
    table_ax.axis('off')
    
    table = table_ax.table(
        cellText=auc_table.values.round(4),
        colLabels=auc_table.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 그래프 저장
    plt.tight_layout(rect=[0, 0.2, 1, 1])  # 테이블을 위한 공간 확보
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # AUC 테이블 CSV로도 저장
    csv_path = output_path.replace('.png', '.csv')
    auc_table.to_csv(csv_path, index=False)
    
    return roc_auc

def generate_classification_report(true_labels, pred_labels, class_names, output_path):
    """
    분류 성능 보고서 생성 및 저장
    
    Args:
        true_labels (list): 실제 레이블 리스트
        pred_labels (list): 예측 레이블 리스트
        class_names (list): 클래스 이름 리스트
        output_path (str): 결과 CSV 저장 경로
        
    Returns:
        dict: 분류 성능 보고서 딕셔너리
    """
    report = classification_report(
        true_labels, pred_labels, 
        target_names=class_names, 
        output_dict=True
    )
    
    # 보고서를 DataFrame으로 변환 후 CSV로 저장
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_path)
    
    return report
