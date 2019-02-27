import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc_curve(y_score, y_test):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)
    # Plot ROC curve
    plt.figure(figsize=(16, 12))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (1 - Specificity)', size=16)
    plt.ylabel('True Positive Rate (Sensitivity)', size=16)
    plt.title('ROC Curve', size=20)
    plt.legend(fontsize=14)
